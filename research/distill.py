"""FIT-only distillation of the selected three-GRU ensemble."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pyarrow.parquet as pq
import torch

from advance import OUT, predict_sequence
from global_wp import GlobalWP
from group_cv import folds
from pilot import (CACHE, N_FEATURES, N_ROWS, ROOT, VALID_BYTES, VALID_ID_DIGEST, WARMUP,
                   check_train, make_gru, sequence_id_digest)
from scale import DEV, FIT, GRU_BATCH, GRU_CHUNK, read_group, weighted_loss

EPOCHS = 5
ALPHA = 0.5
CACHE_BATCH = 8
BLEND_NAMES = ("student", "student_norm50", "student_seed17_50",
               "student_seed37_50", "student_teacher50")


def load_teacher(fold):
    folder = (CACHE / "improve/full_fit" if fold == "full"
              else CACHE / f"improve/fold_{fold}")
    with np.load(folder / "gru96_norm5ep/gru_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    models = []
    for name in ("gru96_norm5ep", "gru96_seed17", "gru96_seed37"):
        model = make_gru(96)
        model.load_state_dict(torch.load(folder / name / "scaled_gru.pt",
                                         map_location="cpu", weights_only=True))
        models.append(model.eval())
    return mean, std, models


def distill_folder(fold):
    return OUT / ("full_fit/distill" if fold == "full" else f"fold_{fold}/distill")


def predict_batch(model, values):
    hidden, predictions = None, []
    for start in range(0, N_ROWS, GRU_CHUNK):
        output, hidden = model(torch.from_numpy(
            np.ascontiguousarray(values[:, start:start + GRU_CHUNK])), hidden)
        predictions.append(output.numpy())
    return np.concatenate(predictions, axis=1)


def teacher_predict_batch(models, mean, std, raw):
    normalized = np.clip((raw - mean) / std, -10, 10)
    return (predict_batch(models[0], normalized) + predict_batch(models[1], raw)
            + predict_batch(models[2], raw)) / 3


def teacher_predict(models, mean, std, raw):
    return teacher_predict_batch(models, mean, std, raw[None])[0]


def cache_teacher(parquet, fold, fit):
    folder = distill_folder(fold)
    folder.mkdir(parents=True, exist_ok=True)
    artifact = folder / "teacher.npy"
    groups_file = folder / "groups.npy"
    if artifact.exists() and groups_file.exists():
        if np.array_equal(np.load(groups_file), fit):
            return np.load(artifact, mmap_mode="r")
        raise ValueError("cached teacher groups differ from fixed fold")
    temporary = folder / "teacher.tmp.npy"
    teacher = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16,
                                        shape=(len(fit), N_ROWS, 2))
    mean, std, models = load_teacher(fold)
    with torch.no_grad():
        for offset in range(0, len(fit), CACHE_BATCH):
            end = min(offset + CACHE_BATCH, len(fit))
            raw = np.empty((end - offset, N_ROWS, N_FEATURES), np.float32)
            for index, group in enumerate(fit[offset:end]):
                raw[index], _ = read_group(parquet, group)
            teacher[offset:end] = teacher_predict_batch(models, mean, std, raw)
            if end // 100 > offset // 100 or end == len(fit):
                print("teacher fold", fold, "cached", end, flush=True)
    teacher.flush()
    del teacher
    temporary.replace(artifact)
    np.save(groups_file, fit)
    return np.load(artifact, mmap_mode="r")


def train(parquet, fold, fit, alpha=ALPHA, suffix=""):
    folder = distill_folder(fold)
    artifact = folder / f"model{suffix}.pt"
    if artifact.exists():
        print("using completed", artifact, flush=True)
        return
    teacher = cache_teacher(parquet, fold, fit)
    mean, std, _ = load_teacher(fold)
    torch.manual_seed(57)
    rng = np.random.default_rng(57)
    model = make_gru(96)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(EPOCHS):
        if epoch == 3:
            for group in optimizer.param_groups:
                group["lr"] = 0.0003
        losses = []
        order = rng.permutation(len(fit))
        for offset in range(0, len(fit), GRU_BATCH):
            indices = order[offset:offset + GRU_BATCH]
            x = np.empty((len(indices), N_ROWS, N_FEATURES), np.float32)
            y = np.empty((len(indices), N_ROWS, 2), np.float32)
            for j, index in enumerate(indices):
                x[j], y[j] = read_group(parquet, fit[index])
            x = np.clip((x - mean) / std, -10, 10)
            t = np.asarray(teacher[indices], np.float32)
            hidden = None
            for start in range(0, N_ROWS, GRU_CHUNK):
                end = min(start + GRU_CHUNK, N_ROWS)
                xb = torch.from_numpy(np.ascontiguousarray(x[:, start:end]))
                yb = torch.from_numpy(np.clip(np.ascontiguousarray(y[:, start:end]), -2, 2))
                tb = torch.from_numpy(np.ascontiguousarray(t[:, start:end]))
                optimizer.zero_grad()
                prediction, hidden = model(xb, hidden)
                skip = WARMUP if start == 0 else 0
                label_loss = weighted_loss(prediction[:, skip:], yb[:, skip:])
                mimic_loss = (((prediction[:, skip:] - tb[:, skip:]).square()
                               * yb[:, skip:].abs()).mean())
                loss = (1 - alpha) * label_loss + alpha * mimic_loss
                loss.backward()
                optimizer.step()
                hidden = hidden.detach()
                losses.append(loss.item())
            if offset + len(indices) == len(fit):
                print("distill fold", fold, "epoch", epoch + 1,
                      "loss", float(np.mean(losses)), flush=True)
    torch.save(model.state_dict(), artifact)


def score(parquet, fold, validation, alpha=ALPHA, suffix=""):
    folder = distill_folder(fold)
    mean, std, teachers = load_teacher(fold)
    student = make_gru(96)
    student.load_state_dict(torch.load(folder / f"model{suffix}.pt", map_location="cpu",
                                       weights_only=True))
    student.eval()
    metrics = {name: GlobalWP() for name in (("student",) if suffix else
                                              ("teacher", "student"))}
    bins = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.000001))
    errors = {name: np.zeros((2, len(bins), 3), np.float64)
              for name in metrics}
    required = np.arange(N_ROWS) >= WARMUP
    with torch.no_grad():
        for n, group in enumerate(validation, 1):
            raw, y = read_group(parquet, group)
            clipped_y = np.clip(y, -2, 2)
            predictions = {
                "student": predict_sequence(student, np.clip((raw - mean) / std, -10, 10)),
            }
            if not suffix:
                predictions["teacher"] = teacher_predict(teachers, mean, std, raw)
            for name, values in predictions.items():
                metrics[name].add(y, values, required)
                clipped_p = np.clip(values, -2, 2)
                for target in range(2):
                    magnitude = np.abs(clipped_y[:, target])
                    for index, (low, high) in enumerate(bins):
                        mask = required & (magnitude > low) & (magnitude <= high)
                        weight = magnitude[mask].astype(np.float64)
                        error = (clipped_p[mask, target] - clipped_y[mask, target]).astype(np.float64)
                        errors[name][target, index] += (weight.sum(),
                                                       np.sum(weight * error * error),
                                                       mask.sum())
            if n % 100 == 0:
                print("distill fold", fold, "scored", n, flush=True)
    result = {"alpha": alpha,
              "scores": {name: metric.result() for name, metric in metrics.items()},
              "weighted_mse_by_abs_target": {}}
    for name, values in errors.items():
        result["weighted_mse_by_abs_target"][name] = {
            f"t{target}": [{"range": [low, high], "weighted_mse": float(values[target, i, 1]
                            / values[target, i, 0]), "rows": int(values[target, i, 2])}
                           for i, (low, high) in enumerate(bins)]
            for target in range(2)}
    (folder / f"score{suffix}.json").write_text(json.dumps(result, indent=2) + "\n")
    print("distill fold", fold, {name: value["weighted_pearson"]
                                  for name, value in result["scores"].items()}, flush=True)


def summarize():
    results = [json.loads((OUT / f"fold_{fold}/distill/score.json").read_text())
               for fold in range(1, 4)]
    summary = {name: {"folds": [result["scores"][name]["weighted_pearson"]
                                 for result in results]}
               for name in ("teacher", "student")}
    for value in summary.values():
        value["mean"] = float(np.mean(value["folds"]))
    summary["student"]["gain_vs_single"] = summary["student"]["mean"] - 0.415425098635339
    summary["student"]["gap_vs_teacher"] = (summary["student"]["mean"]
                                                - summary["teacher"]["mean"])
    output = {"scores": summary,
              "weighted_mse_by_abs_target": [result["weighted_mse_by_abs_target"]
                                                for result in results]}
    (OUT / "distill_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


def summarize_alpha75():
    scores = [json.loads((OUT / f"fold_{fold}/distill/score_alpha75.json").read_text())
              ["scores"]["student"]["weighted_pearson"] for fold in range(1, 4)]
    baseline = json.loads((OUT / "distill_summary.json").read_text())["scores"]["student"]
    output = {"alpha": 0.75, "folds": scores, "mean": float(np.mean(scores)),
              "gain_vs_alpha50": float(np.mean(scores) - baseline["mean"])}
    (OUT / "alpha75_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


def score_alpha75_masks(parquet, fold, validation):
    folder = distill_folder(fold)
    mean, std, _ = load_teacher(fold)
    model = make_gru(96)
    model.load_state_dict(torch.load(folder / "model_alpha75.pt", map_location="cpu",
                                     weights_only=True))
    model.eval()
    names = ("all", "early", "late", "random20")
    scores = {name: GlobalWP() for name in names}
    required = np.arange(N_ROWS) >= WARMUP
    steps = np.arange(N_ROWS)
    with torch.no_grad():
        for offset in range(0, len(validation), CACHE_BATCH):
            batch = validation[offset:offset + CACHE_BATCH]
            raw = np.empty((len(batch), N_ROWS, N_FEATURES), np.float32)
            targets = []
            for index, group in enumerate(batch):
                raw[index], y = read_group(parquet, group)
                targets.append(y)
            values = predict_batch(model, np.clip((raw - mean) / std, -10, 10))
            for index, (group, y) in enumerate(zip(batch, targets)):
                selected = {"all": required, "early": required & (steps < 10_000),
                            "late": steps >= 10_000,
                            "random20": required & (np.random.default_rng(
                                20260917 + int(group)).random(N_ROWS) < 0.2)}
                for name, mask in selected.items():
                    scores[name].add(y, values[index], mask)
    result = {name: metric.result() for name, metric in scores.items()}
    (folder / "alpha75_masks.json").write_text(json.dumps(result, indent=2) + "\n")
    print("alpha75 masks fold", fold,
          {name: value["weighted_pearson"] for name, value in result.items()}, flush=True)


def summarize_alpha75_masks():
    results = [json.loads((OUT / f"fold_{fold}/distill/alpha75_masks.json").read_text())
               for fold in range(1, 4)]
    baseline = json.loads((OUT / "blend_summary.json").read_text())["scores"]["student"]
    output = {}
    for mask in ("all", "early", "late", "random20"):
        values = [result[mask]["weighted_pearson"] for result in results]
        output[mask] = {"folds": values, "mean": float(np.mean(values)),
                        "gain_vs_alpha50": float(np.mean(values)
                                                   - baseline[f"{mask}_mean"])}
    output["passes"] = all(output[mask]["gain_vs_alpha50"] >= 0
                           for mask in ("all", "early", "late", "random20"))
    (OUT / "alpha75_masks_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


def score_blends(parquet, fold, validation):
    """Screen a fixed, small set of two-GRU blends on FIT-CV only."""
    folder = distill_folder(fold)
    mean, std, teachers = load_teacher(fold)
    student = make_gru(96)
    student.load_state_dict(torch.load(folder / "model.pt", map_location="cpu",
                                       weights_only=True))
    student.eval()
    masks = ("all", "early", "late", "random20")
    scores = {name: {mask: GlobalWP() for mask in masks} for name in BLEND_NAMES}
    required = np.arange(N_ROWS) >= WARMUP
    steps = np.arange(N_ROWS)
    with torch.no_grad():
        for offset in range(0, len(validation), CACHE_BATCH):
            batch = validation[offset:offset + CACHE_BATCH]
            raw = np.empty((len(batch), N_ROWS, N_FEATURES), np.float32)
            targets = []
            for index, group in enumerate(batch):
                raw[index], y = read_group(parquet, group)
                targets.append(y)
            normalized = np.clip((raw - mean) / std, -10, 10)
            student_values = predict_batch(student, normalized)
            norm_values = predict_batch(teachers[0], normalized)
            seed17_values = predict_batch(teachers[1], raw)
            seed37_values = predict_batch(teachers[2], raw)
            teacher_values = (norm_values + seed17_values + seed37_values) / 3
            predictions = {
                "student": student_values,
                "student_norm50": (student_values + norm_values) / 2,
                "student_seed17_50": (student_values + seed17_values) / 2,
                "student_seed37_50": (student_values + seed37_values) / 2,
                "student_teacher50": (student_values + teacher_values) / 2,
            }
            for index, (group, y) in enumerate(zip(batch, targets)):
                selected = {
                    "all": required,
                    "early": required & (steps < 10_000),
                    "late": steps >= 10_000,
                    "random20": required & (np.random.default_rng(
                        20260917 + int(group)).random(N_ROWS) < 0.2),
                }
                for name, values in predictions.items():
                    for mask, chosen in selected.items():
                        scores[name][mask].add(y, values[index], chosen)
            end = offset + len(batch)
            if end // 100 > offset // 100 or end == len(validation):
                print("blend fold", fold, "scored", end, flush=True)
    result = {name: {mask: metric.result() for mask, metric in by_mask.items()}
              for name, by_mask in scores.items()}
    (folder / "blend_score.json").write_text(json.dumps(result, indent=2) + "\n")
    print("blend fold", fold,
          {name: value["all"]["weighted_pearson"] for name, value in result.items()},
          flush=True)


def summarize_blends():
    results = [json.loads((OUT / f"fold_{fold}/distill/blend_score.json").read_text())
               for fold in range(1, 4)]
    masks = ("all", "early", "late", "random20")
    summary = {}
    for name in BLEND_NAMES:
        summary[name] = {
            "folds": [result[name]["all"]["weighted_pearson"] for result in results],
            **{f"{mask}_mean": float(np.mean([
                result[name][mask]["weighted_pearson"] for result in results]))
               for mask in masks},
        }
    student = summary["student"]
    for name, value in summary.items():
        value["gain_vs_student"] = value["all_mean"] - student["all_mean"]
        value["passes"] = (name != "student"
                           and value["gain_vs_student"] >= 0.002
                           and all(value[f"{mask}_mean"] >= student[f"{mask}_mean"]
                                   for mask in ("early", "late", "random20")))
    output = {"selection_rule": (
                  "FIT-CV all-row gain >= 0.002 over student and no regression in "
                  "early, late, or deterministic random20 means"),
              "deployment_eligible": [name for name in BLEND_NAMES
                                       if summary[name]["passes"]
                                       and name != "student_teacher50"],
              "scores": summary}
    (OUT / "blend_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


def score_full(parquet, mode, suffix=""):
    folder = distill_folder("full")
    mean, std, teachers = load_teacher("full")
    student = make_gru(96)
    student.load_state_dict(torch.load(folder / f"model{suffix}.pt", map_location="cpu",
                                       weights_only=True))
    student.eval()
    scores = {name: GlobalWP() for name in (("student",) if suffix else
                                            ("teacher", "student"))}
    required = np.arange(N_ROWS) >= WARMUP
    seen, selected_rows = set(), 0
    with torch.no_grad():
        groups = list(range(parquet.num_row_groups) if mode == "oos" else DEV)
        for offset in range(0, len(groups), CACHE_BATCH):
            batch = groups[offset:offset + CACHE_BATCH]
            raw = np.empty((len(batch), N_ROWS, N_FEATURES), np.float32)
            targets, masks = [], []
            for index, group in enumerate(batch):
                if mode == "oos":
                    from oos_global import read_group as read_valid_group
                    seq, need, raw[index], y, mask = read_valid_group(parquet, int(group))
                    if seq in seen or not np.array_equal(need, required):
                        raise ValueError("duplicate OOS sequence or changed prediction mask")
                    seen.add(seq)
                else:
                    raw[index], y = read_group(parquet, group)
                    mask = required
                targets.append(y)
                masks.append(mask)
                selected_rows += int(mask.sum())
            values = {"student": predict_batch(
                student, np.clip((raw - mean) / std, -10, 10))}
            if not suffix:
                values["teacher"] = teacher_predict_batch(teachers, mean, std, raw)
            for index, (y, mask) in enumerate(zip(targets, masks)):
                for name, prediction in values.items():
                    scores[name].add(y, prediction[index], mask)
            end = offset + len(batch)
            if end // 200 > offset // 200:
                print("distill", mode, "scored", end, flush=True)
    if mode == "oos" and (len(seen) != 1_873 or selected_rows != 3_528_273):
        raise ValueError("OOS sequence count or official mask changed")
    if mode == "dev" and selected_rows != len(DEV) * (N_ROWS - WARMUP):
        raise ValueError("DEV selected row count changed")
    result = {"split": mode, "selected_rows": selected_rows,
              "scores": {name: metric.result() for name, metric in scores.items()}}
    (folder / f"{mode}_scores{suffix}.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print("distill", mode, {name: value["weighted_pearson"]
                            for name, value in result["scores"].items()}, flush=True)


def selfcheck():
    prediction = torch.tensor([[[1.0, -1.0]]])
    target = torch.tensor([[[0.5, -0.5]]])
    teacher = torch.tensor([[[0.75, -0.75]]])
    label = weighted_loss(prediction, target)
    mimic = ((prediction - teacher).square() * target.abs()).mean()
    assert np.isclose(((1 - ALPHA) * label + ALPHA * mimic).item(), 0.078125)
    model = make_gru(4).eval()
    values = np.zeros((2, N_ROWS, N_FEATURES), np.float32)
    with torch.no_grad():
        batched = predict_batch(model, values)
        single = predict_sequence(model, values[0])
    assert batched.shape == (2, N_ROWS, 2) and np.allclose(batched[0], single)
    assert BLEND_NAMES[:4] == ("student", "student_norm50", "student_seed17_50",
                              "student_seed37_50")
    print("distillation loss check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("selfcheck", "run", "summary", "blend",
                                            "blend_summary", "alpha75", "alpha75_summary",
                                            "alpha75_masks", "alpha75_masks_summary", "full",
                                            "full_alpha75", "score_dev", "score_dev_alpha75",
                                            "score_oos", "score_oos_alpha75"))
    parser.add_argument("--fold", type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    if args.command == "selfcheck":
        selfcheck()
        return
    if args.command == "summary":
        summarize()
        return
    if args.command == "blend_summary":
        summarize_blends()
        return
    if args.command == "alpha75_summary":
        summarize_alpha75()
        return
    if args.command == "alpha75_masks_summary":
        summarize_alpha75_masks()
        return
    torch.set_num_threads(3 if args.command in ("full", "full_alpha75") else 1)
    if args.command in ("score_oos", "score_oos_alpha75"):
        parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
        if (parquet.num_row_groups != 1_873
                or (ROOT / "datasets/valid.parquet").stat().st_size != VALID_BYTES
                or sequence_id_digest(parquet) != VALID_ID_DIGEST):
            raise ValueError("official validation file changed")
        score_full(parquet, "oos", "_alpha75" if args.command.endswith("alpha75") else "")
        return
    parquet = pq.ParquetFile(ROOT / "datasets/train.parquet")
    check_train(parquet)
    if args.command == "full":
        train(parquet, "full", FIT)
        return
    if args.command == "full_alpha75":
        train(parquet, "full", FIT, alpha=0.75, suffix="_alpha75")
        return
    if args.command in ("score_dev", "score_dev_alpha75"):
        score_full(parquet, "dev", "_alpha75" if args.command.endswith("alpha75") else "")
        return
    for fold, fit, validation in folds():
        if args.fold is None or args.fold == fold:
            if args.command == "blend":
                score_blends(parquet, fold, validation)
            elif args.command == "alpha75":
                train(parquet, fold, fit, alpha=0.75, suffix="_alpha75")
                score(parquet, fold, validation, alpha=0.75, suffix="_alpha75")
            elif args.command == "alpha75_masks":
                score_alpha75_masks(parquet, fold, validation)
            else:
                train(parquet, fold, fit)
                score(parquet, fold, validation)


if __name__ == "__main__":
    main()
