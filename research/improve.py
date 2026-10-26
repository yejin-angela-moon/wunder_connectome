"""Sequential FIT-only improvement screens on the existing fixed group folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from global_wp import GlobalWP
from group_cv import folds, selfcheck as check_folds
from pilot import (CACHE, ROOT, N_ROWS, N_FEATURES, WARMUP, VALID_BYTES, VALID_ID_DIGEST,
                   check_train, gru_predict, make_gru, make_mlp, sequence_id_digest)
from scale import DEV, FIT, history_features, lag_features, read_group, train_gru, train_mlp, train_ridge

OUT = CACHE / "improve"
SEEDS = (7, 17, 37)
WIDTHS = (64, 96)


def gru_file(fold, width, seed):
    if seed == 7:
        return CACHE / "group_cv" / f"fold_{fold}" / ("gru96" if width == 96 else "gru") / "scaled_gru.pt"
    return OUT / f"fold_{fold}" / f"gru{width}_seed{seed}" / "scaled_gru.pt"


def stage1(parquet, number, fit, validation):
    folder = OUT / f"fold_{number}"
    folder.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS[1:]:
        for width in WIDTHS:
            artifact = gru_file(number, width, seed)
            artifact.parent.mkdir(parents=True, exist_ok=True)
            if artifact.exists():
                continue
            print("train", number, width, seed, flush=True)
            train_gru(parquet, fit, artifact.parent, hidden_size=width, seed=seed)
    models = {}
    for seed in SEEDS:
        for width in WIDTHS:
            path = gru_file(number, width, seed)
            model = make_gru(width)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            models[f"gru{width}_seed{seed}"] = model.eval()
    scores = {name: GlobalWP() for name in models}
    mask = np.arange(N_ROWS) >= WARMUP
    with torch.no_grad():
        for n, group in enumerate(validation, 1):
            x, y = read_group(parquet, group)
            for name, model in models.items():
                scores[name].add(y, gru_predict(model, x), mask)
            if n % 100 == 0:
                print("stage1 scored", number, n, flush=True)
    result = {name: acc.result() for name, acc in scores.items()}
    (folder / "stage1.json").write_text(json.dumps(result, indent=2) + "\n")
    print("stage1 fold", number, {key: round(value["weighted_pearson"], 6)
                                  for key, value in result.items()}, flush=True)


def summary_stage1():
    scores = [json.loads((OUT / f"fold_{i}" / "stage1.json").read_text()) for i in range(1, 4)]
    result = {}
    for name in scores[0]:
        values = [fold[name]["weighted_pearson"] for fold in scores]
        result[name] = {"folds": values, "mean": float(np.mean(values))}
    for seed in SEEDS:
        narrow = result[f"gru64_seed{seed}"]["mean"]
        wide = result[f"gru96_seed{seed}"]["mean"]
        result[f"width_gain_seed{seed}"] = wide - narrow
    (OUT / "stage1_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def stage2(parquet, number, fit, validation, name):
    options = {
        "gru96_norm": {"standardize": True},
        "gru96_5ep_fixedlr": {"epochs": 5},
        "gru96_5ep": {"epochs": 5, "decay_after": 3},
        "gru96_norm5ep": {"standardize": True, "epochs": 5, "decay_after": 3},
        "gru96_corr": {"corr_weight": 0.05},
    }
    folder = OUT / f"fold_{number}" / name
    folder.mkdir(parents=True, exist_ok=True)
    artifact = folder / "scaled_gru.pt"
    if not artifact.exists():
        print("train", number, name, flush=True)
        train_gru(parquet, fit, folder, hidden_size=96, **options[name])
    model = make_gru(96)
    model.load_state_dict(torch.load(artifact, map_location="cpu", weights_only=True))
    model.eval()
    standardizer = None
    if options[name].get("standardize"):
        with np.load(folder / "gru_standardizer.npz") as source:
            standardizer = source["mean"], source["std"]
    scores = GlobalWP()
    mask = np.arange(N_ROWS) >= WARMUP
    with torch.no_grad():
        for n, group in enumerate(validation, 1):
            x, y = read_group(parquet, group)
            if standardizer is not None:
                x = np.clip((x - standardizer[0]) / standardizer[1], -10, 10)
            scores.add(y, gru_predict(model, x), mask)
            if n % 100 == 0:
                print("scored", number, name, n, flush=True)
    result = scores.result()
    (folder / "score.json").write_text(json.dumps(result, indent=2) + "\n")
    print("stage2 fold", number, name, result["weighted_pearson"], flush=True)


def summary_stage2(name):
    values = [json.loads((OUT / f"fold_{i}" / name / "score.json").read_text())["weighted_pearson"]
              for i in range(1, 4)]
    result = {"folds": values, "mean": float(np.mean(values))}
    (OUT / f"{name}_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def stage3(parquet, number, fit, validation, name):
    lags, ema_span = {"ridge_lag8": ((1, 8), None),
                      "ridge_lag32": ((1, 8, 32), None),
                      "ridge_ema32": ((1, 8), 32)}[name]
    folder = OUT / f"fold_{number}" / name
    folder.mkdir(parents=True, exist_ok=True)
    artifact = folder / "scaled_ridge.npy"
    if not artifact.exists():
        print("train", number, name, flush=True)
        train_ridge(parquet, fit, folder, lags=lags, ema_span=ema_span)
    weights = np.load(artifact).astype(np.float64)
    scores = GlobalWP()
    mask = np.arange(N_ROWS) >= WARMUP
    for n, group in enumerate(validation, 1):
        x, y = read_group(parquet, group)
        features = history_features(x, lags, ema_span)[WARMUP:]
        if not np.isfinite(features).all():
            raise ValueError("nonfinite required-row history feature")
        values = np.zeros((N_ROWS, 2), np.float32)
        values[WARMUP:] = np.clip(features.astype(np.float64) @ weights[:-1] + weights[-1],
                                  -2, 2).astype(np.float32)
        scores.add(y, values, mask)
        if n % 100 == 0:
            print("scored", number, name, n, flush=True)
    result = scores.result()
    (folder / "score64.json").write_text(json.dumps(result, indent=2) + "\n")
    print("stage3 fold", number, name, result["weighted_pearson"], flush=True)


def stage4(parquet, number, fit, validation, name):
    options = {"mlp_norm4": {"epochs": 4, "lags": ()},
               "mlp_norm_lag8": {"epochs": 2, "lags": (1, 8)},
               "mlp_norm_lag32": {"epochs": 2, "lags": (1, 8, 32)},
               "mlp_norm_lag32_4": {"epochs": 4, "lags": (1, 8, 32)}}[name]
    lags = options["lags"]
    folder = OUT / f"fold_{number}" / name
    folder.mkdir(parents=True, exist_ok=True)
    artifact = folder / "scaled_mlp.pt"
    if not artifact.exists():
        print("train", number, name, flush=True)
        train_mlp(parquet, fit, folder, standardize=True, **options)
    model = make_mlp(N_FEATURES * (1 + len(lags)))
    model.load_state_dict(torch.load(artifact, map_location="cpu", weights_only=True))
    model.eval()
    with np.load(folder / "mlp_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    scores = GlobalWP()
    mask = np.arange(N_ROWS) >= WARMUP
    with torch.no_grad():
        for n, group in enumerate(validation, 1):
            x, y = read_group(parquet, group)
            if lags:
                x = history_features(x, lags)
            x = np.clip((x - mean) / std, -10, 10)
            values = model(torch.from_numpy(np.ascontiguousarray(x))).numpy()
            scores.add(y, values, mask)
            if n % 100 == 0:
                print("scored", number, name, n, flush=True)
    result = scores.result()
    (folder / "score.json").write_text(json.dumps(result, indent=2) + "\n")
    print("stage4 fold", number, name, result["weighted_pearson"], flush=True)


def stage5(parquet, number, validation):
    models = {}
    for name, width, seed in (("gru96_seed7", 96, 7), ("gru96_seed17", 96, 17),
                              ("gru96_seed37", 96, 37), ("gru64_seed7", 64, 7)):
        model = make_gru(width)
        model.load_state_dict(torch.load(gru_file(number, width, seed), map_location="cpu", weights_only=True))
        models[name] = model.eval()
    folder = OUT / f"fold_{number}"
    gru = make_gru(96)
    gru.load_state_dict(torch.load(folder / "gru96_norm5ep/scaled_gru.pt",
                                    map_location="cpu", weights_only=True))
    models["gru96_norm5ep"] = gru.eval()
    with np.load(folder / "gru96_norm5ep/gru_standardizer.npz") as source:
        gru_mean, gru_std = source["mean"], source["std"]
    ridge = np.load(folder / "ridge_ema32/scaled_ridge.npy").astype(np.float64)
    mlp = make_mlp(4 * N_FEATURES)
    mlp.load_state_dict(torch.load(folder / "mlp_norm_lag32_4/scaled_mlp.pt",
                                   map_location="cpu", weights_only=True))
    mlp.eval()
    with np.load(folder / "mlp_norm_lag32_4/mlp_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    masks = ("all", "early", "late", "random20")
    names = ("gru96_seed7", "gru96_norm5ep", "gru96_3seed", "norm5_seedmix",
             "norm5_gru64", "ridge_ema32", "mlp_lag32_4", "norm5_ridge10",
             "norm5_mlp10", "norm5_both10", "scale2", "scale4", "scale8",
             "scale4_shift_pos", "scale4_shift_neg")
    scores = {name: {mask: GlobalWP() for mask in masks} for name in names}
    steps = np.arange(N_ROWS)
    required = steps >= WARMUP
    with torch.no_grad():
        for n, group in enumerate(validation, 1):
            x, y = read_group(parquet, group)
            p = {name: gru_predict(model, np.clip((x - gru_mean) / gru_std, -10, 10)
                   if name == "gru96_norm5ep" else x) for name, model in models.items()}
            g = p["gru96_norm5ep"]
            features = history_features(x, (1, 8), 32)
            if not np.isfinite(features[WARMUP:]).all():
                raise ValueError("nonfinite required-row ensemble feature")
            r = np.zeros((N_ROWS, 2), np.float64)
            r[WARMUP:] = features[WARMUP:].astype(np.float64) @ ridge[:-1] + ridge[-1]
            if not np.isfinite(r[WARMUP:]).all():
                raise ValueError("nonfinite required-row ridge prediction")
            standardized = np.clip((history_features(x, (1, 8, 32)) - mean) / std, -10, 10)
            m = mlp(torch.from_numpy(np.ascontiguousarray(standardized))).numpy()
            predictions = {
                "gru96_seed7": p["gru96_seed7"], "gru96_norm5ep": g,
                "gru96_3seed": (p["gru96_seed7"] + p["gru96_seed17"]
                                  + p["gru96_seed37"]) / 3,
                "norm5_seedmix": (g + p["gru96_seed17"] + p["gru96_seed37"]) / 3,
                "norm5_gru64": (g + p["gru64_seed7"]) / 2,
                "ridge_ema32": r, "mlp_lag32_4": m,
                "norm5_ridge10": 0.9*g + 0.1*r,
                "norm5_mlp10": 0.9*g + 0.1*m,
                "norm5_both10": 0.8*g + 0.1*r + 0.1*m,
                "scale2": 2*g, "scale4": 4*g, "scale8": 8*g,
                "scale4_shift_pos": 4*g + 0.2, "scale4_shift_neg": 4*g - 0.2,
            }
            selected = {"all": required, "early": required & (steps < 10_000),
                        "late": steps >= 10_000,
                        "random20": required & (np.random.default_rng(20260917 + int(group))
                                                 .random(N_ROWS) < 0.2)}
            for name, values in predictions.items():
                for mask, indices in selected.items():
                    scores[name][mask].add(y, values, indices)
            if n % 100 == 0:
                print("stage5 scored", number, n, flush=True)
    result = {name: {mask: acc.result() for mask, acc in masks.items()}
              for name, masks in scores.items()}
    (folder / "stage5.json").write_text(json.dumps(result, indent=2) + "\n")
    print("stage5 fold", number, {key: round(value["all"]["weighted_pearson"], 6)
                                  for key, value in result.items()}, flush=True)


def summary_stage5():
    results = [json.loads((OUT / f"fold_{i}" / "stage5.json").read_text()) for i in range(1, 4)]
    summary = {}
    for name in results[0]:
        values = [fold[name]["all"]["weighted_pearson"] for fold in results]
        summary[name] = {"folds": values, "mean": float(np.mean(values)),
                         "early_mean": float(np.mean([fold[name]["early"]["weighted_pearson"]
                                                      for fold in results])),
                         "late_mean": float(np.mean([fold[name]["late"]["weighted_pearson"]
                                                     for fold in results])),
                         "random20_mean": float(np.mean([fold[name]["random20"]["weighted_pearson"]
                                                         for fold in results]))}
    (OUT / "stage5_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def selfcheck():
    check_folds()
    x = np.arange(10 * N_FEATURES, dtype=np.float32).reshape(10, N_FEATURES)
    history = history_features(x, (1, 8), 32)
    assert history.shape == (10, 4 * N_FEATURES)
    assert np.array_equal(history[:, :2 * N_FEATURES], lag_features(x))
    assert np.all(history[:8, 2 * N_FEATURES:3 * N_FEATURES] == 0)
    assert np.max(np.abs(history[0, 3 * N_FEATURES:])) < 1e-5
    print("improvement feature checks passed")


def fit_full_gru(parquet, name):
    folder = OUT / "full_fit" / name
    folder.mkdir(parents=True, exist_ok=True)
    artifact = folder / "scaled_gru.pt"
    if artifact.exists():
        raise FileExistsError(f"completed full-FIT model already exists: {artifact}")
    if name == "gru96_norm5ep":
        train_gru(parquet, FIT, folder, hidden_size=96, standardize=True,
                  epochs=5, decay_after=3)
    else:
        train_gru(parquet, FIT, folder, hidden_size=96, seed=int(name.rsplit("seed", 1)[1]))


def fit_full_other(parquet, name):
    folder = OUT / "full_fit" / name
    folder.mkdir(parents=True, exist_ok=True)
    artifact = folder / ("scaled_ridge.npy" if name == "ridge_ema32" else "scaled_mlp.pt")
    if artifact.exists():
        raise FileExistsError(f"completed full-FIT model already exists: {artifact}")
    if name == "ridge_ema32":
        train_ridge(parquet, FIT, folder, lags=(1, 8), ema_span=32)
    else:
        train_mlp(parquet, FIT, folder, lags=(1, 8, 32), standardize=True, epochs=4)


def score_full(parquet, mode):
    names = ("gru96_norm5ep", "gru96_seed17", "gru96_seed37")
    models = {}
    for name in names:
        model = make_gru(96)
        model.load_state_dict(torch.load(OUT / "full_fit" / name / "scaled_gru.pt",
                                         map_location="cpu", weights_only=True))
        models[name] = model.eval()
    with np.load(OUT / "full_fit/gru96_norm5ep/gru_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    ridge = np.load(OUT / "full_fit/ridge_ema32/scaled_ridge.npy").astype(np.float64)
    mlp = make_mlp(4 * N_FEATURES)
    mlp.load_state_dict(torch.load(OUT / "full_fit/mlp_lag32_4/scaled_mlp.pt",
                                   map_location="cpu", weights_only=True))
    mlp.eval()
    with np.load(OUT / "full_fit/mlp_lag32_4/mlp_standardizer.npz") as source:
        mlp_mean, mlp_std = source["mean"], source["std"]
    masks = ("official",) if mode == "oos" else ("all", "early", "late", "random20")
    scores = {name: {mask: GlobalWP() for mask in masks}
              for name in (*names, "ridge_ema32", "mlp_lag32_4", "ensemble")}
    steps = np.arange(N_ROWS)
    required = steps >= WARMUP
    seen = set()
    selected_rows = 0
    with torch.no_grad():
        for n, group in enumerate(range(parquet.num_row_groups) if mode == "oos" else DEV, 1):
            if mode == "oos":
                from oos_global import read_group as read_valid_group
                seq, need, x, y, official = read_valid_group(parquet, int(group))
                if seq in seen or not np.array_equal(need, required):
                    raise ValueError("duplicate OOS sequence or changed prediction mask")
                seen.add(seq)
                selected = {"official": official}
            else:
                x, y = read_group(parquet, group)
                selected = {"all": required, "early": required & (steps < 10_000),
                            "late": steps >= 10_000,
                            "random20": required & (np.random.default_rng(20260917 + int(group))
                                                     .random(N_ROWS) < 0.2)}
            selected_rows += int(selected[next(iter(selected))].sum())
            predictions = {name: gru_predict(model, np.clip((x - mean) / std, -10, 10)
                           if name == "gru96_norm5ep" else x) for name, model in models.items()}
            predictions["ensemble"] = sum(predictions.values()) / 3
            ridge_features = history_features(x, (1, 8), 32)
            if not np.isfinite(ridge_features[WARMUP:]).all():
                raise ValueError("nonfinite required-row Ridge feature")
            ridge_values = np.zeros((N_ROWS, 2), np.float64)
            ridge_values[WARMUP:] = (ridge_features[WARMUP:].astype(np.float64)
                                      @ ridge[:-1] + ridge[-1])
            predictions["ridge_ema32"] = ridge_values
            mlp_features = history_features(x, (1, 8, 32))
            mlp_features = np.clip((mlp_features - mlp_mean) / mlp_std, -10, 10)
            predictions["mlp_lag32_4"] = mlp(torch.from_numpy(np.ascontiguousarray(mlp_features))).numpy()
            for name, values in predictions.items():
                if not np.isfinite(np.asarray(values[required], np.float32)).all():
                    raise ValueError(f"nonfinite required-row prediction: {name}")
                for mask, indices in selected.items():
                    scores[name][mask].add(y, values, indices)
            if n % 200 == 0:
                print(mode, "scored groups", n, flush=True)
    if mode == "oos" and (len(seen) != 1_873 or selected_rows != 3_528_273):
        raise ValueError("OOS sequence count or official mask changed")
    if mode == "dev" and selected_rows != len(DEV) * (N_ROWS - WARMUP):
        raise ValueError("DEV selected row count changed")
    result = {"split": mode, "sequences": len(seen) if mode == "oos" else len(DEV),
              "selected_rows": selected_rows,
              "models": {name: {mask: acc.result() for mask, acc in masks.items()}
                         for name, masks in scores.items()}}
    output = OUT / f"full_fit/{mode}_scores.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(mode, {name: round(value[next(iter(value))]["weighted_pearson"], 6)
                 for name, value in result["models"].items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("selfcheck", "stage1", "stage1_summary", "stage2",
                                          "stage2_summary", "stage3", "stage4", "stage5",
                                          "stage5_summary", "fit_best_gru", "fit_ensemble_seed",
                                          "fit_best_ridge", "fit_best_mlp",
                                          "score_dev", "score_oos"))
    parser.add_argument("--fold", type=int, choices=(1, 2, 3))
    parser.add_argument("--name", choices=("gru96_seed17", "gru96_seed37", "gru96_norm",
                                           "gru96_5ep_fixedlr", "gru96_5ep",
                                           "gru96_norm5ep", "gru96_corr",
                                           "ridge_lag8", "ridge_lag32", "ridge_ema32", "mlp_norm4",
                                           "mlp_norm_lag8", "mlp_norm_lag32", "mlp_norm_lag32_4"))
    args = parser.parse_args()
    if args.stage == "selfcheck":
        selfcheck()
        return
    if args.stage == "stage1_summary":
        summary_stage1()
        return
    if args.stage == "stage2_summary":
        summary_stage2(args.name)
        return
    if args.stage == "stage5_summary":
        summary_stage5()
        return
    torch.set_num_threads(1)
    if args.stage == "score_oos":
        parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
        if (parquet.num_row_groups != 1_873 or
                (ROOT / "datasets/valid.parquet").stat().st_size != VALID_BYTES or
                sequence_id_digest(parquet) != VALID_ID_DIGEST):
            raise ValueError("official validation file changed")
        score_full(parquet, "oos")
        return
    parquet = pq.ParquetFile(ROOT / "datasets/train.parquet")
    check_train(parquet)
    if args.stage == "score_dev":
        score_full(parquet, "dev")
        return
    if args.stage == "fit_best_gru":
        fit_full_gru(parquet, "gru96_norm5ep")
        return
    if args.stage == "fit_ensemble_seed":
        if args.name not in ("gru96_seed17", "gru96_seed37"):
            raise ValueError("choose --name gru96_seed17 or gru96_seed37")
        fit_full_gru(parquet, args.name)
        return
    if args.stage == "fit_best_ridge":
        fit_full_other(parquet, "ridge_ema32")
        return
    if args.stage == "fit_best_mlp":
        fit_full_other(parquet, "mlp_lag32_4")
        return
    for number, fit, validation in folds():
        if args.fold is None or args.fold == number:
            if args.stage == "stage1":
                stage1(parquet, number, fit, validation)
            elif args.stage == "stage2":
                stage2(parquet, number, fit, validation, args.name)
            elif args.stage == "stage3":
                stage3(parquet, number, fit, validation, args.name)
            elif args.stage == "stage4":
                stage4(parquet, number, fit, validation, args.name)
            else:
                stage5(parquet, number, validation)


if __name__ == "__main__":
    main()
