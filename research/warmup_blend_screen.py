"""Blend the fixed epoch-4 warm-up-context GRU into cached FIT-only ensembles.

The candidate and every reference member are fitted only on a fold's 1,024
training sequences.  Scores use the disjoint 512-sequence validation part.
Blend weights are fixed in this script rather than fitted to validation labels.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pyarrow.parquet as pq
import torch

from advance import SplitEncoderGRU, predict_sequence
from global_wp import GlobalWP
from group_cv import folds
from pilot import CACHE, N_ROWS, ROOT, WARMUP, check_train, make_gru
from scale import read_group
from warmup_context_screen import WarmupContextGRU


OUT = CACHE / "warmup_context"
WEIGHTS = (0.05, 0.10, 0.20, 0.25)
EXPECTED_STRONG = (0.41808350984759823, 0.43868287863181016, 0.4215085033001944)


def predict_warmup(model, values):
    tensor = torch.from_numpy(np.ascontiguousarray(values))[None]
    context = model.context(tensor)
    hidden, pieces = None, []
    for start in range(0, N_ROWS, 512):
        prediction, hidden = model(tensor[:, start:start + 512], context, hidden)
        pieces.append(prediction[0].numpy())
    return np.concatenate(pieces)


def load_models(fold):
    improve = CACHE / "improve" / f"fold_{fold}"
    with np.load(improve / "gru96_norm5ep/gru_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    models = {}
    for name, path in (
        ("norm5", improve / "gru96_norm5ep/scaled_gru.pt"),
        ("seed17", improve / "gru96_seed17/scaled_gru.pt"),
        ("seed37", improve / "gru96_seed37/scaled_gru.pt"),
    ):
        model = make_gru(96)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        models[name] = model.eval()
    split = SplitEncoderGRU()
    split.load_state_dict(torch.load(CACHE / "advance" / f"fold_{fold}" /
                                     "split_encoder/model.pt", map_location="cpu",
                                     weights_only=True))
    models["split"] = split.eval()
    wide = make_gru(128)
    wide.load_state_dict(torch.load(CACHE / "advance" / f"fold_{fold}" /
                                    "gru128_5ep/model.pt", map_location="cpu",
                                    weights_only=True))
    models["wide"] = wide.eval()
    warmup = WarmupContextGRU()
    warmup.load_state_dict(torch.load(OUT / f"fold_{fold}/seed7_e4.pt",
                                      map_location="cpu", weights_only=True))
    models["warmup"] = warmup.eval()
    return models, mean, std


def score_fold(fold, validation):
    models, mean, std = load_models(fold)
    names = ["three_gru", "strong", "warmup"]
    names += [f"three_warmup_{weight:.2f}" for weight in WEIGHTS]
    names += [f"strong_warmup_{weight:.2f}" for weight in WEIGHTS]
    scores = {name: GlobalWP() for name in names}
    required = np.arange(N_ROWS) >= WARMUP
    parquet = pq.ParquetFile(ROOT / "datasets/train.parquet")
    check_train(parquet)
    with torch.no_grad():
        for n, group in enumerate(validation, 1):
            raw, target = read_group(parquet, int(group))
            normalized = np.clip((raw - mean) / std, -10, 10)
            norm5 = predict_sequence(models["norm5"], normalized)
            seed17 = predict_sequence(models["seed17"], raw)
            seed37 = predict_sequence(models["seed37"], raw)
            three = (norm5 + seed17 + seed37) / 3
            split = predict_sequence(models["split"], normalized)
            wide = predict_sequence(models["wide"], normalized)
            strong = (three + split + wide) / 3
            warmup = predict_warmup(models["warmup"], normalized)
            predictions = {"three_gru": three, "strong": strong, "warmup": warmup}
            for weight in WEIGHTS:
                predictions[f"three_warmup_{weight:.2f}"] = (1 - weight) * three + weight * warmup
                predictions[f"strong_warmup_{weight:.2f}"] = (1 - weight) * strong + weight * warmup
            for name, prediction in predictions.items():
                scores[name].add(target, prediction, required)
            if n % 100 == 0:
                print("warmup blend fold", fold, "scored", n, flush=True)
    result = {name: metric.result() for name, metric in scores.items()}
    baseline = result["strong"]["weighted_pearson"]
    if not np.isclose(baseline, EXPECTED_STRONG[fold - 1], atol=1e-9):
        raise ValueError(f"strong reference did not reproduce: {baseline}")
    folder = OUT / f"fold_{fold}"
    (folder / "blend_e4.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: row["weighted_pearson"] for name, row in result.items()}, indent=2),
          flush=True)


def summarize():
    results = [json.loads((OUT / f"fold_{fold}/blend_e4.json").read_text())
               for fold in range(1, 4)]
    summary = {}
    reference = np.mean([result["strong"]["weighted_pearson"] for result in results])
    for name in results[0]:
        values = [result[name]["weighted_pearson"] for result in results]
        summary[name] = {"folds": values, "mean": float(np.mean(values)),
                         "gain_vs_strong": float(np.mean(values) - reference)}
    (OUT / "blend_e4_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3))
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(3)
    if args.summary:
        summarize()
        return
    if args.fold is None:
        parser.error("--fold or --summary is required")
    _, _, validation = next(plan for plan in folds() if plan[0] == args.fold)
    score_fold(args.fold, validation)


if __name__ == "__main__":
    main()
