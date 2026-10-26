"""Refit three preselected variants on all locked fitting sequences; score train only."""
from __future__ import annotations

import json
import sys

import numpy as np
import pyarrow.parquet as pq
import torch

from global_wp import GlobalWP
from pilot import CACHE, ROOT, N_ROWS, WARMUP, check_train, gru_predict, make_gru, make_mlp
from scale import DEV, FIT, lag_features, read_group, train_gru, train_mlp, train_ridge

sys.path.insert(0, str(ROOT))

OUTPUT = CACHE / "selected"
NAMES = ("ridge_lag", "mlp_norm", "gru96")


def train(parquet, name):
    folder = OUTPUT / name
    folder.mkdir(parents=True, exist_ok=True)
    artifact = folder / ("scaled_ridge.npy" if name == "ridge_lag" else
                         "scaled_mlp.pt" if name == "mlp_norm" else "scaled_gru.pt")
    if artifact.exists():
        raise FileExistsError(f"completed model already exists: {artifact}")
    if name == "ridge_lag":
        train_ridge(parquet, FIT, folder, lag_differences=True)
    elif name == "mlp_norm":
        train_mlp(parquet, FIT, folder, standardize=True)
    else:
        train_gru(parquet, FIT, folder, hidden_size=96)


def predictors():
    weights = np.load(OUTPUT / "ridge_lag/scaled_ridge.npy")
    mlp = make_mlp().eval()
    mlp.load_state_dict(torch.load(OUTPUT / "mlp_norm/scaled_mlp.pt", map_location="cpu", weights_only=True))
    with np.load(OUTPUT / "mlp_norm/mlp_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    gru = make_gru(96).eval()
    gru.load_state_dict(torch.load(OUTPUT / "gru96/scaled_gru.pt", map_location="cpu", weights_only=True))

    def predict_mlp(x):
        standardized = np.clip((x - mean) / std, -10, 10)
        return mlp(torch.from_numpy(np.ascontiguousarray(standardized))).numpy()

    return {"ridge_lag": lambda x: lag_features(x) @ weights[:-1] + weights[-1],
            "mlp_norm": predict_mlp,
            "gru96": lambda x: gru_predict(gru, x)}


def score(parquet):
    from utils import BlockAccumulator                                 # older starter packs only
    models = predictors()
    masks = ("all", "early", "late", "random20")
    global_scores = {name: {mask: GlobalWP() for mask in masks} for name in NAMES}
    block_scores = {name: BlockAccumulator() for name in NAMES}
    steps = np.arange(N_ROWS)
    required = steps >= WARMUP
    with torch.no_grad():
        for n, group in enumerate(DEV, 1):
            x, y = read_group(parquet, group)
            selected = {"all": required, "early": required & (steps < 10_000),
                        "late": steps >= 10_000,
                        "random20": required & (np.random.default_rng(20260917 + int(group)).random(N_ROWS) < 0.2)}
            for name, predict in models.items():
                with np.errstate(all="ignore"):
                    values = predict(x)
                for mask, indices in selected.items():
                    global_scores[name][mask].add(y, values, indices)
                block_scores[name].add(y, values, required)
            if n % 200 == 0:
                print("scored fixed development groups", n, flush=True)
    result = {}
    for name in NAMES:
        per_sequence = np.asarray(block_scores[name].scores).mean(axis=1)
        result[name] = {"global": {mask: acc.result() for mask, acc in global_scores[name].items()},
                        "legacy_block": block_scores[name].result(),
                        "sequence_wp_quantiles": np.quantile(per_sequence, [0.1, 0.5, 0.9]).tolist()}
    (OUTPUT / "fixed_dev_scores.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: {mask: round(value["weighted_pearson"], 6)
                             for mask, value in scores["global"].items()}
                      for name, scores in result.items()}, indent=2))


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in (*NAMES, "score"):
        raise SystemExit("usage: refit_selected.py {ridge_lag|mlp_norm|gru96|score}")
    torch.set_num_threads(1)
    parquet = pq.ParquetFile(ROOT / "datasets/train.parquet")
    check_train(parquet)
    if sys.argv[1] == "score":
        score(parquet)
    else:
        train(parquet, sys.argv[1])


if __name__ == "__main__":
    main()
