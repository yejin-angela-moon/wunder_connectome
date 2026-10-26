"""Rescore frozen pilot and expanded models on train-only development groups."""
from __future__ import annotations

import json
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from global_wp import GlobalWP
from pilot import CACHE, ROOT, N_ROWS, WARMUP, check_train, load_frozen_models, make_gru, make_mlp, predict_frozen
from scale import DEV, read_group


def expanded_models():
    ridge = np.load(CACHE / "scaled_ridge.npy")
    mlp = make_mlp()
    mlp.load_state_dict(torch.load(CACHE / "scaled_mlp.pt", map_location="cpu", weights_only=True))
    gru = make_gru()
    gru.load_state_dict(torch.load(CACHE / "scaled_gru.pt", map_location="cpu", weights_only=True))
    return ridge, mlp.eval(), gru.eval()


def main():
    torch.set_num_threads(1)
    parquet = pq.ParquetFile(ROOT / "datasets/train.parquet")
    check_train(parquet)
    models = {"pilot": load_frozen_models(), "expanded": expanded_models()}
    names = [f"{stage}_{name}" for stage in models for name in ("ridge", "mlp", "gru")]
    masks = ("all", "early", "late", "random20")
    global_scores = {name: {mask: GlobalWP() for mask in masks} for name in names}
    from utils import BlockAccumulator
    block_scores = {name: BlockAccumulator() for name in names}
    steps = np.arange(N_ROWS)
    required = steps >= WARMUP
    started = time.monotonic()
    with torch.no_grad():
        for n, group in enumerate(DEV, 1):
            x, y = read_group(parquet, group)
            random20 = np.random.default_rng(20260917 + int(group)).random(N_ROWS) < 0.2
            selected = {"all": required, "early": required & (steps < 10_000),
                        "late": steps >= 10_000, "random20": required & random20}
            for stage, frozen in models.items():
                predictions = predict_frozen(frozen, x)
                for model, values in predictions.items():
                    name = f"{stage}_{model}"
                    for mask, indices in selected.items():
                        global_scores[name][mask].add(y, values, indices)
                    block_scores[name].add(y, values, required)
            if n % 200 == 0:
                print("rescored train-development groups", n, flush=True)
    result = {}
    for name in names:
        per_sequence = np.asarray(block_scores[name].scores).mean(axis=1)
        result[name] = {"global": {mask: score.result() for mask, score in global_scores[name].items()},
                        "legacy_block": block_scores[name].result(),
                        "sequence_wp_quantiles": np.quantile(per_sequence, [0.1, 0.5, 0.9]).tolist()}
    (CACHE / "train_dev_global_rescore.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: {mask: round(v["weighted_pearson"], 6)
                             for mask, v in scores["global"].items()} for name, scores in result.items()}, indent=2))
    print("seconds", round(time.monotonic() - started, 2))


if __name__ == "__main__":
    main()
