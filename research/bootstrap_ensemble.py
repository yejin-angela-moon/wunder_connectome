"""Sequence bootstrap for the small gains among deployable cached ensembles."""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from ensemble_screen import MEMBERS, load
from pilot import CACHE, N_ROWS, ROOT

OUT = CACHE / "bootstrap_ensemble.json"


def sufficient(y, prediction, counts):
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    result = np.empty((len(counts), 2, 6), np.float64)
    y = np.clip(y.astype(np.float64), -2, 2)
    prediction = np.clip(prediction.astype(np.float64), -2, 2)
    for target in range(2):
        value, estimate = y[:, target], prediction[:, target]
        weight = np.abs(value)
        columns = np.column_stack((weight, weight * value, weight * estimate,
                                   weight * value**2, weight * estimate**2,
                                   weight * value * estimate))
        result[:, target] = np.add.reduceat(columns, starts)
    return result


def scores(sums):
    sw, sy, sp, syy, spp, syp = np.moveaxis(sums, -1, 0)
    vy = syy - sy**2 / sw
    vp = spp - sp**2 / sw
    covariance = syp - sy * sp / sw
    return np.mean(covariance / np.sqrt(vy * vp), axis=-1)


def selfcheck():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(12, 2)).astype(np.float32)
    p = y + rng.normal(scale=0.2, size=y.shape)
    stats = sufficient(y, p, np.array([5, 7]))
    assert np.isfinite(stats).all() and scores(stats.sum(0)[None]).shape == (1,)
    print("sequence sufficient-statistic bootstrap self-check passed")


def main():
    mask = pq.read_table(ROOT / "datasets/valid_mask.parquet", columns=["is_scored"])
    counts = mask["is_scored"].combine_chunks().to_numpy(
        zero_copy_only=False).reshape(-1, N_ROWS).sum(1)
    names = list(MEMBERS)
    specs = {
        "v21": ("split", "sd71", "sd72", "g71", "g72"),
        "v23_proxy": ("g192a50", "split", "sd71", "g71"),
        "continued": ("g192a50", "split", "c71", "c72"),
    }
    blocks = {name: [] for name in specs}
    for half in (0, 1):
        p, y, _ = load(half)
        selected_counts = counts[np.arange(len(counts)) % 2 != half]
        if selected_counts.sum() != len(y):
            raise ValueError("cached rows do not match validation masks")
        for label, members in specs.items():
            indices = [names.index(member) for member in members]
            blocks[label].append(sufficient(y, p[:, indices].mean(1), selected_counts))
    blocks = {name: np.concatenate(parts) for name, parts in blocks.items()}
    rng = np.random.default_rng(20261002)
    draws = rng.integers(len(counts), size=(5000, len(counts)))
    boot = {name: scores(stats[draws].sum(1)) for name, stats in blocks.items()}
    result = {"point": {name: float(scores(stats.sum(0)[None])[0]) for name, stats in blocks.items()},
              "gain_vs_v21": {}}
    for name in ("v23_proxy", "continued"):
        difference = boot[name] - boot["v21"]
        result["gain_vs_v21"][name] = {
            "mean": float(difference.mean()),
            "ci95": np.quantile(difference, [0.025, 0.975]).tolist(),
            "probability_positive": float(np.mean(difference > 0)),
        }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import sys
    selfcheck() if sys.argv[-1:] == ["selfcheck"] else main()
