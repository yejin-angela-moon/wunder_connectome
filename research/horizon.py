"""Train-only oracle scan for the unknown target horizon.

Future rows are used only to diagnose the label construction.  They are never
submission features.  Models fit the locked pilot FIT sequences and score the
disjoint pilot DEV sequences.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from global_wp import GlobalWP
from pilot import CACHE, DEV_GROUPS, FIT_GROUPS, N_ROWS, WARMUP

OUT = CACHE / "horizon_scan.json"
PRICE = np.r_[0:22, 44:48]
ROW_HORIZONS = (1, 4, 16, 64, 256, 512, 1024)
EVENT_HORIZONS = (1, 2, 4, 8, 16, 32)
STEP = 8
RIDGE = 100.0


def indices(x, horizon, event=False):
    rows = np.arange(WARMUP, N_ROWS, STEP)
    if not event:
        return rows[rows + horizon < N_ROWS], rows[rows + horizon < N_ROWS] + horizon
    changed = np.r_[True, np.any(x[1:, :22] != x[:-1, :22], axis=1)]
    starts = np.flatnonzero(changed)
    segment = np.searchsorted(starts, rows, side="right") - 1
    keep = segment + horizon < len(starts)
    return rows[keep], starts[segment[keep] + horizon]


def design(x, horizon, event=False, oracle=True):
    now, future = indices(x, horizon, event)
    current = x[now][:, PRICE]
    values = np.column_stack((current, x[future][:, PRICE] - current)) if oracle else current
    return values.astype(np.float64), now


def fit(x, y, horizon, event=False, oracle=True):
    total = np.zeros(len(PRICE) * (2 if oracle else 1), np.float64)
    square = np.zeros_like(total)
    count = 0
    for values in x:
        a, _ = design(values, horizon, event, oracle)
        total += a.sum(0); square += (a * a).sum(0); count += len(a)
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean * mean, 1e-6))
    width = len(mean) + 1
    gram = np.zeros((2, width, width), np.float64)
    rhs = np.zeros((2, width), np.float64)
    for values, targets in zip(x, y):
        a, rows = design(values, horizon, event, oracle)
        a = np.column_stack((np.clip((a - mean) / std, -10, 10), np.ones(len(a))))
        targets = np.clip(targets[rows].astype(np.float64), -2, 2)
        for target in range(2):
            weight = np.sqrt(np.abs(targets[:, target]) + 1e-4)
            weighted = a * weight[:, None]
            with np.errstate(all="ignore"):
                gram[target] += weighted.T @ weighted
                rhs[target] += weighted.T @ (targets[:, target] * weight)
    coefficients = np.empty((width, 2), np.float64)
    for target in range(2):
        gram[target].flat[::width + 1] += RIDGE
        coefficients[:, target] = np.linalg.solve(gram[target], rhs[target])
    return mean, std, coefficients


def score(x, y, horizon, fitted, event=False, oracle=True):
    mean, std, coefficients = fitted
    metric = GlobalWP()
    for values, targets in zip(x, y):
        a, rows = design(values, horizon, event, oracle)
        a = np.column_stack((np.clip((a - mean) / std, -10, 10), np.ones(len(a))))
        with np.errstate(all="ignore"):
            prediction = a @ coefficients
        metric.add(targets[rows], prediction, np.ones(len(rows), bool))
    return metric.result()


def run():
    fit_x = np.load(CACHE / "fit_x.npy", mmap_mode="r")
    fit_y = np.load(CACHE / "fit_y.npy", mmap_mode="r")
    dev_x = np.load(CACHE / "dev_x.npy", mmap_mode="r")
    dev_y = np.load(CACHE / "dev_y.npy", mmap_mode="r")
    if fit_x.shape != (FIT_GROUPS, N_ROWS, 112) or dev_x.shape != (DEV_GROUPS, N_ROWS, 112):
        raise ValueError("run research/pilot.py prepare first")
    results = []
    for kind, horizons, event in (("row", ROW_HORIZONS, False),
                                  ("event", EVENT_HORIZONS, True)):
        for horizon in horizons:
            baseline = score(dev_x, dev_y, horizon,
                             fit(fit_x, fit_y, horizon, event, False), event, False)
            oracle = score(dev_x, dev_y, horizon,
                           fit(fit_x, fit_y, horizon, event, True), event, True)
            result = {"kind": kind, "horizon": horizon,
                      "baseline": baseline["weighted_pearson"],
                      "oracle": oracle["weighted_pearson"],
                      "gain": oracle["weighted_pearson"] - baseline["weighted_pearson"],
                      "t0": oracle["t0"], "t1": oracle["t1"],
                      "rows": oracle["selected_rows"]}
            results.append(result)
            print(result, flush=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")


def selfcheck():
    x = np.zeros((N_ROWS, 112), np.float32)
    x[:, 0] = np.arange(N_ROWS)
    now, future = indices(x, 4)
    assert np.array_equal(future, now + 4)
    x[:, :22] = np.arange(N_ROWS)[:, None] // 10
    now, future = indices(x, 2, True)
    assert future[0] == 110 and np.all(future > now)
    print("horizon row/event indexing checks passed")


if __name__ == "__main__":
    import sys
    selfcheck() if sys.argv[-1:] == ["selfcheck"] else run()
