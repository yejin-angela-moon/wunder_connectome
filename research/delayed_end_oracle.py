"""Causal diagnostic: reconstruct the prior target at the next event.

Unlike delayed_oracle.py, this uses the last row of the completed interval,
which is observable when the following price event arrives.  It then tests
whether that delayed estimate is persistent enough to predict the new
interval and whether a fitted transition adds value.
"""
from __future__ import annotations

import json

import numpy as np

from delayed_oracle import PRICE, solve, starts
from global_wp import GlobalWP
from pilot import CACHE, DEV_GROUPS, FIT_GROUPS, N_ROWS, WARMUP

OUT = CACHE / "delayed_end_oracle.json"


def reconstruction_rows(values, targets):
    event = starts(values)
    end = event[1:] - 1
    before = values[end][:, PRICE]
    after = values[event[1:]][:, PRICE]
    features = np.column_stack((before, after, after - before))
    return event, features, targets[end]


def fit_reconstruction(x, y):
    pairs = [reconstruction_rows(values, targets) for values, targets in zip(x, y)]
    raw = np.concatenate([pair[1] for pair in pairs])
    mean, std = raw.mean(0), raw.std(0)
    std[std < 1e-6] = 1
    normalized = [np.clip((pair[1] - mean) / std, -10, 10) for pair in pairs]
    return mean, std, solve(normalized, [pair[2] for pair in pairs], raw.shape[1])


def reconstruct(values, fitted):
    event = starts(values)
    end = event[1:] - 1
    before = values[end][:, PRICE]
    after = values[event[1:]][:, PRICE]
    features = np.column_stack((before, after, after - before))
    mean, std, coefficients = fitted
    design = np.column_stack((np.clip((features - mean) / std, -10, 10),
                              np.ones(len(features))))
    with np.errstate(all="ignore"):
        prediction = design @ coefficients
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite delayed reconstruction")
    return event, prediction


def fit_transition(x, y, fitted_reconstruction):
    rows, targets = [], []
    for values, truth in zip(x, y):
        event, previous = reconstruct(values, fitted_reconstruction)
        current = values[event[1:]][:, PRICE]
        before = values[event[1:] - 1][:, PRICE]
        rows.append(np.column_stack((previous, before, current, current - before)))
        targets.append(truth[event[1:]])
    raw = np.concatenate(rows)
    mean, std = raw.mean(0), raw.std(0)
    std[std < 1e-6] = 1
    normalized = [np.clip((features - mean) / std, -10, 10) for features in rows]
    return mean, std, solve(normalized, targets, raw.shape[1])


def predictions(values, fitted_reconstruction, transition):
    event, previous = reconstruct(values, fitted_reconstruction)
    current = values[event[1:]][:, PRICE]
    before = values[event[1:] - 1][:, PRICE]
    features = np.column_stack((previous, before, current, current - before))
    mean, std, coefficients = transition
    design = np.column_stack((np.clip((features - mean) / std, -10, 10),
                              np.ones(len(features))))
    with np.errstate(all="ignore"):
        updated = design @ coefficients
    if not np.isfinite(updated).all():
        raise ValueError("nonfinite delayed transition")
    lagged_rows = np.zeros((N_ROWS, 2), np.float32)
    updated_rows = np.zeros((N_ROWS, 2), np.float32)
    for index, begin in enumerate(event[1:]):
        end = event[index + 2] if index + 2 < len(event) else N_ROWS
        lagged_rows[begin:end] = previous[index]
        updated_rows[begin:end] = updated[index]
    return lagged_rows, updated_rows


def score(x, y, fitted_reconstruction, transition):
    metrics = {name: GlobalWP() for name in ("reconstructed_previous", "transition")}
    required = np.arange(N_ROWS) >= WARMUP
    for values, targets in zip(x, y):
        previous, updated = predictions(values, fitted_reconstruction, transition)
        metrics["reconstructed_previous"].add(targets, previous, required)
        metrics["transition"].add(targets, updated, required)
    return {name: metric.result() for name, metric in metrics.items()}


def selfcheck():
    values = np.zeros((N_ROWS, 112), np.float32)
    values[:, :22] = (np.arange(N_ROWS) // 10)[:, None]
    targets = np.column_stack((np.arange(N_ROWS) // 10,
                               -(np.arange(N_ROWS) // 10))).astype(np.float32)
    event, features, y = reconstruction_rows(values, targets)
    assert np.array_equal(event[:4], [0, 10, 20, 30])
    assert np.array_equal(y[:3], targets[[9, 19, 29]])
    assert features.shape[1] == 3 * len(PRICE)
    print("completed-interval endpoint and next-event indexing checks passed")


def run():
    x = np.load(CACHE / "fit_x.npy", mmap_mode="r")
    y = np.load(CACHE / "fit_y.npy", mmap_mode="r")
    dx = np.load(CACHE / "dev_x.npy", mmap_mode="r")
    dy = np.load(CACHE / "dev_y.npy", mmap_mode="r")
    if x.shape != (FIT_GROUPS, N_ROWS, 112) or dx.shape != (DEV_GROUPS, N_ROWS, 112):
        raise ValueError("unexpected locked pilot cache")
    reconstruction = fit_reconstruction(x, y)
    transition = fit_transition(x, y, reconstruction)
    result = score(dx, dy, reconstruction, transition)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    import sys
    selfcheck() if sys.argv[-1:] == ["selfcheck"] else run()
