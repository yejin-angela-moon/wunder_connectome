"""Train-only diagnostic for causal target updates at observed price events."""
from __future__ import annotations

import json

import numpy as np

from global_wp import GlobalWP
from pilot import CACHE, DEV_GROUPS, FIT_GROUPS, N_ROWS, WARMUP

OUT = CACHE / "event_delta_diagnostic.json"
PRICE = np.r_[0:22, 44:48]
RIDGE = 100.0


def rows(values, targets):
    price = values[:, :22]
    event = np.flatnonzero(np.r_[True, np.any(price[1:] != price[:-1], axis=1)])
    old, new = values[event[:-1]][:, PRICE], values[event[1:]][:, PRICE]
    features = np.column_stack((old, new, new - old)).astype(np.float64)
    delta = (targets[event[1:]] - targets[event[:-1]]).astype(np.float64)
    return event, features, delta


def fit(x, y):
    samples = [rows(values, targets) for values, targets in zip(x, y)]
    raw = np.concatenate([sample[1] for sample in samples])
    mean, std = raw.mean(0), raw.std(0)
    std[std < 1e-6] = 1
    width = raw.shape[1]
    gram = np.zeros((2, width + 1, width + 1), np.float64)
    rhs = np.zeros((2, width + 1), np.float64)
    for _, features, delta in samples:
        design = np.column_stack((np.clip((features - mean) / std, -10, 10),
                                  np.ones(len(features))))
        for target in range(2):
            with np.errstate(all="ignore"):
                update_gram = design.T @ design
                update_rhs = design.T @ delta[:, target]
            if not np.isfinite(update_gram).all() or not np.isfinite(update_rhs).all():
                raise ValueError("nonfinite event-delta sufficient statistics")
            gram[target] += update_gram
            rhs[target] += update_rhs
    coefficients = np.empty((width + 1, 2), np.float64)
    for target in range(2):
        gram[target].flat[::width + 2] += RIDGE
        coefficients[:, target] = np.linalg.solve(gram[target], rhs[target])
    return mean, std, coefficients


def predict(features, fitted):
    mean, std, coefficients = fitted
    design = np.column_stack((np.clip((features - mean) / std, -10, 10),
                              np.ones(len(features))))
    with np.errstate(all="ignore"):
        result = design @ coefficients
    if not np.isfinite(result).all():
        raise ValueError("nonfinite event-delta prediction")
    return result


def score(x, y, fitted):
    metrics = {name: GlobalWP() for name in
               ("true_previous", "true_previous_plus_delta", "true_anchor_integrated")}
    required = np.arange(N_ROWS) >= WARMUP
    event_true, event_pred = [], []
    for values, targets in zip(x, y):
        event, features, delta = rows(values, targets)
        estimated_delta = predict(features, fitted)
        event_true.append(delta); event_pred.append(estimated_delta)
        previous = np.zeros((N_ROWS, 2), np.float32)
        updated = np.zeros((N_ROWS, 2), np.float32)
        integrated = np.zeros((N_ROWS, 2), np.float32)
        state = targets[event[0]].astype(np.float64).copy()
        for j, begin in enumerate(event[1:]):
            end = event[j + 2] if j + 2 < len(event) else N_ROWS
            previous[begin:end] = targets[event[j]]
            updated[begin:end] = targets[event[j]] + estimated_delta[j]
            state += estimated_delta[j]
            integrated[begin:end] = state
        metrics["true_previous"].add(targets, previous, required)
        metrics["true_previous_plus_delta"].add(targets, updated, required)
        metrics["true_anchor_integrated"].add(targets, integrated, required)
    true, estimated = np.concatenate(event_true), np.concatenate(event_pred)
    result = {name: metric.result() for name, metric in metrics.items()}
    result["event_delta"] = {
        "correlation": [float(np.corrcoef(true[:, target], estimated[:, target])[0, 1])
                        for target in range(2)],
        "rmse": [float(np.sqrt(np.mean((true[:, target] - estimated[:, target]) ** 2)))
                 for target in range(2)],
        "zero_rate": [float(np.mean(true[:, target] == 0)) for target in range(2)],
        "events": len(true),
    }
    return result


def selfcheck():
    values = np.zeros((N_ROWS, 112), np.float32)
    values[:, :22] = (np.arange(N_ROWS) // 10)[:, None]
    targets = np.column_stack((np.arange(N_ROWS) // 10, -(np.arange(N_ROWS) // 10))).astype(np.float32)
    event, features, delta = rows(values, targets)
    assert np.array_equal(event[:4], [0, 10, 20, 30])
    assert features.shape[1] == 3 * len(PRICE) and np.all(delta == (1, -1))
    print("event-to-target-delta indexing check passed")


def run():
    fit_x = np.load(CACHE / "fit_x.npy", mmap_mode="r")
    fit_y = np.load(CACHE / "fit_y.npy", mmap_mode="r")
    dev_x = np.load(CACHE / "dev_x.npy", mmap_mode="r")
    dev_y = np.load(CACHE / "dev_y.npy", mmap_mode="r")
    if fit_x.shape != (FIT_GROUPS, N_ROWS, 112) or dev_x.shape != (DEV_GROUPS, N_ROWS, 112):
        raise ValueError("unexpected locked pilot cache")
    result = score(dev_x, dev_y, fit(fit_x, fit_y))
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    import sys
    selfcheck() if sys.argv[-1:] == ["selfcheck"] else run()
