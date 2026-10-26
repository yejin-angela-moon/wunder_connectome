"""Causal diagnostic: reconstruct the prior interval after its next event arrives."""
from __future__ import annotations

import json

import numpy as np

from global_wp import GlobalWP
from pilot import CACHE, DEV_GROUPS, FIT_GROUPS, N_ROWS, WARMUP

OUT = CACHE / "delayed_oracle.json"
PRICE = np.r_[0:22, 44:48]
RIDGE = 100.0


def starts(values):
    price = values[:, :22]
    return np.flatnonzero(np.r_[True, np.any(price[1:] != price[:-1], axis=1)])


def solve(rows, targets, width, ridge=RIDGE):
    gram = np.zeros((2, width + 1, width + 1), np.float64)
    rhs = np.zeros((2, width + 1), np.float64)
    for features, y in zip(rows, targets):
        design = np.column_stack((features.astype(np.float64), np.ones(len(features))))
        y = np.clip(y.astype(np.float64), -2, 2)
        for target in range(2):
            weight = np.sqrt(np.abs(y[:, target]) + 1e-4)
            weighted = design * weight[:, None]
            with np.errstate(all="ignore"):
                update_gram = weighted.T @ weighted
                update_rhs = weighted.T @ (y[:, target] * weight)
            if not np.isfinite(update_gram).all() or not np.isfinite(update_rhs).all():
                raise ValueError("nonfinite delayed-oracle sufficient statistics")
            gram[target] += update_gram
            rhs[target] += update_rhs
    coefficients = np.empty((width + 1, 2), np.float64)
    for target in range(2):
        gram[target].flat[::width + 2] += ridge
        coefficients[:, target] = np.linalg.solve(gram[target], rhs[target])
    return coefficients


def oracle_rows(values, targets):
    event = starts(values)
    current = values[event[:-1]][:, PRICE]
    future_change = values[event[1:]][:, PRICE] - current
    return np.column_stack((current, future_change)), targets[event[:-1]]


def fit_oracle(x, y):
    pairs = [oracle_rows(values, targets) for values, targets in zip(x, y)]
    raw = np.concatenate([pair[0] for pair in pairs])
    mean, std = raw.mean(0), raw.std(0)
    std[std < 1e-6] = 1
    normalized = [np.clip((pair[0] - mean) / std, -10, 10) for pair in pairs]
    return mean, std, solve(normalized, [pair[1] for pair in pairs], 2 * len(PRICE))


def reconstruct_previous(values, oracle):
    event = starts(values)
    current = values[event[:-1]][:, PRICE]
    observed_change = values[event[1:]][:, PRICE] - current
    features = np.column_stack((current, observed_change))
    mean, std, coefficients = oracle
    design = np.column_stack((np.clip((features - mean) / std, -10, 10),
                              np.ones(len(features))))
    with np.errstate(all="ignore"):
        prediction = design @ coefficients
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite delayed-oracle reconstruction")
    return event, prediction


def transition_rows(values, targets, oracle):
    event, previous_target = reconstruct_previous(values, oracle)
    current = values[event[1:]][:, PRICE]
    event_change = current - values[event[:-1]][:, PRICE]
    features = np.column_stack((previous_target, current, event_change))
    return features, targets[event[1:]]


def fit_transition(x, y):
    feature_rows, target_rows = [], []
    indices = np.arange(len(x))
    for held in (indices[::2], indices[1::2]):
        train = np.setdiff1d(indices, held)
        oracle = fit_oracle(x[train], y[train])
        for index in held:
            features, targets = transition_rows(x[index], y[index], oracle)
            feature_rows.append(features); target_rows.append(targets)
    raw = np.concatenate(feature_rows)
    mean, std = raw.mean(0), raw.std(0)
    std[std < 1e-6] = 1
    normalized = [np.clip((features - mean) / std, -10, 10)
                  for features in feature_rows]
    coefficients = solve(normalized, target_rows, raw.shape[1])
    return mean, std, coefficients


def predictions(values, oracle, transition):
    event, previous = reconstruct_previous(values, oracle)
    current = values[event[1:]][:, PRICE]
    event_change = current - values[event[:-1]][:, PRICE]
    features = np.column_stack((previous, current, event_change))
    mean, std, coefficients = transition
    design = np.column_stack((np.clip((features - mean) / std, -10, 10),
                              np.ones(len(features))))
    with np.errstate(all="ignore"):
        estimate = design @ coefficients
    if not np.isfinite(estimate).all():
        raise ValueError("nonfinite delayed transition prediction")
    lagged = np.zeros((N_ROWS, 2), np.float32)
    fitted = np.zeros((N_ROWS, 2), np.float32)
    for j, begin in enumerate(event[1:]):
        end = event[j + 2] if j + 2 < len(event) else N_ROWS
        lagged[begin:end] = previous[j]
        fitted[begin:end] = estimate[j]
    return lagged, fitted


def score(x, y, oracle, transition):
    metrics = {name: GlobalWP() for name in ("reconstructed_lag", "transition")}
    required = np.arange(N_ROWS) >= WARMUP
    for values, targets in zip(x, y):
        lagged, fitted = predictions(values, oracle, transition)
        metrics["reconstructed_lag"].add(targets, lagged, required)
        metrics["transition"].add(targets, fitted, required)
    return {name: metric.result() for name, metric in metrics.items()}


def selfcheck():
    values = np.zeros((N_ROWS, 112), np.float32)
    values[:, :22] = (np.arange(N_ROWS) // 10)[:, None]
    event = starts(values)
    assert np.array_equal(event[:4], [0, 10, 20, 30])
    coefficients = np.zeros((2 * len(PRICE) + 1, 2))
    coefficients[-1] = (1, -1)
    oracle = (np.zeros(2 * len(PRICE)), np.ones(2 * len(PRICE)), coefficients)
    found, previous = reconstruct_previous(values, oracle)
    assert np.array_equal(found, event) and np.all(previous == (1, -1))
    print("delayed event reconstruction indexing check passed")


def run():
    fit_x = np.load(CACHE / "fit_x.npy", mmap_mode="r")
    fit_y = np.load(CACHE / "fit_y.npy", mmap_mode="r")
    dev_x = np.load(CACHE / "dev_x.npy", mmap_mode="r")
    dev_y = np.load(CACHE / "dev_y.npy", mmap_mode="r")
    if fit_x.shape != (FIT_GROUPS, N_ROWS, 112) or dev_x.shape != (DEV_GROUPS, N_ROWS, 112):
        raise ValueError("unexpected locked pilot cache")
    oracle = fit_oracle(fit_x, fit_y)
    transition = fit_transition(fit_x, fit_y)
    result = score(dev_x, dev_y, oracle, transition)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    import sys
    selfcheck() if sys.argv[-1:] == ["selfcheck"] else run()
