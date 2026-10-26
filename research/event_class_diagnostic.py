"""Train-only oracle diagnostic for coarse next-price-event classes."""
from __future__ import annotations

from collections import Counter
import json

import numpy as np

from global_wp import GlobalWP
from pilot import CACHE, DEV_GROUPS, FIT_GROUPS, N_ROWS, WARMUP

OUT = CACHE / "event_class_diagnostic.json"
PRICE = np.r_[0:22, 44:48]
STEP = 8
RIDGE = 100.0


def event_patterns(values):
    price = values[:, :22]
    starts = np.flatnonzero(np.r_[True, np.any(price[1:] != price[:-1], axis=1)])
    return [np.sign(row).astype(np.int8).tobytes() for row in np.diff(price[starts], axis=0)]


def design(values, mapping):
    price = values[:, :22]
    starts = np.flatnonzero(np.r_[True, np.any(price[1:] != price[:-1], axis=1)])
    rows = np.arange(WARMUP, N_ROWS, STEP)
    segment = np.searchsorted(starts, rows, side="right") - 1
    keep = segment + 1 < len(starts)
    rows, segment = rows[keep], segment[keep]
    delta = price[starts[segment + 1]] - price[rows]
    classes = np.fromiter((mapping.get(np.sign(row).astype(np.int8).tobytes(), len(mapping))
                           for row in delta), np.int64, count=len(delta))
    onehot = np.eye(len(mapping) + 1, dtype=np.float32)[classes]
    return np.column_stack((values[rows][:, PRICE], onehot)).astype(np.float64), rows


def fit(x, y, mapping):
    width = len(PRICE) + len(mapping) + 1
    total, square, count = np.zeros(width), np.zeros(width), 0
    for values in x:
        features, _ = design(values, mapping)
        total += features.sum(0); square += (features * features).sum(0); count += len(features)
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean * mean, 1e-6))
    gram = np.zeros((2, width + 1, width + 1), np.float64)
    rhs = np.zeros((2, width + 1), np.float64)
    for values, targets in zip(x, y):
        features, rows = design(values, mapping)
        features = np.column_stack((np.clip((features - mean) / std, -10, 10),
                                    np.ones(len(features))))
        targets = np.clip(targets[rows].astype(np.float64), -2, 2)
        for target in range(2):
            weight = np.sqrt(np.abs(targets[:, target]) + 1e-4)
            weighted = features * weight[:, None]
            gram[target] += weighted.T @ weighted
            rhs[target] += weighted.T @ (targets[:, target] * weight)
    coefficients = np.empty((width + 1, 2))
    for target in range(2):
        gram[target].flat[::width + 2] += RIDGE
        coefficients[:, target] = np.linalg.solve(gram[target], rhs[target])
    return mean, std, coefficients


def score(x, y, mapping, fitted):
    mean, std, coefficients = fitted
    metric = GlobalWP()
    for values, targets in zip(x, y):
        features, rows = design(values, mapping)
        features = np.column_stack((np.clip((features - mean) / std, -10, 10),
                                    np.ones(len(features))))
        metric.add(targets[rows], features @ coefficients, np.ones(len(rows), bool))
    return metric.result()


def selfcheck():
    values = np.zeros((N_ROWS, 112), np.float32)
    values[:, :22] = (np.arange(N_ROWS) // 10)[:, None]
    mapping = {np.ones(22, np.int8).tobytes(): 0}
    features, rows = design(values, mapping)
    assert rows[0] == WARMUP and features.shape[1] == len(PRICE) + 2
    assert np.all(features[:, -2] == 1) and np.all(features[:, -1] == 0)
    print("next-event class indexing check passed")


def run():
    fit_x = np.load(CACHE / "fit_x.npy", mmap_mode="r")
    fit_y = np.load(CACHE / "fit_y.npy", mmap_mode="r")
    dev_x = np.load(CACHE / "dev_x.npy", mmap_mode="r")
    dev_y = np.load(CACHE / "dev_y.npy", mmap_mode="r")
    if fit_x.shape != (FIT_GROUPS, N_ROWS, 112) or dev_x.shape != (DEV_GROUPS, N_ROWS, 112):
        raise ValueError("unexpected locked pilot cache")
    counts = Counter(pattern for values in fit_x for pattern in event_patterns(values))
    total = sum(counts.values())
    result = []
    for size in (0, 4, 8, 16, 32):
        mapping = {pattern: index for index, (pattern, _) in enumerate(counts.most_common(size))}
        fitted = fit(fit_x, fit_y, mapping)
        value = score(dev_x, dev_y, mapping, fitted)
        coverage = sum(count for _, count in counts.most_common(size)) / total if size else 0
        row = {"classes": size, "fit_event_coverage": coverage, **value}
        result.append(row)
        print(row, flush=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    import sys
    selfcheck() if sys.argv[-1:] == ["selfcheck"] else run()
