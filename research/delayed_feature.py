"""Delayed estimate of the completed price interval's target, as a causal input feature.

Targets stay constant between i0 price-block changes and carry over from interval to interval (knowing the
previous interval's exact target scores WP 0.98). At each i0 price event, a ridge model reconstructs the
interval that just ended from i0 book and trade prices at its last row and at the event row (and their
difference); the estimate is held until the next event (zeros before the first). Fitted on train sequences
outside the 6,000-sequence student pool and outside DEV, so neither student training nor any evaluation
sees in-sample reconstructions.

    python research/delayed_feature.py fit        # writes cache/delayed_feature.npz
    python research/delayed_feature.py --selfcheck
"""
from __future__ import annotations

import argparse

import numpy as np
import pyarrow.parquet as pq

from delayed_oracle import PRICE, solve, starts
from pilot import CACHE, ROOT
from scale import DEV, read_group

PATH = CACHE / "delayed_feature.npz"


def event_rows(values):
    """Event rows after the first, and the reconstruction features for the interval ending before each."""
    event = starts(values)[1:]
    before, after = values[event - 1][:, PRICE], values[event][:, PRICE]
    return event, np.column_stack((before, after, after - before))


def estimates(values, fitted):
    """(rows, 2) delayed target estimates for one sequence of raw features; causal."""
    mean, std, coefficients = fitted
    event, features = event_rows(values)
    out = np.zeros((len(values), 2), np.float32)
    if len(event):
        design = np.column_stack((np.clip((features - mean) / std, -10, 10), np.ones(len(features))))
        values_at = np.clip(design @ coefficients, -2, 2).astype(np.float32)
        index = np.searchsorted(event, np.arange(len(values)), side="right") - 1   # last event at or before row
        out[index >= 0] = values_at[index[index >= 0]]
    return out


def load(path=PATH):
    with np.load(path) as z:
        return z["mean"], z["std"], z["coefficients"]


def fit(count=1500):
    n_train = 10_607
    base = np.sort(np.random.default_rng(123).choice(n_train, 2_000, replace=False))     # ensdistill pool
    extra = np.random.default_rng(124).choice(np.setdiff1d(np.arange(n_train), base), 4_000, replace=False)
    free = np.setdiff1d(np.arange(n_train), np.concatenate([base, extra, DEV]))
    chosen = np.sort(np.random.default_rng(7).choice(free, count, replace=False))
    train = pq.ParquetFile(ROOT / "datasets/train.parquet")
    rows, targets = [], []
    for g in chosen:
        values, y = read_group(train, int(g))
        event, features = event_rows(values)
        rows.append(features)
        targets.append(y[event - 1])
    raw = np.concatenate(rows)
    mean, std = raw.mean(0), raw.std(0)
    std[std < 1e-6] = 1
    coefficients = solve([np.clip((r - mean) / std, -10, 10) for r in rows], targets, raw.shape[1])
    np.savez(PATH, mean=mean, std=std, coefficients=coefficients, groups=chosen)
    print(f"fitted on {len(chosen)} train sequences outside the pool and DEV, {len(raw)} events -> {PATH}")


def selfcheck():
    from delayed_end_oracle import reconstruct
    rng = np.random.default_rng(0)
    values = np.repeat(rng.normal(size=(200, 112)).astype(np.float32), 100, axis=0)[:20_000]
    values[:, 22:44] += rng.normal(size=(20_000, 22)).astype(np.float32)        # volumes move every row
    fitted = (np.zeros(78), np.ones(78), rng.normal(size=(79, 2)) * 0.01)
    out = estimates(values, fitted)
    _, reference = reconstruct(values, fitted)
    event = starts(values)[1:]
    assert np.allclose(out[event], np.clip(reference, -2, 2), atol=1e-6)        # same as the pilot's reconstruction
    assert np.all(out[:event[0]] == 0) and np.all(out[event[0]:event[1]] == out[event[0]])
    changed = values.copy(); changed[15_000:] += 1                                # future change leaves the past alone
    assert np.array_equal(estimates(changed, fitted)[:15_000], out[:15_000])
    print("delayed estimates: pilot-equivalent at events, held between events, causal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("fit",))
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
    elif args.command == "fit":
        fit()
    else:
        parser.print_help()
