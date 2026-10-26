"""Cross-fit whether explicit within-sequence time improves a cached ensemble."""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from ensemble_screen import MEMBERS, load, wp
from pilot import CACHE, N_ROWS, ROOT

OUT = CACHE / "time_screen.json"


def design(prediction, step, time):
    if not time:
        return np.column_stack((prediction, np.ones(len(step))))
    u = step / (N_ROWS - 1)
    return np.column_stack((prediction, u, u**2, prediction * u, np.ones(len(step))))


def fit(prediction, y, step, time):
    coefficients = []
    for target in range(2):
        x = design(prediction[:, target], step, time)
        value = np.clip(y[:, target].astype(np.float64), -2, 2)
        weight = np.abs(value)
        with np.errstate(all="ignore"):
            gram = x.T @ (weight[:, None] * x)
            rhs = x.T @ (weight * value)
        if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
            raise ValueError("nonfinite time-screen sufficient statistics")
        gram.flat[::len(gram) + 1] += 1e-6 * np.trace(gram) / len(gram)
        coefficients.append(np.linalg.solve(gram, rhs))
    return coefficients


def predict(prediction, step, coefficients, time):
    with np.errstate(all="ignore"):
        result = np.column_stack([design(prediction[:, target], step, time) @ coefficients[target]
                                  for target in range(2)])
    if not np.isfinite(result).all():
        raise ValueError("nonfinite time-screen prediction")
    return result


def selfcheck():
    step = np.arange(10)
    p = np.ones((10, 2))
    y = np.column_stack((step, -step))
    coefficients = fit(p, y, step, True)
    assert predict(p, step, coefficients, True).shape == y.shape
    print("time residual screen self-check passed")


def main():
    masks = pq.read_table(ROOT / "datasets/valid_mask.parquet", columns=["is_scored"])
    masks = masks["is_scored"].combine_chunks().to_numpy(
        zero_copy_only=False).reshape(-1, N_ROWS)
    names = list(MEMBERS)
    members = [names.index(name) for name in ("split", "sd71", "sd72", "g71", "g72")]
    halves = []
    for half in (0, 1):
        p, y, _ = load(half)
        groups = np.arange(len(masks))[np.arange(len(masks)) % 2 != half]
        step = np.concatenate([np.flatnonzero(masks[group]) for group in groups])
        if len(step) != len(y):
            raise ValueError("cached rows do not match validation masks")
        halves.append((p[:, members].mean(1), y, step))
    report = {"plain": [], "calibration": [], "time": []}
    for test_half in (0, 1):
        train_p, train_y, train_step = halves[1 - test_half]
        test_p, test_y, test_step = halves[test_half]
        report["plain"].append(wp(test_y, test_p))
        for name, use_time in (("calibration", False), ("time", True)):
            coefficients = fit(train_p, train_y, train_step, use_time)
            report[name].append(wp(test_y, predict(test_p, test_step, coefficients, use_time)))
    result = {name: {"halves": scores, "mean": float(np.mean(scores))}
              for name, scores in report.items()}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import sys
    selfcheck() if sys.argv[-1:] == ["selfcheck"] else main()
