"""Cross-fitted screen for causal within-price-interval i0 flow features."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pyarrow.parquet as pq

from global_wp import GlobalWP
from pilot import CACHE, ROOT
from stack import FULL

PREDICTIONS = {
    0: CACHE / "maskft/pred_v1_u0.06_lr0.0002_s7_e2-3_h0.npz",  # odd groups
    1: CACHE / "maskft/pred_v1_u0.06_lr0.0002_s7_e2-3_h1.npz",  # even groups
}
def columns(instrument):
    prefix = f"i{instrument}"
    return [f"{prefix}_p{i}" for i in range(22)] + [f"{prefix}_v{i}" for i in range(22)] + [
        f"{prefix}_dp{i}" for i in range(4)] + [f"{prefix}_dv{i}" for i in range(4)]


WIDTH = 107


def flow_features(raw, mean, std):
    values = np.clip((raw - mean) / std, -10, 10)
    one = np.zeros_like(values)
    one[1:] = values[1:] - values[:-1]
    changed = np.r_[True, np.any(raw[1:, :22] != raw[:-1, :22], axis=1)]
    starts = np.flatnonzero(changed)
    segment = np.cumsum(changed) - 1
    since = values - values[starts[segment]]
    age = np.log1p(np.arange(len(raw)) - starts[segment])[:, None]
    return np.column_stack((one, since, age))


def solve(gram, rhs):
    coefficients = np.empty((gram.shape[1], 2))
    for target in range(2):
        matrix = gram[target].copy()
        matrix.flat[::len(matrix) + 1] += 1e-4 * np.trace(matrix) / len(matrix)
        coefficients[:, target] = np.linalg.solve(matrix, rhs[target])
    return coefficients


def run(instrument=0):
    features = columns(instrument)
    output = CACHE / f"flow_screen_i{instrument}.json"
    with np.load(FULL / "gru96_norm5ep/gru_standardizer.npz") as source:
        sl = slice(52 * instrument, 52 * (instrument + 1))
        mean, std = source["mean"][sl], source["std"][sl]
    cached = {}
    for half, path in PREDICTIONS.items():
        with np.load(path) as source:
            cached[half] = {key: source[key] for key in ("p", "y")}
    pointers = {0: 0, 1: 0}
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    gram = np.zeros((2, 2, 2, WIDTH + 1, WIDTH + 1), np.float64)
    rhs = np.zeros((2, 2, 2, WIDTH + 1), np.float64)

    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group, columns=["is_scored", *features, "t0", "t1"],
                                       use_threads=False)
        mask = table["is_scored"].to_numpy()
        raw = np.column_stack([table[c].to_numpy() for c in features]).astype(np.float32)
        y = np.column_stack((table["t0"].to_numpy(), table["t1"].to_numpy()))[mask]
        source = 1 if group % 2 == 0 else 0
        start, end = pointers[source], pointers[source] + len(y)
        prediction = cached[source]["p"][start:end]
        expected = cached[source]["y"][start:end]
        pointers[source] = end
        if not np.array_equal(y, expected):
            raise ValueError(f"cached prediction order mismatch at group {group}")
        extra = flow_features(raw, mean, std)[mask]
        full = np.column_stack((prediction, extra, np.ones(len(y))))[::4].astype(np.float64)
        control = np.column_stack((prediction, np.ones(len(y))))[::4].astype(np.float64)
        target = np.clip(y[::4].astype(np.float64), -2, 2)
        half = group % 2
        for k in range(2):
            weight = np.abs(target[:, k])
            for j, design in enumerate((control, full)):
                weighted = design * weight[:, None]
                with np.errstate(all="ignore"):
                    gram[j, half, k, :len(design[0]), :len(design[0])] += weighted.T @ design
                    rhs[j, half, k, :len(design[0])] += weighted.T @ target[:, k]
        if group % 200 == 199:
            print("fit groups", group + 1, flush=True)

    coefficients = [[solve(gram[j, half, :, :width, :width], rhs[j, half, :, :width])
                     for half in (0, 1)]
                    for j, width in enumerate((3, WIDTH + 1))]
    metrics = [[GlobalWP(), GlobalWP()] for _ in range(2)]
    pointers = {0: 0, 1: 0}
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group, columns=["is_scored", *features, "t0", "t1"],
                                       use_threads=False)
        mask = table["is_scored"].to_numpy()
        raw = np.column_stack([table[c].to_numpy() for c in features]).astype(np.float32)
        y = np.column_stack((table["t0"].to_numpy(), table["t1"].to_numpy()))[mask]
        source = 1 if group % 2 == 0 else 0
        start, end = pointers[source], pointers[source] + len(y)
        prediction = cached[source]["p"][start:end]
        pointers[source] = end
        designs = (np.column_stack((prediction, np.ones(len(y)))),
                   np.column_stack((prediction, flow_features(raw, mean, std)[mask], np.ones(len(y)))))
        half = group % 2
        for j, design in enumerate(designs):
            with np.errstate(all="ignore"):
                prediction = design @ coefficients[j][1 - half]
            metrics[j][half].add(y, prediction, np.ones(len(y), bool))
        if group % 200 == 199:
            print("score groups", group + 1, flush=True)
    report = {}
    candidate = f"plus_i{instrument}_flow"
    for name, pair in zip(("prediction_control", candidate), metrics):
        halves = [metric.result()["weighted_pearson"] for metric in pair]
        report[name] = {"halves": halves, "mean": float(np.mean(halves))}
    report["gain"] = report[candidate]["mean"] - report["prediction_control"]["mean"]
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(report)


def selfcheck():
    raw = np.zeros((8, 52), np.float32)
    raw[3:, :22] = 1
    raw[6:, :22] = 2
    raw[:, 22:] = np.arange(8)[:, None]
    values = flow_features(raw, np.zeros(52), np.ones(52))
    assert values.shape == (8, 105)
    assert values[3, -1] == 0 and np.isclose(values[5, -1], np.log1p(2))
    assert np.all(values[4, 52:74] == 0)
    print("causal flow feature checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("run", "selfcheck"), default="run")
    parser.add_argument("--instrument", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    selfcheck() if args.command == "selfcheck" else run(args.instrument)
