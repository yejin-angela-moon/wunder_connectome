"""Mask-aware linear readout over frozen full-FIT GRU hidden states.

The official score uses only rows selected by the hidden `is_scored` mask. On
public validation those rows behave very differently from the rest, so this
script fits a linear readout of several frozen GRUs' hidden states with the
scored rows emphasized. Selection uses two sequence-disjoint validation halves
(even/odd row group); the final readout is refitted on all validation rows.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pyarrow.parquet as pq
import torch

from global_wp import GlobalWP
from oos_global import read_group as read_valid_group
from pilot import CACHE, N_ROWS, ROOT, WARMUP, make_gru

OUT = CACHE / "stack"
FULL = CACHE / "improve/full_fit"
DISTILL = CACHE / "advance/full_fit/distill"
# name -> (checkpoint, uses standardized input)
MODELS = {
    "v1": (DISTILL / "model.pt", True),
    "v2": (DISTILL / "model_alpha75.pt", True),
    "t_norm": (FULL / "gru96_norm5ep/scaled_gru.pt", True),
    "t_s17": (FULL / "gru96_seed17/scaled_gru.pt", False),
    "t_s37": (FULL / "gru96_seed37/scaled_gru.pt", False),
}
UNSCORED_SAMPLE = 0.06
BATCH = 32


def load_models():
    with np.load(FULL / "gru96_norm5ep/gru_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    models = {}
    for name, (path, _) in MODELS.items():
        model = make_gru(96)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        models[name] = model.eval()
    return mean, std, models


def extract():
    """Hidden states on all scored rows plus a fixed sample of unscored rows."""
    OUT.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    mean, std, models = load_models()
    rng = np.random.default_rng(20260924)
    required = np.arange(N_ROWS) >= WARMUP
    parts = {key: [] for key in ("h", "p", "y", "scored", "half", "seq")}
    torch.set_num_threads(8)
    groups = range(parquet.num_row_groups)
    with torch.no_grad():
        for offset in range(0, len(groups), BATCH):
            batch = groups[offset:offset + BATCH]
            data = [read_valid_group(parquet, g) for g in batch]
            raw = np.stack([d[2] for d in data])
            normalized = np.clip((raw - mean) / std, -10, 10)
            hidden, preds = [], []
            for name, (_, standardized) in MODELS.items():
                values = torch.from_numpy(normalized if standardized else raw)
                h, _ = models[name].rnn(values)
                hidden.append(h.numpy())
                preds.append(models[name].head(h).numpy())
            hidden = np.concatenate(hidden, axis=2)              # B,T,5*96
            preds = np.stack(preds, axis=2)                      # B,T,5,2
            for i, (g, (seq, need, _, y, mask)) in enumerate(zip(batch, data)):
                if not np.array_equal(need, required):
                    raise ValueError("unexpected prediction mask")
                keep = mask | (need & ~mask & (rng.random(N_ROWS) < UNSCORED_SAMPLE))
                parts["h"].append(hidden[i, keep].astype(np.float16))
                parts["p"].append(preds[i, keep])
                parts["y"].append(y[keep])
                parts["scored"].append(mask[keep])
                parts["half"].append(np.full(keep.sum(), g % 2, np.int8))
                parts["seq"].append(np.full(keep.sum(), g, np.int16))
            print("extracted", offset + len(batch), flush=True)
    np.savez(OUT / "valid_hidden.npz", **{k: np.concatenate(v) for k, v in parts.items()})


def wp(y, p, rows):
    acc = GlobalWP()
    acc.add(y[rows], p[rows], np.ones(rows.sum(), bool))
    return acc.result()["weighted_pearson"]


def half_stats(h, y, weight, rows, columns, chunk=250_000):
    """Per-target weighted normal equations over `rows`; weight excludes |y|."""
    width = len(columns) + 1
    a, b = np.zeros((2, width, width)), np.zeros((2, width))
    for start in range(0, len(rows), chunk):
        r = rows[start:start + chunk]
        x = np.ones((len(r), width))
        x[:, :-1] = h[r][:, columns]
        target = np.clip(y[r].astype(np.float64), -2, 2)
        for k in range(2):
            xw = x * (np.abs(target[:, k]) * weight[r])[:, None]
            a[k] += xw.T @ x
            b[k] += xw.T @ target[:, k]
    return a, b


def solve(a, b, lam):
    beta = []
    for k in range(2):
        m = a[k].copy()
        m[np.diag_indices_from(m)] += lam * np.trace(m) / len(m)
        beta.append(np.linalg.solve(m, b[k]))
    return np.stack(beta, 1)


def readout(h, rows, columns, beta, chunk=250_000):
    out = np.empty((len(rows), 2), np.float32)
    for start in range(0, len(rows), chunk):
        r = rows[start:start + chunk]
        out[start:start + len(r)] = h[r][:, columns].astype(np.float64) @ beta[:-1] + beta[-1]
    return out


def evaluate():
    d = np.load(OUT / "valid_hidden.npz")
    h, p, y, scored, half = d["h"], d["p"], d["y"], d["scored"], d["half"]
    names = list(MODELS)
    report = {"single": {}, "blend": {}, "readout": {}}
    for j, name in enumerate(names):
        report["single"][name] = [wp(y, p[:, j], scored & (half == f)) for f in (0, 1)]
    blends = {"v1+teacher3": ["v1", "t_norm", "t_s17", "t_s37"], "teacher3": names[2:],
              "all5": names, "v1+v2": ["v1", "v2"]}
    for label, members in blends.items():
        idx = [names.index(m) for m in members]
        report["blend"][label] = [wp(y, p[:, idx].mean(1), scored & (half == f)) for f in (0, 1)]
    weights = {"all_rows": np.where(scored, 1.0, 1 / UNSCORED_SAMPLE),
               "scored_x16": np.ones(len(y)), "scored_only": scored.astype(np.float64)}
    sets = {"v1": [0], "v1+teacher3": [0, 2, 3, 4], "all5": [0, 1, 2, 3, 4]}
    for set_name, members in sets.items():
        columns = np.concatenate([np.arange(m * 96, (m + 1) * 96) for m in members])
        for weight_name, weight in weights.items():
            stats = [half_stats(h, y, weight, np.flatnonzero(half == f), columns) for f in (0, 1)]
            for lam in (1e-4, 1e-2):
                scores = []
                for f in (0, 1):
                    test = np.flatnonzero((half == f) & scored)
                    beta = solve(*stats[1 - f], lam)
                    scores.append(wp(y[test], readout(h, test, columns, beta), np.ones(len(test), bool)))
                report["readout"][f"{set_name}|{weight_name}|lam{lam:g}"] = scores
                print(set_name, weight_name, lam, np.round(scores, 5), round(np.mean(scores), 5), flush=True)
    for section in report.values():
        for key, value in section.items():
            section[key] = {"halves": value, "mean": float(np.mean(value))}
    (OUT / "evaluate.json").write_text(json.dumps(report, indent=2) + "\n")
    for section, values in report.items():
        for key, value in values.items():
            print(f"{section:8s} {key:32s} {value['mean']:.5f} {np.round(value['halves'], 5)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("extract", "evaluate"))
    {"extract": extract, "evaluate": evaluate}[parser.parse_args().command]()
