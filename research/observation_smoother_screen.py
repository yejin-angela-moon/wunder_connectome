"""Locked pilot for causal smoothing within constant-target price intervals."""
from __future__ import annotations

import json

import numpy as np
import torch

from event_pretrain import BATCH, CHUNK, finetune, normalize
from global_wp import GlobalWP
from pilot import CACHE, DEV_GROUPS, FIT_GROUPS, N_FEATURES, N_ROWS, SEED, WARMUP, make_gru

OUT = CACHE / "observation_smoother_screen.json"
# event_gain controls the update on an i0 price event; row_gain controls all
# later rows in that price interval. No future event or target is used.
GAINS = ((1.0, 0.02), (1.0, 0.05), (1.0, 0.1), (1.0, 0.25),
         (0.5, 0.05), (0.5, 0.1), (0.25, 0.1), (0.0, 0.1))


def smooth(raw, observations, event_gain, row_gain):
    filtered = np.empty_like(observations)
    state = observations[0].astype(np.float64).copy()
    filtered[0] = state
    for row in range(1, len(raw)):
        event = np.any(raw[row, :22] != raw[row - 1, :22])
        gain = event_gain if event else row_gain
        state += gain * (observations[row] - state)
        filtered[row] = state
    return filtered


def score(model, x, y, mean, std):
    metrics = {"raw": GlobalWP(), **{f"e{event:g}_r{row:g}": GlobalWP()
                                    for event, row in GAINS}}
    required = np.arange(N_ROWS) >= WARMUP
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(x), BATCH):
            raw = np.array(x[offset:offset + BATCH])
            values = normalize(raw, mean, std)
            hidden, parts = None, []
            for start in range(0, N_ROWS, CHUNK):
                xb = torch.from_numpy(np.ascontiguousarray(values[:, start:start + CHUNK]))
                prediction, hidden = model(xb, hidden)
                parts.append(prediction.numpy())
            observations = np.concatenate(parts, axis=1)
            for i in range(len(raw)):
                metrics["raw"].add(y[offset + i], observations[i], required)
                for event_gain, row_gain in GAINS:
                    filtered = smooth(raw[i], observations[i], event_gain, row_gain)
                    metrics[f"e{event_gain:g}_r{row_gain:g}"].add(
                        y[offset + i], filtered, required)
    model.train()
    return {name: metric.result() for name, metric in metrics.items()}


def selfcheck():
    raw = np.zeros((8, N_FEATURES), np.float32)
    raw[3:, :22] = 1
    observations = np.arange(16, dtype=np.float32).reshape(8, 2)
    held = smooth(raw, observations, 1, 0)
    assert np.all(held[:3] == observations[0])
    assert np.all(held[3:] == observations[3])
    continuous = smooth(raw, observations, 1, 1)
    assert np.all(continuous == observations)
    print("causal event/row smoothing checks passed")


def main():
    import sys
    if sys.argv[-1:] == ["selfcheck"]:
        selfcheck()
        return
    torch.set_num_threads(1)
    x = np.load(CACHE / "fit_x.npy", mmap_mode="r")
    y = np.load(CACHE / "fit_y.npy", mmap_mode="r")
    dx = np.load(CACHE / "dev_x.npy", mmap_mode="r")
    dy = np.load(CACHE / "dev_y.npy", mmap_mode="r")
    if x.shape != (FIT_GROUPS, N_ROWS, N_FEATURES) or dx.shape != (DEV_GROUPS, N_ROWS, N_FEATURES):
        raise ValueError("unexpected locked pilot cache")
    sampled = np.asarray(x[:, WARMUP::32], np.float64)
    mean = sampled.mean((0, 1)).astype(np.float32)
    std = sampled.std((0, 1)).astype(np.float32)
    std[std < 1e-6] = 1
    torch.manual_seed(SEED)
    model = make_gru(96)
    finetune("control", model, x, y, dx, dy, mean, std, "cpu", 3)
    result = score(model, dx, dy, mean, std)
    baseline = result["raw"]["weighted_pearson"]
    best_name, best_value = max(result.items(), key=lambda item: item[1]["weighted_pearson"])
    output = {"gains": GAINS, "scores": result, "baseline": baseline,
              "best": best_name, "best_score": best_value["weighted_pearson"],
              "gain": best_value["weighted_pearson"] - baseline,
              "passes_gate": best_value["weighted_pearson"] >= baseline + 0.005}
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
