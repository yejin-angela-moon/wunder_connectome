"""Train-only nonlinear diagnostic for delayed previous-target reconstruction."""
from __future__ import annotations

import json

import numpy as np
import torch

from delayed_end_oracle import reconstruction_rows
from global_wp import GlobalWP
from pilot import CACHE, DEV_GROUPS, FIT_GROUPS, N_ROWS, SEED, WARMUP

OUT = CACHE / "delayed_mlp_diagnostic.json"
MODEL = CACHE / "delayed_mlp_diagnostic.pt"
BATCH = 4_096
EPOCHS = 5


def samples(x, y):
    parts = [reconstruction_rows(values, targets) for values, targets in zip(x, y)]
    return np.concatenate([part[1] for part in parts]).astype(np.float32), np.concatenate(
        [part[2] for part in parts]).astype(np.float32)


def make_model(width):
    return torch.nn.Sequential(torch.nn.Linear(width, 64), torch.nn.Tanh(),
                               torch.nn.Linear(64, 2))


def fit(x, y):
    raw, target = samples(x, y)
    mean = raw.astype(np.float64).mean(0).astype(np.float32)
    std = raw.astype(np.float64).std(0).astype(np.float32)
    std[std < 1e-6] = 1
    values = np.clip((raw - mean) / std, -10, 10)
    target = np.clip(target, -2, 2)
    torch.manual_seed(SEED)
    model = make_model(values.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(SEED)
    for epoch in range(EPOCHS):
        losses = []
        order = rng.permutation(len(values))
        for start in range(0, len(values), BATCH):
            indices = order[start:start + BATCH]
            xb = torch.from_numpy(np.ascontiguousarray(values[indices]))
            yb = torch.from_numpy(np.ascontiguousarray(target[indices]))
            optimizer.zero_grad()
            prediction = model(xb)
            weight = yb.abs()
            loss = ((prediction - yb).square() * weight).sum() / weight.sum().clamp_min(1e-6)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("delayed_mlp epoch", epoch + 1, "loss", round(float(np.mean(losses)), 6), flush=True)
    return mean, std, model.eval()


def predict(values, fitted):
    mean, std, model = fitted
    event, features, _ = reconstruction_rows(values, np.zeros((N_ROWS, 2), np.float32))
    x = np.clip((features - mean) / std, -10, 10).astype(np.float32)
    with torch.no_grad():
        prediction = model(torch.from_numpy(x)).numpy()
    return event, prediction


def score(x, y, fitted):
    event_metric, assigned_metric = GlobalWP(), GlobalWP()
    for values, targets in zip(x, y):
        event, prediction = predict(values, fitted)
        end = event[1:] - 1
        event_metric.add(targets[end], prediction, np.ones(len(end), bool))
        assigned = np.zeros((N_ROWS, 2), np.float32)
        for index, begin in enumerate(event[1:]):
            stop = event[index + 2] if index + 2 < len(event) else N_ROWS
            assigned[begin:stop] = prediction[index]
        assigned_metric.add(targets, assigned, np.arange(N_ROWS) >= WARMUP)
    return {"event_previous": event_metric.result(),
            "assigned_current": assigned_metric.result()}


def selfcheck():
    values = np.zeros((N_ROWS, 112), np.float32)
    values[:, :22] = (np.arange(N_ROWS) // 10)[:, None]
    target = np.column_stack((np.arange(N_ROWS) // 10,
                              -(np.arange(N_ROWS) // 10))).astype(np.float32)
    features, y = samples(values[None], target[None])
    assert len(features) == N_ROWS // 10 - 1 and np.array_equal(y[:3], target[[9, 19, 29]])
    print("nonlinear delayed-reconstruction samples align")


def run():
    torch.set_num_threads(1)
    x = np.load(CACHE / "fit_x.npy", mmap_mode="r")
    y = np.load(CACHE / "fit_y.npy", mmap_mode="r")
    dx = np.load(CACHE / "dev_x.npy", mmap_mode="r")
    dy = np.load(CACHE / "dev_y.npy", mmap_mode="r")
    if x.shape[0] != FIT_GROUPS or dx.shape[0] != DEV_GROUPS:
        raise ValueError("unexpected locked pilot cache")
    fitted = fit(x, y)
    result = score(dx, dy, fitted)
    result["passes_reconstruction_gate"] = result["event_previous"]["weighted_pearson"] >= 0.56
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    torch.save({"mean": fitted[0], "std": fitted[1], "state": fitted[2].state_dict()}, MODEL)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    import sys
    selfcheck() if sys.argv[-1:] == ["selfcheck"] else run()
