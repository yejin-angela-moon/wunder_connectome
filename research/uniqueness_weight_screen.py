"""Pilot AFML-style uniqueness weighting for persistent target runs.

The competition does not expose explicit label start/end times.  This screen
therefore uses inverse target-run length only as a train-time proxy, never as
an inference feature.  It is deliberately not called exact label uniqueness.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import pyarrow.parquet as pq

from event_pretrain import normalize
from group_cv import folds
from global_wp import GlobalWP
from maskft import detach_state
from pilot import CACHE, DEV_GROUPS, FIT_GROUPS, N_FEATURES, N_ROWS, ROOT, SEED, WARMUP
from scale import read_group
from warmup_context_screen import CONTROL


BATCH = 16
CHUNK = 512
OUT = CACHE / "uniqueness_weight"


def run_weights(target, power):
    """Inverse persistence-run length, normalized to mean one per sequence."""
    result = np.ones(target.shape[:2], np.float32)
    for batch in range(len(target)):
        per_target = []
        for column in range(2):
            values = target[batch, :, column]
            changed = np.r_[True, values[1:] != values[:-1]]
            starts = np.flatnonzero(changed)
            lengths = np.diff(np.r_[starts, len(values)])
            per_target.append(np.repeat(lengths.astype(np.float32) ** (-power), lengths))
        result[batch] = np.mean(per_target, axis=0)
        if WARMUP < len(values):
            result[batch] /= result[batch, WARMUP:].mean()
            result[batch, :WARMUP] = 0
        else:
            # Keep the helper well-defined for tiny unit-test arrays.
            result[batch] = 0
    return result


def make_gru():
    return torch.nn.ModuleDict({
        "rnn": torch.nn.GRU(N_FEATURES, 96, batch_first=True),
        "head": torch.nn.Linear(96, 2),
    })


def forward(model, values, hidden=None):
    states, hidden = model["rnn"](values, hidden)
    return model["head"](states), hidden


def score(model, x, y, mean, std, device):
    metric = GlobalWP()
    model.eval()
    required = np.arange(N_ROWS) >= WARMUP
    with torch.no_grad():
        for offset in range(0, len(x), BATCH):
            values = torch.from_numpy(normalize(np.array(x[offset:offset + BATCH]), mean, std)).to(device)
            hidden, pieces = None, []
            for start in range(0, N_ROWS, CHUNK):
                prediction, hidden = forward(model, values[:, start:start + CHUNK], hidden)
                pieces.append(prediction.cpu().numpy())
            for target, estimate in zip(y[offset:offset + BATCH], np.concatenate(pieces, 1)):
                metric.add(target, estimate, required)
    model.train()
    return metric.result()


def run(power, epochs, seed, device):
    x = np.load(CACHE / "fit_x.npy", mmap_mode="r")
    y = np.load(CACHE / "fit_y.npy", mmap_mode="r")
    dx = np.load(CACHE / "dev_x.npy", mmap_mode="r")
    dy = np.load(CACHE / "dev_y.npy", mmap_mode="r")
    sampled = np.asarray(x[:, WARMUP::32], np.float64)
    mean = sampled.mean((0, 1)).astype(np.float32)
    std = sampled.std((0, 1)).astype(np.float32)
    std[std < 1e-6] = 1
    torch.manual_seed(seed)
    model = make_gru().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    history = [score(model, dx, dy, mean, std, device)]
    print("uniqueness power", power, "epoch 0", history[-1], flush=True)
    for epoch in range(epochs):
        losses = []
        order = rng.permutation(len(x))
        for offset in range(0, len(x), BATCH):
            batch = order[offset:offset + BATCH]
            values = torch.from_numpy(normalize(np.array(x[batch]), mean, std)).to(device)
            targets_np = np.clip(np.array(y[batch]), -2, 2).astype(np.float32)
            targets = torch.from_numpy(targets_np).to(device)
            weights = torch.from_numpy(run_weights(targets_np, power)).to(device)
            hidden = None
            for start in range(0, N_ROWS, CHUNK):
                sl = slice(start, start + CHUNK)
                optimizer.zero_grad()
                prediction, hidden = forward(model, values[:, sl], hidden)
                begin = WARMUP if start == 0 else 0
                weight = weights[:, sl, None] * targets[:, sl].abs()
                loss = ((prediction[:, begin:] - targets[:, sl][:, begin:]).square()
                        * weight[:, begin:]).sum() / weight[:, begin:].sum().clamp_min(1e-6)
                loss.backward(); optimizer.step()
                hidden = detach_state(hidden)
                losses.append(float(loss.detach()))
        history.append(score(model, dx, dy, mean, std, device))
        print("uniqueness power", power, "epoch", epoch + 1,
              "loss", round(float(np.mean(losses)), 6), history[-1], flush=True)
    fixed = history[-1]["weighted_pearson"]
    result = {"power": power, "seed": seed, "epochs": epochs, "control": CONTROL,
              "history": history, "fixed_gain": fixed - CONTROL,
              "passes_gate": fixed >= CONTROL + 0.005,
              "interpretation": "inverse target persistence-run proxy; exact label spans unavailable"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"power{power:g}_seed{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def read_batch(parquet, groups):
    data = [read_group(parquet, int(group)) for group in groups]
    return np.stack([row[0] for row in data]), np.stack([row[1] for row in data])


def score_groups(model, parquet, groups, mean, std, device):
    metric = GlobalWP()
    required = np.arange(N_ROWS) >= WARMUP
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(groups), BATCH):
            raw, target = read_batch(parquet, groups[offset:offset + BATCH])
            values = torch.from_numpy(normalize(raw, mean, std)).to(device)
            hidden, pieces = None, []
            for start in range(0, N_ROWS, CHUNK):
                prediction, hidden = forward(model, values[:, start:start + CHUNK], hidden)
                pieces.append(prediction.cpu().numpy())
            for y_row, p_row in zip(target, np.concatenate(pieces, 1)):
                metric.add(y_row, p_row, required)
    model.train()
    return metric.result()


def run_fold(power, fold, epochs, seed, device):
    parquet = pq.ParquetFile(ROOT / "datasets/train.parquet")
    number, fit_groups, validation_groups = next(plan for plan in folds() if plan[0] == fold)
    folder = CACHE / "improve" / f"fold_{number}" / "gru96_norm5ep"
    with np.load(folder / "gru_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    control = json.loads((folder / "score.json").read_text())["weighted_pearson"]
    torch.manual_seed(seed)
    model = make_gru().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    history = [score_groups(model, parquet, validation_groups, mean, std, device)]
    print("uniqueness fold", fold, "power", power, "epoch 0", history[-1], flush=True)
    for epoch in range(epochs):
        order = rng.permutation(fit_groups)
        losses = []
        for offset in range(0, len(order), BATCH):
            raw, target_np = read_batch(parquet, order[offset:offset + BATCH])
            values = torch.from_numpy(normalize(raw, mean, std)).to(device)
            target_np = np.clip(target_np, -2, 2).astype(np.float32)
            target = torch.from_numpy(target_np).to(device)
            weights = torch.from_numpy(run_weights(target_np, power)).to(device)
            hidden = None
            for start in range(0, N_ROWS, CHUNK):
                sl = slice(start, start + CHUNK)
                optimizer.zero_grad()
                prediction, hidden = forward(model, values[:, sl], hidden)
                begin = WARMUP if start == 0 else 0
                weight = weights[:, sl, None] * target[:, sl].abs()
                loss = ((prediction[:, begin:] - target[:, sl][:, begin:]).square()
                        * weight[:, begin:]).sum() / weight[:, begin:].sum().clamp_min(1e-6)
                loss.backward(); optimizer.step()
                hidden = detach_state(hidden)
                losses.append(float(loss.detach()))
        history.append(score_groups(model, parquet, validation_groups, mean, std, device))
        print("uniqueness fold", fold, "power", power, "epoch", epoch + 1,
              "loss", round(float(np.mean(losses)), 6), history[-1], flush=True)
    fixed = history[-1]["weighted_pearson"]
    result = {"power": power, "fold": fold, "seed": seed, "epochs": epochs,
              "control": control, "history": history, "fixed": fixed,
              "fixed_gain": fixed - control, "passes_gate": fixed >= control + 0.002,
              "interpretation": "inverse target persistence-run proxy; exact label spans unavailable"}
    out = OUT / f"fold_{fold}"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"power{power:g}_seed{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fold", type=int, choices=(1, 2, 3),
                        help="run sequence-disjoint 1,024/512 group-CV instead of the pilot")
    args = parser.parse_args()
    torch.set_num_threads(3)
    for power in args.power:
        if args.fold is None:
            run(power, args.epochs, args.seed, args.device)
        else:
            run_fold(power, args.fold, args.epochs, args.seed, args.device)


if __name__ == "__main__":
    main()
