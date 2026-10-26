"""Locked pilot for a GRU conditioned on the complete 99-row warm-up.

The callback is allowed to observe all warm-up rows before its first scored
prediction.  This model summarizes only those rows, then supplies the fixed
summary to every recurrent step.  An eventual callback can buffer and replay
the 99 warm-up rows once the context is known.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from event_pretrain import normalize, weighted_loss
from global_wp import GlobalWP
from group_cv import folds
from maskft import detach_state
from pilot import (CACHE, DEV_GROUPS, FIT_GROUPS, N_FEATURES, N_ROWS, ROOT, SEED, WARMUP,
                   check_train)
from scale import read_group


OUT = CACHE / "warmup_context"
CONTROL = 0.3160643776683788
BATCH = 16
CHUNK = 512
WIDTH = 96
CONTEXT = 16


class WarmupContextGRU(torch.nn.Module):
    def __init__(self, width=WIDTH, context_size=CONTEXT):
        super().__init__()
        self.context_net = torch.nn.Sequential(
            torch.nn.Linear(3 * N_FEATURES, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, context_size),
            torch.nn.Tanh(),
        )
        self.rnn = torch.nn.GRU(N_FEATURES + context_size, width, batch_first=True)
        self.head = torch.nn.Linear(width, 2)

    def context(self, values):
        warmup = values[:, :WARMUP]
        mean = warmup.mean(1)
        std = warmup.std(1, unbiased=False)
        movement = (warmup[:, 1:] - warmup[:, :-1]).abs().mean(1)
        return self.context_net(torch.cat((mean, std, movement), -1))

    def forward(self, values, context, hidden=None):
        expanded = context[:, None].expand(-1, values.shape[1], -1)
        states, hidden = self.rnn(torch.cat((values, expanded), -1), hidden)
        return self.head(states), hidden


def score(model, x, y, mean, std, device):
    metric = GlobalWP()
    required = np.arange(N_ROWS) >= WARMUP
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(x), BATCH):
            values = normalize(np.array(x[offset:offset + BATCH]), mean, std)
            tensor = torch.from_numpy(values).to(device)
            context = model.context(tensor)
            hidden, pieces = None, []
            for start in range(0, N_ROWS, CHUNK):
                prediction, hidden = model(tensor[:, start:start + CHUNK], context, hidden)
                pieces.append(prediction.cpu().numpy())
            prediction = np.concatenate(pieces, 1)
            for target, estimate in zip(y[offset:offset + BATCH], prediction):
                metric.add(target, estimate, required)
    model.train()
    return metric.result()


def train(model, x, y, dx, dy, mean, std, device, epochs, seed):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    history = [score(model, dx, dy, mean, std, device)]
    print("warmup_context epoch 0", history[-1], flush=True)
    for epoch in range(epochs):
        if epoch == 3:
            for group in optimizer.param_groups:
                group["lr"] = 3e-4
        order = rng.permutation(len(x))
        losses = []
        for offset in range(0, len(order), BATCH):
            indices = order[offset:offset + BATCH]
            values = normalize(np.array(x[indices]), mean, std)
            targets = np.clip(np.array(y[indices]), -2, 2).astype(np.float32)
            values = torch.from_numpy(values).to(device)
            targets = torch.from_numpy(targets).to(device)
            hidden = None
            for start in range(0, N_ROWS, CHUNK):
                sl = slice(start, start + CHUNK)
                optimizer.zero_grad()
                context = model.context(values)
                prediction, hidden = model(values[:, sl], context, hidden)
                begin = WARMUP if start == 0 else 0
                loss = weighted_loss(prediction[:, begin:], targets[:, sl][:, begin:])
                loss.backward()
                optimizer.step()
                hidden = detach_state(hidden)
                losses.append(float(loss.detach().cpu()))
        history.append(score(model, dx, dy, mean, std, device))
        print("warmup_context epoch", epoch + 1, "loss", round(float(np.mean(losses)), 6),
              history[-1], flush=True)
    return history


def read_train_batch(parquet, groups):
    data = [read_group(parquet, int(group)) for group in groups]
    return np.stack([row[0] for row in data]), np.stack([row[1] for row in data])


def score_groups(model, parquet, groups, mean, std, device):
    metric = GlobalWP()
    required = np.arange(N_ROWS) >= WARMUP
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(groups), BATCH):
            raw, targets = read_train_batch(parquet, groups[offset:offset + BATCH])
            values = torch.from_numpy(normalize(raw, mean, std)).to(device)
            context = model.context(values)
            hidden, pieces = None, []
            for start in range(0, N_ROWS, CHUNK):
                prediction, hidden = model(values[:, start:start + CHUNK], context, hidden)
                pieces.append(prediction.cpu().numpy())
            prediction = np.concatenate(pieces, 1)
            for target, estimate in zip(targets, prediction):
                metric.add(target, estimate, required)
            if offset and offset % 160 == 0:
                print("warmup_context scored groups", offset, flush=True)
    model.train()
    return metric.result()


def train_groups(model, parquet, fit_groups, validation_groups, mean, std, device, epochs, seed):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    history = [score_groups(model, parquet, validation_groups, mean, std, device)]
    print("warmup_context large epoch 0", history[-1], flush=True)
    for epoch in range(epochs):
        if epoch == 3:
            for group in optimizer.param_groups:
                group["lr"] = 3e-4
        order, losses = rng.permutation(fit_groups), []
        for offset in range(0, len(order), BATCH):
            raw, targets = read_train_batch(parquet, order[offset:offset + BATCH])
            values = torch.from_numpy(normalize(raw, mean, std)).to(device)
            targets = torch.from_numpy(np.clip(targets, -2, 2).astype(np.float32)).to(device)
            hidden = None
            for start in range(0, N_ROWS, CHUNK):
                sl = slice(start, start + CHUNK)
                optimizer.zero_grad()
                context = model.context(values)
                prediction, hidden = model(values[:, sl], context, hidden)
                begin = WARMUP if start == 0 else 0
                loss = weighted_loss(prediction[:, begin:], targets[:, sl][:, begin:])
                loss.backward()
                optimizer.step()
                hidden = detach_state(hidden)
                losses.append(float(loss.detach().cpu()))
            if offset and offset % 256 == 0:
                print("warmup_context epoch", epoch + 1, "trained groups", offset, flush=True)
        history.append(score_groups(model, parquet, validation_groups, mean, std, device))
        print("warmup_context large epoch", epoch + 1, "loss", round(float(np.mean(losses)), 6),
              history[-1], flush=True)
    return history


def selfcheck():
    torch.manual_seed(1)
    model = WarmupContextGRU(width=8, context_size=4).eval()
    values = torch.randn(2, 151, N_FEATURES)
    changed = values.clone()
    changed[:, WARMUP:] += 10
    context = model.context(values)
    assert torch.equal(context, model.context(changed))
    full = model(values, context)[0]
    first, hidden = model(values[:, :73], context)
    second = model(values[:, 73:], context, hidden)[0]
    assert torch.allclose(full, torch.cat((first, second), 1), atol=1e-6)
    full.sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    print("warm-up-only context, recurrent continuity, and gradients check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3),
                        help="run a 1,024-train/512-test group-CV confirmation")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    torch.set_num_threads(3)
    if args.fold is not None:
        parquet = pq.ParquetFile(ROOT / "datasets/train.parquet")
        check_train(parquet)
        number, fit_groups, validation_groups = next(
            plan for plan in folds() if plan[0] == args.fold)
        control_folder = CACHE / "improve" / f"fold_{number}" / "gru96_norm5ep"
        with np.load(control_folder / "gru_standardizer.npz") as source:
            mean, std = source["mean"], source["std"]
        control = json.loads((control_folder / "score.json").read_text())["weighted_pearson"]
        torch.manual_seed(args.seed)
        model = WarmupContextGRU().to(args.device)
        history = train_groups(model, parquet, fit_groups, validation_groups, mean, std,
                               args.device, args.epochs, args.seed)
        best = max(row["weighted_pearson"] for row in history[1:])
        fixed = history[-1]["weighted_pearson"]
        result = {"args": vars(args), "control": control, "history": history, "best": best,
                  "gain": best - control, "fixed": fixed, "fixed_gain": fixed - control,
                  "passes_gate": fixed >= control + 0.002,
                  "fit_groups": fit_groups.tolist(), "validation_groups": validation_groups.tolist(),
                  "causality": "context uses rows 0..98 only; first scored row is 99"}
        folder = OUT / f"fold_{number}"
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"seed{args.seed}_e{args.epochs}"
        (folder / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
        torch.save(model.cpu().state_dict(), folder / f"{stem}.pt")
        print(json.dumps({key: result[key] for key in
                          ("best", "gain", "fixed", "fixed_gain", "passes_gate")}, indent=2),
              flush=True)
        return

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

    torch.manual_seed(args.seed)
    model = WarmupContextGRU().to(args.device)
    history = train(model, x, y, dx, dy, mean, std, args.device, args.epochs, args.seed)
    best = max(row["weighted_pearson"] for row in history[1:])
    fixed = history[-1]["weighted_pearson"]
    result = {
        "args": vars(args),
        "control": CONTROL,
        "history": history,
        "best": best,
        "gain": best - CONTROL,
        "fixed": fixed,
        "fixed_gain": fixed - CONTROL,
        "passes_gate": fixed >= CONTROL + 0.005,
        "causality": "context uses rows 0..98 only; first scored row is 99",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"seed{args.seed}_e{args.epochs}"
    (OUT / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(model.cpu().state_dict(), OUT / f"{stem}.pt")
    print(json.dumps({key: result[key] for key in
                      ("best", "gain", "fixed", "fixed_gain", "passes_gate")}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
