"""Locked pilot: initialize a GRU hidden state from the complete warm-up."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from pilot import CACHE, DEV_GROUPS, FIT_GROUPS, N_FEATURES, N_ROWS, SEED, WARMUP
from warmup_context_screen import BATCH, CHUNK, CONTROL, normalize, train


OUT = CACHE / "warmup_initial"


class WarmupInitialGRU(torch.nn.Module):
    def __init__(self, width=96):
        super().__init__()
        self.context_net = torch.nn.Sequential(
            torch.nn.Linear(3 * N_FEATURES, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, width), torch.nn.Tanh(),
        )
        self.rnn = torch.nn.GRU(N_FEATURES, width, batch_first=True)
        self.head = torch.nn.Linear(width, 2)

    def context(self, values):
        warmup = values[:, :WARMUP]
        summary = torch.cat((warmup.mean(1), warmup.std(1, unbiased=False),
                             (warmup[:, 1:] - warmup[:, :-1]).abs().mean(1)), -1)
        return self.context_net(summary)

    def forward(self, values, context, hidden=None):
        if hidden is None:
            hidden = context[None]
        states, hidden = self.rnn(values, hidden)
        return self.head(states), hidden


def selfcheck():
    torch.manual_seed(1)
    model = WarmupInitialGRU(8).eval()
    values = torch.randn(2, 151, N_FEATURES)
    changed = values.clone(); changed[:, WARMUP:] += 20
    context = model.context(values)
    assert torch.equal(context, model.context(changed))
    full = model(values, context)[0]
    first, hidden = model(values[:, :73], context)
    second = model(values[:, 73:], context, hidden)[0]
    assert torch.allclose(full, torch.cat((first, second), 1), atol=1e-6)
    print("warm-up initial-state causality and chunk continuity checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    torch.set_num_threads(3)
    x = np.load(CACHE / "fit_x.npy", mmap_mode="r")
    y = np.load(CACHE / "fit_y.npy", mmap_mode="r")
    dx = np.load(CACHE / "dev_x.npy", mmap_mode="r")
    dy = np.load(CACHE / "dev_y.npy", mmap_mode="r")
    if x.shape != (FIT_GROUPS, N_ROWS, N_FEATURES) or dx.shape != (DEV_GROUPS, N_ROWS, N_FEATURES):
        raise ValueError("unexpected locked pilot cache")
    sampled = np.asarray(x[:, WARMUP::32], np.float64)
    mean, std = sampled.mean((0, 1)).astype(np.float32), sampled.std((0, 1)).astype(np.float32)
    std[std < 1e-6] = 1
    torch.manual_seed(args.seed)
    model = WarmupInitialGRU().to(args.device)
    history = train(model, x, y, dx, dy, mean, std, args.device, args.epochs, args.seed)
    fixed = history[-1]["weighted_pearson"]
    result = {"args": vars(args), "control": CONTROL, "history": history,
              "fixed": fixed, "fixed_gain": fixed - CONTROL,
              "passes_gate": fixed >= CONTROL + 0.005,
              "causality": "initial hidden state uses rows 0..98 only"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"seed{args.seed}_e{args.epochs}.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(model.cpu().state_dict(), OUT / f"seed{args.seed}_e{args.epochs}.pt")
    print(json.dumps({key: result[key] for key in ("fixed", "fixed_gain", "passes_gate")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
