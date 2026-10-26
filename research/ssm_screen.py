"""Locked pilot for one CPU-deployable selective state-space model."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch._higher_order_ops import scan

from event_pretrain import finetune
from pilot import CACHE, DEV_GROUPS, FIT_GROUPS, N_FEATURES, N_ROWS, SEED, WARMUP

OUT = CACHE / "ssm_screen.json"


class SelectiveSSM(torch.nn.Module):
    def __init__(self, width=96, state_size=8):
        super().__init__()
        self.width, self.state_size = width, state_size
        self.input = torch.nn.Linear(N_FEATURES, 3 * width + 2 * state_size)
        rates = torch.arange(1, state_size + 1).log().repeat(width, 1)
        self.log_rates = torch.nn.Parameter(rates)
        self.skip = torch.nn.Parameter(torch.ones(width))
        self.norm = torch.nn.LayerNorm(width)
        self.head = torch.nn.Linear(width, 2)

    @staticmethod
    def step(state, items):
        decay, drive, read = items
        state = decay * state + drive
        output = (state * read[:, None]).sum(-1)
        return state, output.clone()

    def forward(self, values, hidden=None):
        pieces = self.input(values).split(
            (self.width, self.width, self.width, self.state_size, self.state_size), -1)
        value, gate, delta, write, read = pieces
        decay = torch.exp(-torch.nn.functional.softplus(delta)[..., None]
                          * self.log_rates.exp()[None, None])
        drive = (1 - decay) * value[..., None] * write[..., None, :]
        if hidden is None:
            hidden = values.new_zeros(len(values), self.width, self.state_size)
        hidden, output = scan(self.step, hidden,
                              (decay.transpose(0, 1), drive.transpose(0, 1),
                               read.transpose(0, 1)))
        output = output.transpose(0, 1) * torch.nn.functional.silu(gate) + self.skip * value
        return self.head(self.norm(output)), hidden


def selfcheck(device="cpu"):
    torch.manual_seed(1)
    model = SelectiveSSM(8, 3).to(device).eval()
    values = torch.randn(2, 11, N_FEATURES, device=device)
    full = model(values)[0]
    first, hidden = model(values[:, :6])
    second = model(values[:, 6:], hidden)[0]
    assert torch.allclose(full, torch.cat((first, second), 1), atol=1e-6)
    full.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print("selective SSM full/chunk outputs and gradients agree")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck(args.device)
        return

    torch.set_num_threads(1)
    x = np.load(CACHE / "fit_x.npy", mmap_mode="r")
    y = np.load(CACHE / "fit_y.npy", mmap_mode="r")
    dx = np.load(CACHE / "dev_x.npy", mmap_mode="r")
    dy = np.load(CACHE / "dev_y.npy", mmap_mode="r")
    if x.shape != (FIT_GROUPS, N_ROWS, N_FEATURES) or dx.shape != (DEV_GROUPS, N_ROWS, N_FEATURES):
        raise ValueError("unexpected locked pilot cache")
    sampled = np.asarray(x[:, WARMUP::32], np.float64)
    mean, std = sampled.mean((0, 1)).astype(np.float32), sampled.std((0, 1)).astype(np.float32)
    std[std < 1e-6] = 1
    torch.manual_seed(SEED)
    model = SelectiveSSM().to(args.device)
    history = finetune("ssm96n8", model, x, y, dx, dy, mean, std, args.device, args.epochs)
    OUT.write_text(json.dumps({"device": args.device, "history": history}, indent=2) + "\n")
    print(json.dumps({"best": max(row["weighted_pearson"] for row in history[1:])}, indent=2))


if __name__ == "__main__":
    main()
