"""Locked pilot for a small causal multi-resolution window model."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from event_pretrain import finetune
from pilot import CACHE, DEV_GROUPS, FIT_GROUPS, N_FEATURES, N_ROWS, SEED, WARMUP

OUT = CACHE / "multiscale_screen.json"
CONTROL = 0.316064


class MultiScaleMixer(torch.nn.Module):
    """Mix current values with causal changes across three fixed horizons."""

    def __init__(self, width=96, windows=(4, 16, 64)):
        super().__init__()
        self.windows = windows
        self.context = max(windows) - 1
        self.input = torch.nn.Linear(N_FEATURES * (len(windows) + 1), width)
        self.block = torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
            torch.nn.SiLU(),
        )
        self.head = torch.nn.Linear(width, 2)

    @staticmethod
    def rolling_mean(values, start, window):
        prefix = torch.cat((values[:, :1] * 0, values.cumsum(1)), dim=1)
        stop = torch.arange(start + 1, values.shape[1] + 1, device=values.device)
        begin = (stop - window).clamp_min(0)
        count = (stop - begin).to(values.dtype)[None, :, None]
        return (prefix[:, stop] - prefix[:, begin]) / count

    def forward(self, values, state=None):
        history = values[:, :0] if state is None else state
        joined = torch.cat((history, values), dim=1)
        means = [self.rolling_mean(joined, history.shape[1], window)
                 for window in self.windows]
        features = [values, values - means[0]]
        features.extend(left - right for left, right in zip(means, means[1:]))
        encoded = self.input(torch.cat(features, dim=-1))
        return self.head(encoded + self.block(encoded)), joined[:, -self.context:]


def selfcheck(device="cpu"):
    torch.manual_seed(1)
    model = MultiScaleMixer(16, (2, 4, 8)).to(device).eval()
    values = torch.randn(2, 19, N_FEATURES, device=device)
    full = model(values)[0]
    first, state = model(values[:, :6])
    second = model(values[:, 6:], state)[0]
    assert torch.allclose(full, torch.cat((first, second), 1), atol=2e-6)
    full.sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    print("multiscale full/chunk outputs and gradients agree")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
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
    mean = sampled.mean((0, 1)).astype(np.float32)
    std = sampled.std((0, 1)).astype(np.float32)
    std[std < 1e-6] = 1

    torch.manual_seed(args.seed)
    history = finetune("multiscale96", MultiScaleMixer().to(args.device), x, y, dx, dy,
                       mean, std, args.device, args.epochs)
    best = max(row["weighted_pearson"] for row in history[1:])
    result = {"device": args.device, "seed": args.seed, "control": CONTROL, "history": history,
              "best": best, "gain": best - CONTROL, "passes_gate": best >= CONTROL + 0.005}
    output = OUT if args.seed == SEED else OUT.with_name(f"multiscale_screen_seed{args.seed}.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
