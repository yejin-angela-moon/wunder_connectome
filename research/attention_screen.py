"""Locked pilot for one CPU-sized stateful local causal attention model."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from event_pretrain import finetune
from pilot import CACHE, DEV_GROUPS, FIT_GROUPS, N_FEATURES, N_ROWS, SEED, WARMUP

OUT = CACHE / "attention_screen.json"


class LocalAttention(torch.nn.Module):
    def __init__(self, width=64, heads=4, window=64):
        super().__init__()
        self.width, self.heads, self.window = width, heads, window
        self.input = torch.nn.Linear(N_FEATURES, width)
        self.qkv = torch.nn.Linear(width, 3 * width)
        self.output = torch.nn.Linear(width, width)
        self.norm1 = torch.nn.LayerNorm(width)
        self.ff = torch.nn.Sequential(torch.nn.Linear(width, 2 * width), torch.nn.SiLU(),
                                      torch.nn.Linear(2 * width, width))
        self.norm2 = torch.nn.LayerNorm(width)
        self.head = torch.nn.Linear(width, 2)
        self.lag_bias = torch.nn.Parameter(torch.zeros(heads, window))

    def forward(self, values, hidden=None):
        current = self.input(values)
        previous = current[:, :0] if hidden is None else hidden
        combined = torch.cat((previous, current), 1)
        past = previous.shape[1]
        qkv = self.qkv(combined).reshape(*combined.shape[:2], 3, self.heads,
                                         self.width // self.heads)
        query = qkv[:, past:, 0].transpose(1, 2)
        key = qkv[:, :, 1].transpose(1, 2)
        value = qkv[:, :, 2].transpose(1, 2)
        qpos = torch.arange(past, combined.shape[1], device=values.device)[:, None]
        kpos = torch.arange(combined.shape[1], device=values.device)[None]
        lag = qpos - kpos
        valid = (lag >= 0) & (lag < self.window)
        index = lag.clamp(0, self.window - 1).long()
        bias = self.lag_bias[:, index.reshape(-1)].reshape(
            self.heads, *index.shape).masked_fill(~valid[None], float("-inf"))[None]
        attended = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=bias).transpose(1, 2).flatten(2)
        encoded = self.norm1(current + self.output(attended))
        encoded = self.norm2(encoded + self.ff(encoded))
        return self.head(encoded), combined[:, -(self.window - 1):]


def selfcheck(device="cpu"):
    torch.manual_seed(1)
    model = LocalAttention(16, 4, 8).to(device).eval()
    values = torch.randn(2, 11, N_FEATURES, device=device)
    full = model(values)[0]
    first, hidden = model(values[:, :6])
    second = model(values[:, 6:], hidden)[0]
    assert torch.allclose(full, torch.cat((first, second), 1), atol=2e-6)
    full.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print("local attention full/chunk outputs and gradients agree")


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
    history = finetune("attn64h4w64", LocalAttention().to(args.device), x, y, dx, dy,
                       mean, std, args.device, args.epochs)
    OUT.write_text(json.dumps({"device": args.device, "history": history}, indent=2) + "\n")
    print(json.dumps({"best": max(row["weighted_pearson"] for row in history[1:])}, indent=2))


if __name__ == "__main__":
    main()
