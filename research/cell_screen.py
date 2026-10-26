"""Locked pilot screen for cheap recurrent families not covered by the GRU ensemble."""
from __future__ import annotations

import json

import numpy as np
import torch

from event_pretrain import finetune
from pilot import CACHE, DEV_GROUPS, FIT_GROUPS, N_FEATURES, N_ROWS, SEED, WARMUP

OUT = CACHE / "cell_screen.json"


class RecurrentHead(torch.nn.Module):
    def __init__(self, kind, width, layers=1):
        super().__init__()
        cell = torch.nn.RNN if kind == "rnn" else torch.nn.GRU
        self.rnn = cell(N_FEATURES, width, layers, batch_first=True)
        self.head = torch.nn.Linear(width, 2)

    def forward(self, values, hidden=None):
        values, hidden = self.rnn(values, hidden)
        return self.head(values), hidden


class DualGRU(torch.nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.cells = torch.nn.ModuleList([torch.nn.GRU(N_FEATURES, width, batch_first=True)
                                          for _ in range(2)])
        self.heads = torch.nn.ModuleList([torch.nn.Linear(width, 1) for _ in range(2)])

    def forward(self, values, hidden=None):
        hidden = (None, None) if hidden is None else hidden
        outputs, states = zip(*(cell(values, state) for cell, state in zip(self.cells, hidden)))
        return torch.cat([head(output) for head, output in zip(self.heads, outputs)], -1), states


def selfcheck():
    values = torch.randn(2, 11, N_FEATURES)
    for model in (RecurrentHead("rnn", 8), RecurrentHead("gru", 8, 2), DualGRU(8)):
        model.eval()
        full = model(values)[0]
        first, hidden = model(values[:, :6])
        second = model(values[:, 6:], hidden)[0]
        assert torch.allclose(full, torch.cat((first, second), 1), atol=1e-6)
    print("all recurrent candidates preserve state across chunks")


def main():
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
    del sampled
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    candidates = {"rnn128": lambda: RecurrentHead("rnn", 128),
                  "deep_gru64": lambda: RecurrentHead("gru", 64, 2),
                  "dual_gru64": DualGRU}
    result = {}
    for name, build in candidates.items():
        torch.manual_seed(SEED)
        model = build().to(device)
        result[name] = finetune(name, model, x, y, dx, dy, mean, std, device, 5)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: max(row["weighted_pearson"] for row in history[1:])
                      for name, history in result.items()}, indent=2))


if __name__ == "__main__":
    import sys
    selfcheck() if sys.argv[-1:] == ["selfcheck"] else main()
