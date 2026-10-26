"""Distil the cached full-FIT three-GRU teacher into a wider GRU student on MPS.

Same recipe as distill.py (standardized input, |y|-weighted label + imitation
loss, 5 epochs, lr 1e-3 then 3e-4 from epoch 4, 512-row chunks with carried
state), but with a configurable width and run on the Apple GPU when present.

    python research/distill_mps.py --width 128 --alpha 0.5
    python research/distill_mps.py --width 192 --alpha 0.5 --event   # + features.py inputs
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from features import N_EVENT, event_features
from pilot import CACHE, N_ROWS, ROOT, WARMUP, make_gru
from scale import read_group
from stack import DISTILL, FULL

OUT = CACHE / "distill_mps"


class SplitGRU(torch.nn.Module):
    """advance.py's split encoder: one shared encoder for i0 and i1, their
    difference and product, a small encoder for a0-a7, then a GRU."""

    def __init__(self, width):
        super().__init__()
        self.instrument = torch.nn.Sequential(torch.nn.Linear(52, 32), torch.nn.Tanh())
        self.additional = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Tanh())
        self.rnn = torch.nn.GRU(136, width, batch_first=True)
        self.head = torch.nn.Linear(width, 2)

    def forward(self, values, hidden=None):
        i0, i1 = self.instrument(values[..., :52]), self.instrument(values[..., 52:104])
        encoded = torch.cat((i0, i1, i0 - i1, i0 * i1, self.additional(values[..., 104:112])), dim=-1)
        encoded, hidden = self.rnn(encoded, hidden)
        return self.head(encoded), hidden


def build(state):
    """Model matching a saved state dict: SplitGRU or plain GRU."""
    if "context_net.0.weight" in state:
        from warmup_model import WarmupGRU
        return WarmupGRU(state["rnn.weight_hh_l0"].shape[1],
                         state["context_net.2.weight"].shape[0])
    if "blocks.0.gate.weight" in state:                               # minGRU stack (mingru.py)
        from mingru import MinGRUStack
        return MinGRUStack(state["inp.weight"].shape[0], sum(1 for k in state if k.endswith(".gate.weight")))
    width = state["rnn.weight_hh_l0"].shape[1]
    if "instrument.0.weight" in state:
        return SplitGRU(width)
    layers = sum(1 for k in state if k.startswith("rnn.weight_hh_l"))
    return make_gru(width, state["rnn.weight_ih_l0"].shape[1], layers)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHUNK = 512
BATCH = 64
EPOCHS = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--event", action="store_true", help="append causal event-time features")
    parser.add_argument("--split", action="store_true", help="split i0/i1 encoder architecture")
    parser.add_argument("--resume", type=int, default=0,
                        help="continue after this saved epoch (Adam moments restart)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{'split' if args.split else 'gru'}{args.width}{'ev' if args.event else ''}_a{args.alpha:g}_s{args.seed}"
    fit = np.load(DISTILL / "groups.npy")
    teacher = np.load(DISTILL / "teacher.npy", mmap_mode="r")
    with np.load(FULL / "gru96_norm5ep/gru_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    parquet = pq.ParquetFile(ROOT / "datasets/train.parquet")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = (SplitGRU(args.width) if args.split
             else make_gru(args.width, 112 + N_EVENT if args.event else 112)).to(DEVICE)
    if args.resume:
        model.load_state_dict(torch.load(OUT / f"{tag}_e{args.resume}.pt", weights_only=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3 if args.resume < 3 else 3e-4)
    for epoch in range(EPOCHS):
        if epoch < args.resume:
            rng.permutation(len(fit))                 # keep the uninterrupted epoch order
            continue
        if epoch == 3:
            for group in optimizer.param_groups:
                group["lr"] = 3e-4
        started, losses = time.time(), []
        order = rng.permutation(len(fit))
        for offset in range(0, len(fit), BATCH):
            indices = np.sort(order[offset:offset + BATCH])
            x = np.empty((len(indices), N_ROWS, 112), np.float32)
            y = np.empty((len(indices), N_ROWS, 2), np.float32)
            for j, index in enumerate(indices):
                x[j], y[j] = read_group(parquet, fit[index])
            normalized = np.clip((x - mean) / std, -10, 10).astype(np.float32)
            if args.event:
                normalized = np.concatenate([normalized, np.stack([event_features(r) for r in x])], axis=2)
            x = torch.from_numpy(normalized).to(DEVICE)
            y = torch.from_numpy(np.clip(y, -2, 2)).to(DEVICE)
            t = torch.from_numpy(np.asarray(teacher[indices], np.float32)).to(DEVICE)
            hidden = None
            for start in range(0, N_ROWS, CHUNK):
                sl = slice(start + (WARMUP if start == 0 else 0), start + CHUNK)
                optimizer.zero_grad()
                prediction, hidden = model(x[:, start:start + CHUNK], hidden)
                prediction = prediction[:, sl.start - start:]
                weight = y[:, sl].abs()
                label = ((prediction - y[:, sl]).square() * weight).mean()
                mimic = ((prediction - t[:, sl]).square() * weight).mean()
                loss = (1 - args.alpha) * label + args.alpha * mimic
                loss.backward()
                optimizer.step()
                hidden = hidden.detach()
                losses.append(loss.item())
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, OUT / f"{tag}_e{epoch + 1}.pt")
        print(tag, "epoch", epoch + 1, "loss", round(float(np.mean(losses)), 6),
              f"{time.time() - started:.0f}s", flush=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, OUT / f"{tag}.pt")


if __name__ == "__main__":
    main()
