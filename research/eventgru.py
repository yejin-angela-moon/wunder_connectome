"""Event-time GRU: steps only when the i0 price block (columns 0-21) changes.

Targets are constant between i0 price events, so one prediction per interval
suffices. A GRU stepped only at events remembers several times more price
history than a row-level GRU of the same size, and costs a fraction of it at
inference. Its input at event k is the standardized event row, the
standardized row just before it (end of the previous interval), log1p of the
previous interval's length, and log1p(k) / 10. The output at event k is the
prediction for every row of interval k. Loss weight is |y| times the interval's
row count (all required rows on train; scored + u · unscored on validation),
which equals the row-level loss for interval-constant predictions.

    python research/eventgru.py base --width 128          # train on FIT
    python research/eventgru.py ft --half 0               # mask-weighted FT on one validation half
    python research/eventgru.py pred --half 0 --swa 2 3   # held-out scored-row predictions (maskft cache format)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from global_wp import GlobalWP
from oos_global import read_group as read_valid_group
from pilot import CACHE, N_ROWS, ROOT, WARMUP
from scale import FIT, read_group as read_train_group
from stack import FULL

OUT = CACHE / "eventgru"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_INPUT = 226
CHUNK = 512
BATCH = 64
with np.load(FULL / "gru96_norm5ep/gru_standardizer.npz") as _s:
    MEAN, STD = _s["mean"], _s["std"]


class EventGRU(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.rnn = torch.nn.GRU(N_INPUT, width, batch_first=True)
        self.head = torch.nn.Linear(width, 2)

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        return self.head(out), hidden


def events(raw, y, row_weight):
    """One sequence -> event inputs (K, 226), interval targets (K, 2), interval weights (K,), starts (K,)."""
    change = np.r_[True, np.any(raw[1:, :22] != raw[:-1, :22], axis=1)]
    starts = np.flatnonzero(change)
    xs = np.clip((raw - MEAN) / STD, -10, 10).astype(np.float32)
    prev = np.zeros_like(xs[starts])
    prev[1:] = xs[starts[1:] - 1]
    gap = np.r_[0, np.diff(starts)]
    extra = np.stack([np.log1p(gap), np.log1p(np.arange(len(starts))) / 10], axis=1).astype(np.float32)
    inputs = np.concatenate([xs[starts], prev, extra], axis=1)
    segment = np.cumsum(change) - 1
    weight = np.bincount(segment, weights=row_weight, minlength=len(starts)).astype(np.float32)
    return inputs, np.clip(y[starts], -2, 2).astype(np.float32), weight, starts


def required_weight(mask=None, unscored=1.0):
    w = np.ones(N_ROWS) if mask is None else np.where(mask, 1.0, unscored)
    w[:WARMUP] = 0
    return w


def train_epochs(model, loader, epochs, lr, decay_at=None, tag="", save=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        model.train()
        if decay_at is not None and epoch == decay_at:
            for group in optimizer.param_groups:
                group["lr"] = lr * 0.3
        started, losses = time.time(), []
        for inputs, targets, weights in loader(epoch):
            x = torch.from_numpy(inputs).to(DEVICE)
            y = torch.from_numpy(targets).to(DEVICE)
            w = torch.from_numpy(weights).to(DEVICE)[..., None] * y.abs()
            hidden = None
            for start in range(0, x.shape[1], CHUNK):
                sl = slice(start, start + CHUNK)
                optimizer.zero_grad()
                prediction, hidden = model(x[:, sl], hidden)
                total = w[:, sl].sum()
                if total > 0:
                    loss = ((prediction - y[:, sl]).square() * w[:, sl]).sum() / total
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                hidden = hidden.detach()
        print(tag, "epoch", epoch + 1, "loss", round(float(np.mean(losses)), 6),
              f"{time.time() - started:.0f}s", flush=True)
        if save:
            save(epoch + 1)


def pad_batch(items):
    """List of (inputs, targets, weights) -> padded arrays; padding has zero weight."""
    length = max(len(i[0]) for i in items)
    x = np.zeros((len(items), length, N_INPUT), np.float32)
    y = np.zeros((len(items), length, 2), np.float32)
    w = np.zeros((len(items), length), np.float32)
    for j, (a, b, c) in enumerate(items):
        x[j, :len(a)], y[j, :len(a)], w[j, :len(a)] = a, b, c
    return x, y, w


def base(args):
    parquet = pq.ParquetFile(ROOT / "datasets/train.parquet")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    def loader(epoch):
        order = rng.permutation(FIT)
        for offset in range(0, len(order), BATCH * 8):                  # length-bucketed batches
            block = []
            for g in order[offset:offset + BATCH * 8]:
                raw, y = read_train_group(parquet, int(g))
                block.append(events(raw, y, required_weight())[:3])
            block.sort(key=lambda item: len(item[0]))
            for b in range(0, len(block), BATCH):
                yield pad_batch(block[b:b + BATCH])

    model = EventGRU(args.width).to(DEVICE)
    tag = f"event{args.width}_s{args.seed}"
    train_epochs(model, loader, args.epochs, 1e-3, decay_at=3, tag=tag,
                 save=lambda e: torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                                           OUT / f"{tag}_e{e}.pt"))
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, OUT / f"{tag}.pt")


def valid_items(parquet, groups, unscored):
    items = []
    for g in groups:
        _, _, raw, y, mask = read_valid_group(parquet, int(g))
        items.append(events(raw, y, required_weight(mask, unscored))[:3])
    return items


def load_model(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = EventGRU(state["rnn.weight_hh_l0"].shape[1])
    model.load_state_dict(state)
    return model


def row_predictions(model, parquet, groups):
    """Held-out scored-row predictions (rows in group order), matching maskft's cache layout."""
    model = model.to(DEVICE).eval()
    ps, ys = [], []
    with torch.no_grad():
        for offset in range(0, len(groups), BATCH):
            batch = [read_valid_group(parquet, int(g)) for g in groups[offset:offset + BATCH]]
            evs = [events(raw, y, required_weight(mask)) for _, _, raw, y, mask in batch]
            x, _, _ = pad_batch([(e[0], e[1], e[2]) for e in evs])
            out = model(torch.from_numpy(x).to(DEVICE))[0].cpu().numpy()
            for (_, _, raw, y, mask), e, o in zip(batch, evs, out):
                change = np.r_[True, np.any(raw[1:, :22] != raw[:-1, :22], axis=1)]
                segment = np.cumsum(change) - 1
                ps.append(o[segment][mask])
                ys.append(y[mask])
    return np.concatenate(ps), np.concatenate(ys)


def score(p, y):
    acc = GlobalWP()
    acc.add(y, p, np.ones(len(y), bool))
    return acc.result()["weighted_pearson"]


def ft(args):
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    groups = np.arange(1_873)
    fit_groups, eval_groups = groups[groups % 2 == args.half], groups[groups % 2 != args.half]
    items = valid_items(parquet, fit_groups, args.unscored)
    model = load_model(OUT / f"{args.base}.pt").to(DEVICE)
    print("frozen held-out scored WP", round(score(*row_predictions(model, parquet, eval_groups)), 6), flush=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    def loader(epoch):
        order = rng.permutation(len(items))
        for b in range(0, len(order), 32):
            yield pad_batch([items[i] for i in order[b:b + 32]])

    tag = f"{args.base}_ft_h{args.half}"

    def save(epoch):
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, OUT / f"{tag}_e{epoch}.pt")
        print(tag, "epoch", epoch, "held-out scored WP",
              round(score(*row_predictions(model, parquet, eval_groups)), 6), flush=True)

    train_epochs(model, loader, args.epochs, args.lr, tag=tag, save=save)


def pred(args):
    """Write SWA held-out predictions where maskft.combo looks for cached member predictions."""
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    groups = np.arange(1_873)
    eval_groups = groups[groups % 2 != args.half]
    states = [torch.load(OUT / f"{args.base}_ft_h{args.half}_e{e}.pt", weights_only=True) for e in args.swa]
    model = EventGRU(states[0]["rnn.weight_hh_l0"].shape[1])
    model.load_state_dict({k: sum(s[k] for s in states) / len(states) for k in states[0]})
    p, y = row_predictions(model, parquet, eval_groups)
    name = f"{args.base}_uEV_e{'-'.join(map(str, args.swa))}_h{args.half}"
    np.savez(CACHE / "maskft" / f"pred_{name}.npz", p=p, y=y)
    print(name, "held-out scored WP", round(score(p, y), 6), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("base", "ft", "pred"))
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--base", default="event128_s91")
    parser.add_argument("--half", type=int, default=0, choices=(0, 1))
    parser.add_argument("--unscored", type=float, default=0.06)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--swa", type=int, nargs="+", default=[2, 3])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"base": base, "ft": ft, "pred": pred}[args.command](args)


if __name__ == "__main__":
    main()
