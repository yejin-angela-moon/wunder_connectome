"""Mask-aware fine-tuning on validation plus train sequences weighted by a learned mask probability.

train.parquet has no `is_scored` column. A small causal GRU ("mask net") learns
P(is_scored | past inputs) on the validation fitting half. Train rows then get
the expected mask weight `q + u * (1 - q)`; validation rows use the true mask
`1 if scored else u`. Both are multiplied by |y|, as in maskft.py.

`--half 0/1` fits on that validation parity and scores the other half's
official scored rows after each epoch. `--half -1` fits on all validation
sequences (final refit, no held-out score).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from global_wp import GlobalWP
from maskft import load
from oos_global import read_group as read_valid_group
from pilot import CACHE, N_ROWS, ROOT, WARMUP
from scale import read_group as read_train_group

OUT = CACHE / "maskmix"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHUNK = 512
BATCH = 64
N_TRAIN = 10_607
REQUIRED = np.arange(N_ROWS) >= WARMUP


class MaskNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.GRU(112, 64, batch_first=True)
        self.head = torch.nn.Linear(64, 1)

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        return self.head(out)[..., 0], hidden


def split(half):
    groups = np.arange(1_873)
    if half < 0:
        return groups, None
    return groups[groups % 2 == half], groups[groups % 2 != half]


def normalize(x, scaler):
    return np.clip((x - scaler[0]) / scaler[1], -10, 10).astype(np.float32)


def read_valid(parquet, groups):
    data = [read_valid_group(parquet, int(g)) for g in groups]
    return (np.stack([d[2] for d in data]), np.stack([d[3] for d in data]),
            np.stack([d[4] for d in data]))


def fit_mask_net(parquet, groups, scaler, path, epochs=3, seed=11):
    if path.exists():
        net = MaskNet()
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return net.to(DEVICE).eval()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = MaskNet().to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(epochs):
        losses = []
        order = rng.permutation(groups)
        for offset in range(0, len(order), BATCH):
            x, _, mask = read_valid(parquet, order[offset:offset + BATCH])
            x = torch.from_numpy(normalize(x, scaler)).to(DEVICE)
            target = torch.from_numpy(mask.astype(np.float32)).to(DEVICE)
            weight = torch.from_numpy(np.broadcast_to(REQUIRED, mask.shape).astype(np.float32)).to(DEVICE)
            hidden = None
            for start in range(0, N_ROWS, CHUNK):
                sl = slice(start, start + CHUNK)
                optimizer.zero_grad()
                logit, hidden = net(x[:, sl], hidden)
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                    logit, target[:, sl], reduction="none") * weight[:, sl]).sum() / weight[:, sl].sum().clamp_min(1)
                loss.backward()
                optimizer.step()
                hidden = hidden.detach()
                losses.append(loss.item())
        print("mask net epoch", epoch + 1, "bce", round(float(np.mean(losses)), 5), flush=True)
    torch.save(net.state_dict(), path)
    return net.eval()


def mask_probability(net, x_normalized):
    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy(x_normalized).to(DEVICE))[0]).cpu().numpy()


def mask_auc(net, parquet, groups, scaler, rng):
    scores, labels = [], []
    for offset in range(0, len(groups), BATCH):
        x, _, mask = read_valid(parquet, groups[offset:offset + BATCH])
        q = mask_probability(net, normalize(x, scaler))
        keep = np.broadcast_to(REQUIRED, mask.shape) & (rng.random(mask.shape) < 0.05)
        scores.append(q[keep])
        labels.append(mask[keep])
    s, m = np.concatenate(scores), np.concatenate(labels)
    rank = np.empty(len(s))
    rank[np.argsort(s)] = np.arange(len(s))
    n1 = m.sum()
    return float((rank[m].sum() - n1 * (n1 - 1) / 2) / (n1 * (len(m) - n1))), float(s.mean()), float(m.mean())


def evaluate(model, parquet, groups, scaler):
    acc = GlobalWP()
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(groups), BATCH):
            x, y, mask = read_valid(parquet, groups[offset:offset + BATCH])
            x = normalize(x, scaler) if scaler is not None else x
            prediction = model(torch.from_numpy(x).to(DEVICE))[0].cpu().numpy()
            for i in range(len(x)):
                acc.add(y[i], prediction[i], mask[i])
    model.train()
    return acc.result()["weighted_pearson"]


def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="v1")
    parser.add_argument("--half", type=int, default=0, choices=(-1, 0, 1))
    parser.add_argument("--unscored", type=float, default=0.06)
    parser.add_argument("--train-per-epoch", type=int, default=2_808)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-pool", help=".npy of train row groups to sample from (default: all)")
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()
    DEVICE = args.device
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    valid = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    train = pq.ParquetFile(ROOT / "datasets/train.parquet")
    model, scaler = load(args.model)
    _, mask_scaler = load("v1")                     # the mask net always sees standardized input
    fit_groups, eval_groups = split(args.half)
    half_name = "all" if args.half < 0 else f"h{args.half}"
    net = fit_mask_net(valid, fit_groups, mask_scaler, OUT / f"masknet_{half_name}.pt")
    if eval_groups is not None:
        auc = mask_auc(net, valid, eval_groups, mask_scaler, np.random.default_rng(3))
        print("mask net held-out AUC", round(auc[0], 4), "mean q", round(auc[1], 4),
              "scored rate", round(auc[2], 4), flush=True)
    tag = (f"{args.model}_u{args.unscored:g}_lr{args.lr:g}_t{args.train_per_epoch}"
           f"{'_pool' if args.train_pool else ''}_s{args.seed}_{half_name}")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = model.to(DEVICE).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    if eval_groups is not None:
        history.append(evaluate(model, valid, eval_groups, scaler))
        print(tag, "epoch 0 held-out scored WP", round(history[-1], 6), flush=True)
    for epoch in range(args.epochs):
        started, losses = time.time(), []
        items = [("v", g) for g in fit_groups]
        pool = np.load(args.train_pool) if args.train_pool else np.arange(N_TRAIN)
        items += [("t", g) for g in rng.choice(pool, min(args.train_per_epoch, len(pool)), replace=False)]
        items = [items[i] for i in rng.permutation(len(items))]
        for offset in range(0, len(items), BATCH):
            batch = items[offset:offset + BATCH]
            xs, ys, ws = [], [], []
            for source, group in batch:
                if source == "v":
                    _, _, x, y, mask = read_valid_group(valid, int(group))
                    ws.append(np.where(mask, 1.0, args.unscored))
                else:
                    x, y = read_train_group(train, int(group))
                    ws.append(None)
                xs.append(x)
                ys.append(y)
            raw = np.stack(xs)
            normalized = normalize(raw, mask_scaler)
            if any(w is None for w in ws):
                q = mask_probability(net, normalized)
                ws = [w if w is not None else q[i] + args.unscored * (1 - q[i]) for i, w in enumerate(ws)]
            w = np.stack(ws).astype(np.float32)
            w[:, :WARMUP] = 0
            y = np.clip(np.stack(ys), -2, 2)
            x = torch.from_numpy(normalized if scaler is not None else raw).to(DEVICE)
            y = torch.from_numpy(y).to(DEVICE)
            w = torch.from_numpy(w).to(DEVICE)[..., None] * y.abs()
            hidden = None
            for start in range(0, N_ROWS, CHUNK):
                sl = slice(start, start + CHUNK)
                optimizer.zero_grad()
                prediction, hidden = model(x[:, sl], hidden)
                loss = ((prediction - y[:, sl]).square() * w[:, sl]).sum() / w[:, sl].sum().clamp_min(1e-6)
                loss.backward()
                optimizer.step()
                hidden = hidden.detach()
                losses.append(loss.item())
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, OUT / f"{tag}_e{epoch + 1}.pt")
        if eval_groups is not None:
            history.append(evaluate(model, valid, eval_groups, scaler))
        print(tag, "epoch", epoch + 1, "loss", round(float(np.mean(losses)), 6), "held-out scored WP",
              round(history[-1], 6) if eval_groups is not None else None,
              f"{time.time() - started:.0f}s", flush=True)
    (OUT / f"{tag}.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=2) + "\n")


if __name__ == "__main__":
    main()
