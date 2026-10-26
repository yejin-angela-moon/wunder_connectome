"""Dedicated GRU for the spread-like component S = -(t0 + t1).

Targets decompose into S = -(t0 + t1), which behaves like the current spread,
and D = t0 - t1, the future move. The joint models predict S poorly (corr
0.08 in the 0.657 tick group, 0.62 overall on validation scored rows), since
S carries little of the loss next to D. This model predicts S alone from
standardized plus event-time inputs, trained on FIT with weight |t0| + |t1|.
`blend` swaps the ensemble's S for a mix with this prediction and keeps its D.

    python research/sgru.py train --width 128
    python research/sgru.py blend --model sgru128_s101
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from features import N_EVENT, event_features
from global_wp import GlobalWP
from oos_global import read_group as read_valid_group
from pilot import CACHE, N_ROWS, ROOT, WARMUP, make_gru
from scale import FIT, read_group as read_train_group
from stack import FULL

OUT = CACHE / "sgru"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHUNK, BATCH = 512, 64
with np.load(FULL / "gru96_norm5ep/gru_standardizer.npz") as _s:
    MEAN, STD = _s["mean"], _s["std"]


def inputs(raw):
    x = np.clip((raw - MEAN) / STD, -10, 10).astype(np.float32)
    return np.concatenate([x, np.stack([event_features(r) for r in raw])], axis=-1)


def train(args):
    parquet = pq.ParquetFile(ROOT / "datasets/train.parquet")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = make_gru(args.width, 112 + N_EVENT)
    model.head = torch.nn.Linear(args.width, 1)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tag = f"sgru{args.width}_s{args.seed}"
    for epoch in range(args.epochs):
        if epoch == 3:
            for group in optimizer.param_groups:
                group["lr"] = 3e-4
        started, losses = time.time(), []
        order = rng.permutation(FIT)
        for offset in range(0, len(order), BATCH):
            data = [read_train_group(parquet, int(g)) for g in order[offset:offset + BATCH]]
            raw = np.stack([d[0] for d in data])
            y = np.clip(np.stack([d[1] for d in data]), -2, 2)
            x = torch.from_numpy(inputs(raw)).to(DEVICE)
            s = torch.from_numpy(-(y[..., 0] + y[..., 1])[..., None]).to(DEVICE)
            w = np.abs(y).sum(-1)
            w[:, :WARMUP] = 0
            w = torch.from_numpy(w[..., None].astype(np.float32)).to(DEVICE)
            hidden = None
            for start in range(0, N_ROWS, CHUNK):
                sl = slice(start, start + CHUNK)
                optimizer.zero_grad()
                prediction, hidden = model(x[:, sl], hidden)
                loss = ((prediction - s[:, sl]).square() * w[:, sl]).sum() / w[:, sl].sum().clamp_min(1e-6)
                loss.backward()
                optimizer.step()
                hidden = hidden.detach()
                losses.append(loss.item())
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, OUT / f"{tag}_e{epoch + 1}.pt")
        print(tag, "epoch", epoch + 1, "loss", round(float(np.mean(losses)), 6),
              f"{time.time() - started:.0f}s", flush=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, OUT / f"{tag}.pt")


def wp(y, p):
    acc = GlobalWP()
    acc.add(y.astype(np.float32), p.astype(np.float32), np.ones(len(y), bool))
    return acc.result()["weighted_pearson"]


def blend(args):
    """Both validation halves: replace the ensemble's S by a mix with the dedicated S model."""
    state = torch.load(OUT / f"{args.model}.pt", map_location="cpu", weights_only=True)
    model = make_gru(state["rnn.weight_hh_l0"].shape[1], 112 + N_EVENT)
    model.head = torch.nn.Linear(model.head.in_features, 1)
    model.load_state_dict(state)
    model = model.to(DEVICE).eval()
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    members = ["v1", "gru128_a0.5_s57", "gru192_a0.5_s57", "gru128_a0.75_s58", "gru192ev_a0.5_s59",
               "gru192_a0.75_s60", "gru128_a0.75_s61", "split128_a0.75_s81"]
    results = {}
    for half in (0, 1):
        groups = np.arange(1_873)
        groups = groups[groups % 2 != half]
        cached = [np.load(CACHE / "maskft" / f"pred_{m}_u0.06_lr0.0002_s7_e2-3_h{half}.npz") for m in members]
        p = np.mean([c["p"].astype(np.float64) for c in cached], 0)
        y = cached[0]["y"]
        s_hat, ticks = [], []
        with torch.no_grad():
            for offset in range(0, len(groups), 32):
                batch = [read_valid_group(parquet, int(g)) for g in groups[offset:offset + 32]]
                out = model(torch.from_numpy(inputs(np.stack([b[2] for b in batch]))).to(DEVICE))[0].cpu().numpy()
                for (_, _, _, yy, mask), o in zip(batch, out):
                    s_hat.append(o[mask, 0])
                    a = np.abs(yy[:, 0])
                    ticks.append(np.full(mask.sum(), round(float(a[a > 0].min()), 3)))
        s_hat, ticks = np.concatenate(s_hat), np.concatenate(ticks)
        np.savez(OUT / f"shat_{args.model}_h{half}.npz", s_hat=s_hat, ticks=ticks, p=p, y=y)
        yc = np.clip(y.astype(np.float64), -2, 2)
        S = -(yc[:, 0] + yc[:, 1])
        pS, pD = -(p[:, 0] + p[:, 1]), p[:, 0] - p[:, 1]
        print(f"half {half}: corr(S) ensemble {np.corrcoef(pS, S)[0, 1]:.3f}  dedicated {np.corrcoef(s_hat, S)[0, 1]:.3f}")
        for k in np.unique(ticks):
            m = ticks == k
            print(f"   tick {k}: corr(S) ensemble {np.corrcoef(pS[m], S[m])[0, 1]:.3f}  dedicated {np.corrcoef(s_hat[m], S[m])[0, 1]:.3f}")
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            s_mix = (1 - a) * pS + a * s_hat
            q = np.stack([(pD - s_mix) / 2, (-pD - s_mix) / 2], 1)
            results.setdefault(a, []).append(wp(y, q))
    for a, s in results.items():
        print(f"S from dedicated model weight {a:.2f}: h0 {s[0]:.6f} h1 {s[1]:.6f} mean {np.mean(s):.6f}"
              f"  gain {np.mean(s) - np.mean(results[0.0]):+.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "blend"))
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--model", default="sgru128_s101")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"train": train, "blend": blend}[args.command](args)


if __name__ == "__main__":
    main()
