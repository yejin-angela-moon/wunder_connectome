"""Tick-group specialists: continue mask-weighted fine-tuning on one tick group only.

Sequences fall into four target tick-size groups (0.125, 0.241, 0.469, 0.657),
identifiable from warm-up rows. This starts from a member's half-split
fine-tuned weights (SWA of the given epochs), continues on the fitting half's
sequences of one group, and scores the other half's scored rows of that group.

    python research/groupft.py --member v1_u0.06_lr0.0002_s7 --tick 0.657 --half 0
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pyarrow.parquet as pq
import torch

from global_wp import GlobalWP
from maskft import BATCH, CHUNK, OUT as FT_OUT, batch_arrays, load
from pilot import CACHE, N_ROWS, ROOT, WARMUP

OUT = CACHE / "groupft"
TICKS = (0.125, 0.241, 0.469, 0.657)


def tick_groups(parquet):
    """Rounded smallest nonzero |target| per validation sequence (the tick-size group)."""
    ticks = []
    for g in range(parquet.metadata.num_row_groups):
        y = parquet.read_row_group(g, columns=["t0", "t1"])
        a = np.abs(np.concatenate([y.column("t0").to_numpy(), y.column("t1").to_numpy()]))
        ticks.append(round(float(a[a > 1e-6].min()), 3))
    return np.array(ticks)


def score(model, parquet, groups, scaler):
    acc = GlobalWP()
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(groups), BATCH):
            x, y, mask = batch_arrays(parquet, groups[offset:offset + BATCH], scaler)
            prediction = model(torch.from_numpy(x))[0].numpy()
            for i in range(len(x)):
                acc.add(y[i], prediction[i], mask[i])
    model.train()
    return acc.result()["weighted_pearson"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", default="v1_u0.06_lr0.0002_s7")
    parser.add_argument("--swa", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--tick", type=float, default=0.657)
    parser.add_argument("--half", type=int, default=0, choices=(0, 1))
    parser.add_argument("--unscored", type=float, default=0.06)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.set_num_threads(6)
    OUT.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    ticks = tick_groups(parquet)
    groups = np.flatnonzero(np.isclose(ticks, args.tick))
    fit_groups, eval_groups = groups[groups % 2 == args.half], groups[groups % 2 != args.half]
    model, scaler = load(args.member.split("_u")[0])
    states = [torch.load(FT_OUT / f"{args.member}_h{args.half}_e{e}.pt", map_location="cpu",
                         weights_only=True) for e in args.swa]
    model.load_state_dict({k: sum(s[k] for s in states) / len(states) for k in states[0]})
    tag = f"{args.member}_t{args.tick:g}_lr{args.lr:g}_h{args.half}"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = [score(model, parquet, eval_groups, scaler)]
    print(tag, "fit", len(fit_groups), "eval", len(eval_groups), "epoch 0", round(history[-1], 6), flush=True)
    for epoch in range(args.epochs):
        order = rng.permutation(fit_groups)
        for offset in range(0, len(order), BATCH):
            x, y, mask = batch_arrays(parquet, order[offset:offset + BATCH], scaler)
            y = np.clip(y, -2, 2)
            weight = np.where(mask, 1.0, args.unscored).astype(np.float32)
            weight[:, :WARMUP] = 0
            hidden = None
            for start in range(0, N_ROWS, CHUNK):
                sl = slice(start, start + CHUNK)
                xb = torch.from_numpy(np.ascontiguousarray(x[:, sl]))
                yb = torch.from_numpy(np.ascontiguousarray(y[:, sl]))
                wb = torch.from_numpy(np.ascontiguousarray(weight[:, sl]))[..., None] * yb.abs()
                optimizer.zero_grad()
                prediction, hidden = model(xb, hidden)
                loss = ((prediction - yb).square() * wb).sum() / wb.sum().clamp_min(1e-6)
                loss.backward()
                optimizer.step()
                hidden = hidden.detach()
        torch.save(model.state_dict(), OUT / f"{tag}_e{epoch + 1}.pt")
        history.append(score(model, parquet, eval_groups, scaler))
        print(tag, "epoch", epoch + 1, round(history[-1], 6), flush=True)
    (OUT / f"{tag}.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=2) + "\n")


if __name__ == "__main__":
    main()
