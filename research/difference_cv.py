"""Larger group-CV gate for the frozen dedicated-D recipe."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pyarrow.parquet as pq
import torch

from global_wp import GlobalWP
from group_cv import folds
from pilot import CACHE, N_ROWS, ROOT, WARMUP, check_train, make_gru
from scale import GRU_BATCH, GRU_CHUNK, read_group

OUT = CACHE / "difference_cv"
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def train(parquet, fold, fit):
    folder = OUT / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    artifact = folder / "model.pt"
    if artifact.exists():
        return
    with np.load(CACHE / f"improve/fold_{fold}/gru96_norm5ep/gru_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    model = make_gru(96)
    model.head = torch.nn.Linear(96, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(5):
        if epoch == 3:
            for group in optimizer.param_groups:
                group["lr"] = 3e-4
        losses = []
        order = rng.permutation(fit)
        for offset in range(0, len(order), GRU_BATCH):
            indices = order[offset:offset + GRU_BATCH]
            data = [read_group(parquet, int(group)) for group in indices]
            x = np.clip((np.stack([row[0] for row in data]) - mean) / std, -10, 10)
            y = np.clip(np.stack([row[1] for row in data]), -2, 2)
            d = (y[..., :1] - y[..., 1:]).astype(np.float32)
            weight = np.abs(y).sum(-1, keepdims=True).astype(np.float32)
            hidden = None
            for start in range(0, N_ROWS, GRU_CHUNK):
                stop = min(start + GRU_CHUNK, N_ROWS)
                xb = torch.from_numpy(np.ascontiguousarray(x[:, start:stop]))
                db = torch.from_numpy(np.ascontiguousarray(d[:, start:stop]))
                wb = torch.from_numpy(np.ascontiguousarray(weight[:, start:stop]))
                optimizer.zero_grad()
                prediction, hidden = model(xb, hidden)
                begin = WARMUP if start == 0 else 0
                loss = ((prediction[:, begin:] - db[:, begin:]).square()
                        * wb[:, begin:]).sum() / wb[:, begin:].sum().clamp_min(1e-6)
                loss.backward(); optimizer.step(); hidden = hidden.detach()
                losses.append(loss.item())
        print("difference fold", fold, "epoch", epoch + 1,
              "loss", round(float(np.mean(losses)), 6), flush=True)
    torch.save(model.state_dict(), artifact)


def score(parquet, fold, validation):
    folder = OUT / f"fold_{fold}"
    with np.load(CACHE / f"improve/fold_{fold}/gru96_norm5ep/gru_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    base = make_gru(96)
    base.load_state_dict(torch.load(CACHE / f"improve/fold_{fold}/gru96_norm5ep/scaled_gru.pt",
                                    map_location="cpu", weights_only=True))
    difference = make_gru(96)
    difference.head = torch.nn.Linear(96, 1)
    difference.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
    base.eval(); difference.eval()
    metrics = {weight: GlobalWP() for weight in WEIGHTS}
    required = np.arange(N_ROWS) >= WARMUP
    with torch.no_grad():
        for n, group in enumerate(validation, 1):
            raw, y = read_group(parquet, int(group))
            x = np.clip((raw - mean) / std, -10, 10)
            base_hidden = difference_hidden = None
            predictions = {weight: [] for weight in WEIGHTS}
            for start in range(0, N_ROWS, GRU_CHUNK):
                xb = torch.from_numpy(np.ascontiguousarray(x[start:start + GRU_CHUNK])[None])
                p, base_hidden = base(xb, base_hidden)
                d_hat, difference_hidden = difference(xb, difference_hidden)
                spread, base_d = -(p[..., :1] + p[..., 1:]), p[..., :1] - p[..., 1:]
                for weight in WEIGHTS:
                    d = (1 - weight) * base_d + weight * d_hat
                    predictions[weight].append(torch.cat(
                        ((d - spread) / 2, (-d - spread) / 2), -1)[0].numpy())
            for weight, parts in predictions.items():
                metrics[weight].add(y, np.concatenate(parts), required)
            if n % 100 == 0:
                print("difference fold", fold, "scored", n, flush=True)
    result = {str(weight): metric.result() for weight, metric in metrics.items()}
    baseline = result["0.0"]["weighted_pearson"]
    frozen = result["0.5"]["weighted_pearson"]
    output = {"scores": result, "baseline": baseline, "frozen": frozen,
              "frozen_gain": frozen - baseline, "passes_gate": frozen >= baseline + 0.002}
    (folder / "score.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


def selfcheck():
    plans = list(folds())
    assert len(plans) == 3 and all(len(fit) == 1_024 and len(dev) == 512
                                   for _, fit, dev in plans)
    prediction = torch.randn(7, 2)
    spread = -(prediction[:, :1] + prediction[:, 1:])
    direction = prediction[:, :1] - prediction[:, 1:]
    rebuilt = torch.cat(((direction - spread) / 2, (-direction - spread) / 2), -1)
    assert torch.allclose(prediction, rebuilt)
    print("difference CV split and reconstruction checks passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "selfcheck"))
    parser.add_argument("--fold", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    if args.command == "selfcheck":
        selfcheck(); return
    torch.set_num_threads(1)
    parquet = pq.ParquetFile(ROOT / "datasets/train.parquet")
    check_train(parquet)
    OUT.mkdir(parents=True, exist_ok=True)
    fold, fit, validation = next(plan for plan in folds() if plan[0] == args.fold)
    train(parquet, fold, fit)
    score(parquet, fold, validation)


if __name__ == "__main__":
    main()
