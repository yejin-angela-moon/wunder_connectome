"""Locked pilot for a dedicated future-direction component D = t0 - t1."""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import torch

from event_pretrain import BATCH, CHUNK, finetune, normalize
from global_wp import GlobalWP
from pilot import CACHE, DEV_GROUPS, FIT_GROUPS, N_FEATURES, N_ROWS, SEED, WARMUP, make_gru

OUT = CACHE / "difference_screen.json"
WEIGHTS = (0.25, 0.5, 0.75, 1.0)


def train_epoch(model, optimizer, x, y, mean, std, device, rng):
    losses = []
    order = rng.permutation(len(x))
    for offset in range(0, len(x), BATCH):
        indices = order[offset:offset + BATCH]
        values = normalize(np.array(x[indices]), mean, std)
        targets = np.clip(np.array(y[indices]), -2, 2).astype(np.float32)
        difference = (targets[..., :1] - targets[..., 1:])
        weight = np.abs(targets).sum(-1, keepdims=True)
        hidden = None
        for start in range(0, N_ROWS, CHUNK):
            stop = min(start + CHUNK, N_ROWS)
            xb = torch.from_numpy(np.ascontiguousarray(values[:, start:stop])).to(device)
            db = torch.from_numpy(np.ascontiguousarray(difference[:, start:stop])).to(device)
            wb = torch.from_numpy(np.ascontiguousarray(weight[:, start:stop])).to(device)
            optimizer.zero_grad()
            prediction, hidden = model(xb, hidden)
            begin = WARMUP if start == 0 else 0
            loss = ((prediction[:, begin:] - db[:, begin:]).square()
                    * wb[:, begin:]).sum() / wb[:, begin:].sum().clamp_min(1e-6)
            loss.backward()
            optimizer.step()
            hidden = hidden.detach()
            losses.append(loss.item())
    return float(np.mean(losses))


def score(control, difference, x, y, mean, std, device):
    scores = {weight: GlobalWP() for weight in (0.0, *WEIGHTS)}
    control.eval(); difference.eval()
    required = np.arange(N_ROWS) >= WARMUP
    with torch.no_grad():
        for offset in range(0, len(x), BATCH):
            values = normalize(np.array(x[offset:offset + BATCH]), mean, std)
            base_hidden = difference_hidden = None
            predictions = {weight: [] for weight in scores}
            for start in range(0, N_ROWS, CHUNK):
                stop = min(start + CHUNK, N_ROWS)
                xb = torch.from_numpy(np.ascontiguousarray(values[:, start:stop])).to(device)
                base, base_hidden = control(xb, base_hidden)
                d_hat, difference_hidden = difference(xb, difference_hidden)
                spread = -(base[..., :1] + base[..., 1:])
                base_d = base[..., :1] - base[..., 1:]
                for weight in scores:
                    mixed_d = (1 - weight) * base_d + weight * d_hat
                    predictions[weight].append(torch.cat(
                        ((mixed_d - spread) / 2, (-mixed_d - spread) / 2), -1).cpu().numpy())
            for i in range(len(values)):
                target = y[offset + i]
                for weight, parts in predictions.items():
                    scores[weight].add(target, np.concatenate(parts, axis=1)[i], required)
    control.train(); difference.train()
    return {str(weight): metric.result() for weight, metric in scores.items()}


def selfcheck():
    prediction = torch.randn(5, 2)
    spread = -(prediction[:, :1] + prediction[:, 1:])
    difference = prediction[:, :1] - prediction[:, 1:]
    rebuilt = torch.cat(((difference - spread) / 2, (-difference - spread) / 2), -1)
    assert torch.allclose(prediction, rebuilt)
    print("S/D decomposition reconstructs t0/t1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
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
    initial = make_gru(96)
    control = copy.deepcopy(initial).to(args.device)
    finetune("control", control, x, y, dx, dy, mean, std, args.device, 3)
    difference = copy.deepcopy(initial)
    difference.head = torch.nn.Linear(96, 1)
    difference = difference.to(args.device)
    optimizer = torch.optim.Adam(difference.parameters(), lr=1e-3)
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(args.epochs):
        if epoch == 3:
            for group in optimizer.param_groups:
                group["lr"] = 3e-4
        loss = train_epoch(difference, optimizer, x, y, mean, std, args.device, rng)
        scores = score(control, difference, dx, dy, mean, std, args.device)
        history.append({"epoch": epoch + 1, "loss": loss, "scores": scores})
        print("difference epoch", epoch + 1, "loss", round(loss, 6),
              {weight: round(value["weighted_pearson"], 6)
               for weight, value in scores.items()}, flush=True)
    baseline = history[0]["scores"]["0.0"]["weighted_pearson"]
    best = max((value["weighted_pearson"], row["epoch"], float(weight))
               for row in history for weight, value in row["scores"].items()
               if float(weight) > 0)
    frozen = history[-1]["scores"]["0.5"]["weighted_pearson"]
    result = {"seed": args.seed, "history": history, "baseline": baseline, "best": best[0],
              "best_epoch": best[1], "best_weight": best[2], "gain": best[0] - baseline,
              "frozen_epoch5_weight05": frozen, "frozen_gain": frozen - baseline,
              "passes_gate": frozen >= baseline + 0.005}
    output = OUT if args.seed == SEED else OUT.with_name(f"difference_screen_seed{args.seed}.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
