"""Locked pilot for longer truncated backpropagation through time.

The candidate uses batch 4 x 2,048 rows, matching the control's batch 16 x
512 rows per optimizer update and its 320 updates per epoch.  Only the
candidate is rerun because the control is the exactly reproducible GRU96
pilot whose epoch-3 score is frozen below.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from event_pretrain import normalize, score, weighted_loss
from maskft import detach_state
from pilot import CACHE, DEV_GROUPS, FIT_GROUPS, N_FEATURES, N_ROWS, SEED, WARMUP, make_gru

OUT = CACHE / "long_bptt_screen"
CONTROL = 0.3160643776683788
CHUNK = 2_048
BATCH = 4


def train(model, x, y, dx, dy, mean, std, device, epochs, seed):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    history = [score(model, dx, dy, mean, std, device)]
    print("long_bptt epoch 0", history[-1], flush=True)
    for epoch in range(epochs):
        losses = []
        order = rng.permutation(len(x))
        for offset in range(0, len(x), BATCH):
            indices = order[offset:offset + BATCH]
            values = normalize(np.array(x[indices]), mean, std)
            targets = np.clip(np.array(y[indices]), -2, 2).astype(np.float32)
            hidden = None
            for start in range(0, N_ROWS, CHUNK):
                stop = min(start + CHUNK, N_ROWS)
                xb = torch.from_numpy(np.ascontiguousarray(values[:, start:stop])).to(device)
                yb = torch.from_numpy(np.ascontiguousarray(targets[:, start:stop])).to(device)
                optimizer.zero_grad()
                prediction, hidden = model(xb, hidden)
                begin = WARMUP if start == 0 else 0
                loss = weighted_loss(prediction[:, begin:], yb[:, begin:])
                loss.backward()
                optimizer.step()
                hidden = detach_state(hidden)
                losses.append(loss.item())
        history.append(score(model, dx, dy, mean, std, device))
        print("long_bptt epoch", epoch + 1, "loss", round(float(np.mean(losses)), 6),
              history[-1], flush=True)
    return history


def selfcheck():
    assert BATCH * CHUNK == 16 * 512
    assert (FIT_GROUPS // BATCH) * ((N_ROWS + CHUNK - 1) // CHUNK) == 320
    torch.manual_seed(1)
    model = make_gru(8).eval()
    values = torch.randn(2, 83, N_FEATURES)
    full = model(values)[0]
    first, hidden = model(values[:, :47])
    second = model(values[:, 47:], hidden)[0]
    assert torch.allclose(full, torch.cat((first, second), 1), atol=1e-6)
    print("matched optimizer exposure and recurrent continuity checks passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
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
    model = make_gru(96).to(args.device)
    history = train(model, x, y, dx, dy, mean, std, args.device, args.epochs, args.seed)
    best = max(row["weighted_pearson"] for row in history[1:])
    result = {"seed": args.seed, "chunk": CHUNK, "batch": BATCH,
              "control": CONTROL, "history": history, "best": best,
              "gain": best - CONTROL, "passes_gate": best >= CONTROL + 0.005}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"seed{args.seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(model.cpu().state_dict(), OUT / f"seed{args.seed}.pt")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
