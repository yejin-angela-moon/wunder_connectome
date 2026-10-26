"""Locked pilot for a cross-fitted delayed-target feature in a causal GRU."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from delayed_end_oracle import fit_reconstruction, reconstruct
from event_pretrain import BATCH, CHUNK, normalize, weighted_loss
from global_wp import GlobalWP
from maskft import detach_state
from pilot import CACHE, DEV_GROUPS, FIT_GROUPS, N_FEATURES, N_ROWS, SEED, WARMUP, make_gru

OUT = CACHE / "delayed_feature_screen"
CONTROL = 0.3160643776683788


def delayed_features(values, fitted):
    event, estimate = reconstruct(values, fitted)
    output = np.zeros((N_ROWS, 2), np.float32)
    for index, begin in enumerate(event[1:]):
        end = event[index + 2] if index + 2 < len(event) else N_ROWS
        output[begin:end] = estimate[index]
    return np.clip(output, -2, 2)


def make_candidate(base):
    candidate = make_gru(96, N_FEATURES + 2)
    with torch.no_grad():
        candidate.rnn.weight_ih_l0[:, :N_FEATURES].copy_(base.rnn.weight_ih_l0)
        candidate.rnn.weight_ih_l0[:, N_FEATURES:].zero_()
        candidate.rnn.weight_hh_l0.copy_(base.rnn.weight_hh_l0)
        candidate.rnn.bias_ih_l0.copy_(base.rnn.bias_ih_l0)
        candidate.rnn.bias_hh_l0.copy_(base.rnn.bias_hh_l0)
        candidate.head.load_state_dict(base.head.state_dict())
    return candidate


def train_epoch(model, optimizer, x, y, features, mean, std, device, rng):
    losses = []
    order = rng.permutation(len(x))
    for offset in range(0, len(x), BATCH):
        indices = order[offset:offset + BATCH]
        raw = normalize(np.array(x[indices]), mean, std)
        values = np.concatenate((raw, features[indices]), axis=2)
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
    return float(np.mean(losses))


def score(model, x, y, features, mean, std, device):
    metric = GlobalWP()
    required = np.arange(N_ROWS) >= WARMUP
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(x), BATCH):
            raw = normalize(np.array(x[offset:offset + BATCH]), mean, std)
            values = np.concatenate((raw, features[offset:offset + BATCH]), axis=2)
            hidden, pieces = None, []
            for start in range(0, N_ROWS, CHUNK):
                xb = torch.from_numpy(np.ascontiguousarray(values[:, start:start + CHUNK])).to(device)
                prediction, hidden = model(xb, hidden)
                pieces.append(prediction.cpu().numpy())
            prediction = np.concatenate(pieces, 1)
            for index in range(len(values)):
                metric.add(y[offset + index], prediction[index], required)
    model.train()
    return metric.result()


def selfcheck():
    torch.manual_seed(1)
    base = make_gru(96).eval()
    candidate = make_candidate(base).eval()
    x = torch.randn(2, 17, N_FEATURES)
    extra = torch.randn(2, 17, 2)
    assert torch.allclose(base(x)[0], candidate(torch.cat((x, extra), -1))[0], atol=1e-6)
    print("zero-initialized delayed inputs exactly preserve the base GRU")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=3)
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
    indices = np.arange(FIT_GROUPS)
    fitted = [fit_reconstruction(x[indices % 2 != parity], y[indices % 2 != parity])
              for parity in (0, 1)]
    train_features = np.stack([delayed_features(values, fitted[index % 2])
                               for index, values in enumerate(x)])
    full = fit_reconstruction(x, y)
    dev_features = np.stack([delayed_features(values, full) for values in dx])
    torch.manual_seed(args.seed)
    model = make_candidate(make_gru(96)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(args.seed)
    history = [score(model, dx, dy, dev_features, mean, std, args.device)]
    for epoch in range(args.epochs):
        loss = train_epoch(model, optimizer, x, y, train_features, mean, std, args.device, rng)
        history.append(score(model, dx, dy, dev_features, mean, std, args.device))
        print("delayed_feature epoch", epoch + 1, "loss", round(loss, 6), history[-1], flush=True)
    best = max(row["weighted_pearson"] for row in history[1:])
    result = {"seed": args.seed, "cross_fitted_train_feature": True, "control": CONTROL,
              "history": history, "best": best, "gain": best - CONTROL,
              "passes_gate": best >= CONTROL + 0.005}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"seed{args.seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(model.cpu().state_dict(), OUT / f"seed{args.seed}.pt")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
