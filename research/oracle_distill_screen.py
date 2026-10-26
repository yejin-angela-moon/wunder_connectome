"""Locked pilot: distil a future-event oracle into a strictly causal GRU.

The linear teacher may inspect the next i0 price event only while producing
cross-fitted targets for FIT sequences.  The deployed student receives the
same current/past 112-feature stream as the control and is scored without the
teacher or any future input.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from event_pretrain import BATCH, CHUNK, normalize, score, weighted_loss
from horizon import PRICE, fit as fit_oracle
from maskft import detach_state
from pilot import CACHE, DEV_GROUPS, FIT_GROUPS, N_FEATURES, N_ROWS, SEED, WARMUP, make_gru

OUT = CACHE / "oracle_distill_screen"
ALPHA = 0.5
CONTROL = 0.3160643776683788


def teacher_prediction(values, fitted):
    """Future-event teacher output and validity; used during training only."""
    price = values[:, :22]
    starts = np.flatnonzero(np.r_[True, np.any(price[1:] != price[:-1], axis=1)])
    segment = np.searchsorted(starts, np.arange(len(values)), side="right") - 1
    valid = segment + 1 < len(starts)
    rows = np.flatnonzero(valid)
    future = starts[segment[valid] + 1]
    current = values[rows][:, PRICE]
    features = np.column_stack((current, values[future][:, PRICE] - current)).astype(np.float64)
    mean, std, coefficients = fitted
    design = np.column_stack((np.clip((features - mean) / std, -10, 10),
                              np.ones(len(features))))
    with np.errstate(all="ignore"):
        prediction = design @ coefficients
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite privileged teacher prediction")
    output = np.zeros((len(values), 2), np.float32)
    output[valid] = np.clip(prediction, -2, 2).astype(np.float32)
    return output, valid


def train_epoch(model, optimizer, x, y, mean, std, teachers, device, rng, alpha):
    losses = []
    order = rng.permutation(len(x))
    for offset in range(0, len(x), BATCH):
        indices = order[offset:offset + BATCH]
        raw = np.array(x[indices])
        values = normalize(raw, mean, std)
        targets = np.clip(np.array(y[indices]), -2, 2).astype(np.float32)
        privileged, valid = [], []
        for index, sequence in zip(indices, raw):
            prediction, keep = teacher_prediction(sequence, teachers[int(index) % 2])
            privileged.append(prediction)
            valid.append(keep)
        privileged = np.stack(privileged)
        valid = np.stack(valid)
        hidden = None
        for start in range(0, N_ROWS, CHUNK):
            stop = min(start + CHUNK, N_ROWS)
            xb = torch.from_numpy(np.ascontiguousarray(values[:, start:stop])).to(device)
            yb = torch.from_numpy(np.ascontiguousarray(targets[:, start:stop])).to(device)
            tb = torch.from_numpy(np.ascontiguousarray(privileged[:, start:stop])).to(device)
            vb = torch.from_numpy(np.ascontiguousarray(valid[:, start:stop])).to(device)
            optimizer.zero_grad()
            prediction, hidden = model(xb, hidden)
            begin = WARMUP if start == 0 else 0
            label = weighted_loss(prediction[:, begin:], yb[:, begin:])
            weight = yb[:, begin:].abs() * vb[:, begin:, None]
            mimic = (((prediction[:, begin:] - tb[:, begin:]).square() * weight).sum() /
                     weight.sum().clamp_min(1e-6))
            loss = (1 - alpha) * label + alpha * mimic
            loss.backward()
            optimizer.step()
            hidden = detach_state(hidden)
            losses.append(loss.item())
    return float(np.mean(losses))


def selfcheck():
    values = np.zeros((40, N_FEATURES), np.float32)
    values[10:, :22] = 1
    values[25:, :22] = 3
    width = 2 * len(PRICE)
    coefficients = np.zeros((width + 1, 2))
    coefficients[-1] = (1, -1)
    prediction, valid = teacher_prediction(
        values, (np.zeros(width), np.ones(width), coefficients))
    assert valid[:25].all() and not valid[25:].any()
    assert np.all(prediction[:25] == (1, -1)) and np.all(prediction[25:] == 0)
    print("future-event teacher indexing and final-interval masking checks passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--alphas", type=float, nargs="*",
                        help="per-epoch teacher weights; overrides --epochs")
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
    # teachers[parity] is fitted on the opposite parity and predicts this parity.
    teachers = [fit_oracle(x[indices % 2 != parity], y[indices % 2 != parity], 1, True, True)
                for parity in (0, 1)]
    torch.manual_seed(args.seed)
    model = make_gru(96).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(args.seed)
    alphas = args.alphas if args.alphas else [ALPHA] * args.epochs
    history = [score(model, dx, dy, mean, std, args.device)]
    for epoch, alpha in enumerate(alphas):
        loss = train_epoch(model, optimizer, x, y, mean, std, teachers, args.device, rng, alpha)
        history.append(score(model, dx, dy, mean, std, args.device))
        print("oracle_distill epoch", epoch + 1, "alpha", alpha, "loss", round(loss, 6),
              history[-1], flush=True)
    best = max(row["weighted_pearson"] for row in history[1:])
    result = {"seed": args.seed, "alphas": alphas, "control": CONTROL,
              "cross_fitted_teacher": True, "history": history, "best": best,
              "gain": best - CONTROL, "passes_gate": best >= CONTROL + 0.005}
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"seed{args.seed}_a" + "-".join(f"{alpha:g}" for alpha in alphas)
    (OUT / f"{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(model.cpu().state_dict(), OUT / f"{tag}.pt")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
