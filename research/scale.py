"""Scale all three pilot models using only the locked train fitting groups."""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from pilot import CACHE, ROOT, N_FEATURES, N_ROWS, SEED, WARMUP, check_train, gru_predict, make_gru, make_mlp

sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS, TARGET_COLUMNS, validate_sequence

FIT = np.arange(10_607)[np.arange(10_607) % 10 != 0]
DEV = np.arange(0, 10_607, 10)
SAMPLE_STEP = 32
MLP_EPOCHS = 2
GRU_EPOCHS = 3
GRU_BATCH = 64
GRU_CHUNK = 512


def read_group(parquet, group):
    table = parquet.read_row_group(int(group), columns=["seq_ix", "step_in_seq", "need_prediction",
                                                       *FEATURE_COLUMNS, *TARGET_COLUMNS], use_threads=False)
    validate_sequence(table)
    x = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    y = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"nonfinite input or target in train group {group}")
    return x, y


def weighted_loss(prediction, target):
    return ((prediction - target).square() * target.abs()).mean()


def lag_features(x):
    """Current features and one-step differences, resetting at each sequence."""
    change = np.empty_like(x)
    change[0] = 0
    change[1:] = x[1:] - x[:-1]
    return np.column_stack((x, change))


def history_features(x, lags, ema_span=None):
    """Current row and causal changes at specified lags; each call is one sequence."""
    columns = [x]
    for lag in lags:
        change = np.zeros_like(x)
        change[lag:] = x[lag:] - x[:-lag]
        columns.append(change)
    if ema_span is not None:
        from scipy.signal import lfilter
        alpha = 2 / (ema_span + 1)
        average, _ = lfilter([alpha], [1, alpha - 1], x, axis=0,
                             zi=((1 - alpha) * x[0])[None, :])
        columns.append(x - average.astype(np.float32))
    return np.column_stack(columns)


def train_ridge(parquet, fit_groups=FIT, cache=CACHE, lag_differences=False,
                lags=(), ema_span=None):
    lags = (1,) if lag_differences else lags
    width = N_FEATURES * (1 + len(lags) + (ema_span is not None))
    gram = np.zeros((2, width + 1, width + 1), np.float64)
    rhs = np.zeros((2, width + 1), np.float64)
    for n, group in enumerate(fit_groups, 1):
        x, y = read_group(parquet, group)
        if lags or ema_span is not None:
            x = history_features(x, lags, ema_span)
        design = np.column_stack((x[WARMUP::SAMPLE_STEP].astype(np.float64),
                                  np.ones(len(x[WARMUP::SAMPLE_STEP]))))
        targets = np.clip(y[WARMUP::SAMPLE_STEP].astype(np.float64), -2, 2)
        for k in range(2):
            weights = np.sqrt(np.abs(targets[:, k]) + 1e-4)
            weighted = design * weights[:, None]
            with np.errstate(all="ignore"):
                gram[k] += weighted.T @ weighted
                rhs[k] += weighted.T @ (targets[:, k] * weights)
        if n % 1_000 == 0:
            print("ridge fitted groups", n, flush=True)
    if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
        raise ValueError("nonfinite ridge sufficient statistics")
    alpha = 100.0 * len(fit_groups) / 128  # Same regularization per sampled row as the pilot.
    coefficients = np.empty((width + 1, 2), np.float32)
    for k in range(2):
        gram[k].flat[::width + 2] += alpha
        coefficients[:, k] = np.linalg.solve(gram[k], rhs[k])
    np.save(cache / "scaled_ridge.npy", coefficients)
    print("ridge done", {"groups": len(fit_groups), "alpha": alpha}, flush=True)


def train_mlp(parquet, fit_groups=FIT, cache=CACHE, sample_step=SAMPLE_STEP,
              standardize=False, lags=(), seed=SEED, epochs=MLP_EPOCHS):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    width = N_FEATURES * (1 + len(lags))
    model = make_mlp(width)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if standardize:
        total = np.zeros(width, np.float64)
        square = np.zeros(width, np.float64)
        count = 0
        for group in fit_groups:
            x, _ = read_group(parquet, group)
            if lags:
                x = history_features(x, lags)
            sampled = x[WARMUP::sample_step].astype(np.float64)
            total += sampled.sum(axis=0)
            square += (sampled * sampled).sum(axis=0)
            count += len(sampled)
        mean = (total / count).astype(np.float32)
        std = np.sqrt(np.maximum(square / count - (total / count)**2, 1e-6)).astype(np.float32)
        np.savez(cache / "mlp_standardizer.npz", mean=mean, std=std)
    for epoch in range(epochs):
        losses = []
        for n, group in enumerate(rng.permutation(fit_groups), 1):
            x, y = read_group(parquet, group)
            if lags:
                x = history_features(x, lags)
            sampled = x[WARMUP::sample_step]
            if standardize:
                sampled = np.clip((sampled - mean) / std, -10, 10)
            xb = torch.from_numpy(np.ascontiguousarray(sampled))
            yb = torch.from_numpy(np.clip(np.ascontiguousarray(y[WARMUP::sample_step]), -2, 2))
            optimizer.zero_grad()
            loss = weighted_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if n % 1_000 == 0:
                print("mlp epoch", epoch + 1, "fitted groups", n, flush=True)
        print("mlp epoch", epoch + 1, "mean loss", float(np.mean(losses)), flush=True)
        torch.save(model.state_dict(), cache / f"scaled_mlp_epoch{epoch + 1}.pt")
    torch.save(model.state_dict(), cache / "scaled_mlp.pt")


def train_gru(parquet, fit_groups=FIT, cache=CACHE, hidden_size=64,
              seed=SEED, standardize=False, epochs=GRU_EPOCHS,
              decay_after=None, corr_weight=0.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_gru(hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if standardize:
        total = np.zeros(N_FEATURES, np.float64)
        square = np.zeros(N_FEATURES, np.float64)
        count = 0
        for group in fit_groups:
            values, _ = read_group(parquet, group)
            sampled = values[WARMUP::SAMPLE_STEP].astype(np.float64)
            total += sampled.sum(axis=0)
            square += (sampled * sampled).sum(axis=0)
            count += len(sampled)
        mean = (total / count).astype(np.float32)
        std = np.sqrt(np.maximum(square / count - (total / count)**2, 1e-6)).astype(np.float32)
        np.savez(cache / "gru_standardizer.npz", mean=mean, std=std)
    for epoch in range(epochs):
        if decay_after is not None and epoch == decay_after:
            for group in optimizer.param_groups:
                group["lr"] = 0.0003
        losses = []
        order = rng.permutation(fit_groups)
        for offset in range(0, len(order), GRU_BATCH):
            indices = order[offset:offset + GRU_BATCH]
            x = np.empty((len(indices), N_ROWS, N_FEATURES), np.float32)
            y = np.empty((len(indices), N_ROWS, 2), np.float32)
            for j, group in enumerate(indices):
                x[j], y[j] = read_group(parquet, group)
            if standardize:
                x = np.clip((x - mean) / std, -10, 10)
            hidden = None
            for start in range(0, N_ROWS, GRU_CHUNK):
                xb = torch.from_numpy(np.ascontiguousarray(x[:, start:start + GRU_CHUNK]))
                yb = torch.from_numpy(np.clip(np.ascontiguousarray(y[:, start:start + GRU_CHUNK]), -2, 2))
                optimizer.zero_grad()
                prediction, hidden = model(xb, hidden)
                skip = WARMUP if start == 0 else 0
                loss = weighted_loss(prediction[:, skip:], yb[:, skip:])
                if corr_weight:
                    target = yb[:, skip:].reshape(-1, 2)
                    estimated = prediction[:, skip:].reshape(-1, 2)
                    weights = target.abs()
                    weights = weights / weights.sum(dim=0).clamp_min(1e-12)
                    target_centered = target - (weights * target).sum(dim=0)
                    estimated_centered = estimated - (weights * estimated).sum(dim=0)
                    covariance = (weights * target_centered * estimated_centered).sum(dim=0)
                    variance = ((weights * target_centered.square()).sum(dim=0)
                                * (weights * estimated_centered.square()).sum(dim=0))
                    correlation = covariance / variance.clamp_min(1e-12).sqrt()
                    loss = loss + corr_weight * (1 - correlation.mean())
                loss.backward()
                optimizer.step()
                hidden = hidden.detach()
                losses.append(loss.item())
            if (offset + len(indices)) % 1_024 < GRU_BATCH:
                print("gru epoch", epoch + 1, "fitted groups", offset + len(indices), flush=True)
        print("gru epoch", epoch + 1, "mean loss", float(np.mean(losses)), flush=True)
        torch.save(model.state_dict(), cache / f"scaled_gru_epoch{epoch + 1}.pt")
    torch.save(model.state_dict(), cache / "scaled_gru.pt")


def score_dev(parquet, dev_groups=DEV, cache=CACHE):
    from utils import BlockAccumulator                                 # older starter packs only
    ridge = np.load(cache / "scaled_ridge.npy")
    mlp = make_mlp()
    mlp.load_state_dict(torch.load(cache / "scaled_mlp.pt", map_location="cpu", weights_only=True))
    mlp.eval()
    gru = make_gru()
    gru.load_state_dict(torch.load(cache / "scaled_gru.pt", map_location="cpu", weights_only=True))
    gru.eval()
    accumulators = {name: BlockAccumulator() for name in ("ridge", "mlp", "gru")}
    mask = np.arange(N_ROWS) >= WARMUP
    with torch.no_grad():
        for n, group in enumerate(dev_groups, 1):
            x, y = read_group(parquet, group)
            with np.errstate(all="ignore"):
                predictions = {"ridge": x @ ridge[:-1] + ridge[-1],
                               "mlp": mlp(torch.from_numpy(x)).numpy(),
                               "gru": gru_predict(gru, x)}
            for name, values in predictions.items():
                full = np.full((N_ROWS, 2), np.nan, np.float32)
                full[WARMUP:] = values[WARMUP:]
                accumulators[name].add(y, full, mask)
            if n % 200 == 0:
                print("scored development groups", n, flush=True)
    results = {name: acc.result() for name, acc in accumulators.items()}
    print(json.dumps(results, indent=2), flush=True)
    (cache / "scaled_dev_scores.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def selfcheck(parquet):
    assert len(FIT) == 9_546 and len(DEV) == 1_061 and not set(FIT) & set(DEV)
    x, y = read_group(parquet, FIT[0])
    assert x.shape == (N_ROWS, N_FEATURES) and y.shape == (N_ROWS, 2)
    torch.manual_seed(SEED)
    model = make_gru()
    xb = torch.from_numpy(x[:GRU_CHUNK].reshape(1, GRU_CHUNK, N_FEATURES))
    yb = torch.from_numpy(np.clip(y[:GRU_CHUNK].reshape(1, GRU_CHUNK, 2), -2, 2))
    loss = weighted_loss(model(xb)[0][:, WARMUP:], yb[:, WARMUP:])
    loss.backward()
    assert np.isfinite(loss.item())
    print("scaled training smoke check passed")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"ridge", "mlp", "gru", "score", "selfcheck"}:
        raise SystemExit("usage: scale.py {ridge|mlp|gru|score|selfcheck}")
    torch.set_num_threads(1)
    parquet = pq.ParquetFile(ROOT / "datasets/train.parquet")
    check_train(parquet)
    started = time.monotonic()
    {"ridge": train_ridge, "mlp": train_mlp, "gru": train_gru,
     "score": score_dev, "selfcheck": selfcheck}[sys.argv[1]](parquet)
    print("seconds", round(time.monotonic() - started, 2))
