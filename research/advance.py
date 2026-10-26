"""Ordered FIT-only screens for the next model-improvement stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.signal import lfilter

from global_wp import GlobalWP
from group_cv import folds
from multiscale_screen import MultiScaleMixer
from pilot import CACHE, N_FEATURES, N_ROWS, ROOT, WARMUP, check_train, make_gru, make_mlp
from scale import GRU_BATCH, GRU_CHUNK, history_features, read_group, weighted_loss

OUT = CACHE / "advance"
LAGS = (1, 8, 32, 128)
OPTIONS = {"ms_lags": (), "ms_lags_ema": (8, 32, 128)}
GRAD_CLIP = {"ms_lags": 1.0, "ms_lags_ema": None}
SPECS = {
    "split_encoder": {"epochs": 5},
    "target_head": {"epochs": 5},
    "gru96_8ep": {"epochs": 8},
    "gru128_5ep": {"epochs": 5, "hidden": 128},
    "gru96_clip": {"epochs": 5, "clip": 1.0},
    "gru96_windows": {"epochs": 5, "windows": True},
    "tcn48": {"epochs": 5},
    "multiscale96": {"epochs": 3},
}
PLAIN = tuple(SPECS)
MLP = ("mlp_offset",)
ADAPTIVE = ("causal_center", "causal_norm")
REGIME = ("regime_head",)
BATCH = 16
EPOCHS = 5


class MultiScaleGRU(torch.nn.Module):
    def __init__(self, components, hidden_size=96):
        super().__init__()
        self.history_weight = torch.nn.Parameter(
            torch.full((N_FEATURES, components), 1 / components))
        self.rnn = torch.nn.GRU(2 * N_FEATURES, hidden_size, batch_first=True)
        self.head = torch.nn.Linear(hidden_size, 2)

    def forward(self, current, history, hidden=None):
        context = (history * self.history_weight).sum(dim=-1)
        values, hidden = self.rnn(torch.cat((current, context), dim=-1), hidden)
        return self.head(values), hidden


class SplitEncoderGRU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.instrument = torch.nn.Sequential(torch.nn.Linear(52, 32), torch.nn.Tanh())
        self.additional = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Tanh())
        self.rnn = torch.nn.GRU(136, 96, batch_first=True)
        self.head = torch.nn.Linear(96, 2)

    def forward(self, values, hidden=None):
        i0, i1 = self.instrument(values[..., :52]), self.instrument(values[..., 52:104])
        encoded = torch.cat((i0, i1, i0 - i1, i0 * i1,
                             self.additional(values[..., 104:])), dim=-1)
        encoded, hidden = self.rnn(encoded, hidden)
        return self.head(encoded), hidden


class TargetHeadGRU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.GRU(N_FEATURES, 96, batch_first=True)
        self.common = torch.nn.Sequential(torch.nn.Linear(96, 16), torch.nn.Tanh(),
                                          torch.nn.Linear(16, 1))
        self.difference = torch.nn.Sequential(torch.nn.Linear(96, 32), torch.nn.Tanh(),
                                              torch.nn.Linear(32, 1))

    def forward(self, values, hidden=None):
        values, hidden = self.rnn(values, hidden)
        common, difference = self.common(values), self.difference(values)
        return torch.cat((common + difference, common - difference), dim=-1), hidden


class CausalTCN(torch.nn.Module):
    def __init__(self, width=48):
        super().__init__()
        self.input = torch.nn.Linear(N_FEATURES, width)
        self.dilations = (1, 2, 4, 8, 16, 32, 64, 128)
        self.convolutions = torch.nn.ModuleList([
            torch.nn.Conv1d(width, width, 3, dilation=dilation)
            for dilation in self.dilations])
        self.head = torch.nn.Linear(width, 2)

    def forward(self, values, states=None):
        values = torch.tanh(self.input(values))
        if states is None:
            states = [None] * len(self.dilations)
        next_states = []
        for convolution, dilation, state in zip(self.convolutions, self.dilations, states):
            context = 2 * dilation
            prefix = (torch.zeros(values.shape[0], context, values.shape[2],
                                  dtype=values.dtype, device=values.device)
                      if state is None else state)
            joined = torch.cat((prefix, values), dim=1)
            update = torch.tanh(convolution(joined.transpose(1, 2)).transpose(1, 2))
            next_states.append(joined[:, -context:])
            values = (values + update) * (2 ** -0.5)
        return self.head(values), next_states


class RegimeGRU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.GRU(N_FEATURES, 96, batch_first=True)
        self.low = torch.nn.Linear(96, 2)
        self.high = torch.nn.Linear(96, 2)

    def forward(self, values, high, hidden=None):
        values, hidden = self.rnn(values, hidden)
        return torch.where(high[..., None], self.high(values), self.low(values)), hidden


def make_plain(name):
    if name == "split_encoder":
        return SplitEncoderGRU()
    if name == "target_head":
        return TargetHeadGRU()
    if name == "tcn48":
        return CausalTCN()
    if name == "multiscale96":
        return MultiScaleMixer()
    return make_gru(SPECS[name].get("hidden", 96))


def window_epoch(model, optimizer, x, y, rng, clip=None):
    losses = []
    batch = len(x)
    for _ in range(N_ROWS // GRU_CHUNK):
        starts = rng.integers(128, N_ROWS - GRU_CHUNK + 1, size=batch)
        burn = np.stack([x[j, start - 128:start] for j, start in enumerate(starts)])
        values = np.stack([x[j, start:start + GRU_CHUNK] for j, start in enumerate(starts)])
        targets = np.stack([y[j, start:start + GRU_CHUNK] for j, start in enumerate(starts)])
        with torch.no_grad():
            _, hidden = model(torch.from_numpy(np.ascontiguousarray(burn)))
        optimizer.zero_grad()
        prediction, _ = model(torch.from_numpy(np.ascontiguousarray(values)), hidden)
        loss = weighted_loss(prediction, torch.from_numpy(
            np.clip(np.ascontiguousarray(targets), -2, 2)))
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        losses.append(loss.item())
    return losses


def standardizer(parquet, groups):
    total = np.zeros(N_FEATURES, np.float64)
    square = np.zeros(N_FEATURES, np.float64)
    count = 0
    for group in groups:
        x, _ = read_group(parquet, group)
        sampled = x[WARMUP::32].astype(np.float64)
        total += sampled.sum(axis=0)
        square += (sampled * sampled).sum(axis=0)
        count += len(sampled)
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean**2, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


def history_chunk(x, start, end, ema_spans, ema_state):
    """Causal within-sequence history components for one contiguous chunk."""
    current = x[:, start:end]
    pieces = []
    for lag in LAGS:
        change = np.zeros_like(current)
        first = max(start, lag)
        if first < end:
            change[:, first - start:] = x[:, first:end] - x[:, first - lag:end - lag]
        pieces.append(change)
    for index, span in enumerate(ema_spans):
        alpha = np.float32(2 / (span + 1))
        average, ema_state[index] = lfilter([alpha], [1, alpha - 1], current, axis=1,
                                            zi=ema_state[index])
        pieces.append(current - average.astype(np.float32))
    return current, np.stack(pieces, axis=-1)


def adaptive_chunk(x, start, end, state, mode, span=128):
    alpha = np.float32(2 / (span + 1))
    current = x[:, start:end]
    mean, state[0] = lfilter([alpha], [1, alpha - 1], current, axis=1, zi=state[0])
    second, state[1] = lfilter([alpha], [1, alpha - 1], current * current,
                               axis=1, zi=state[1])
    centered = current - mean.astype(np.float32)
    if mode == "causal_center":
        return np.clip(centered, -10, 10)
    variance = np.maximum(second - mean * mean, 1e-3)
    return np.clip(centered / np.sqrt(variance).astype(np.float32), -10, 10)


def adaptive_state(batch, span=128):
    alpha = np.float32(2 / (span + 1))
    return [np.zeros((batch, 1, N_FEATURES), np.float32),
            np.full((batch, 1, N_FEATURES), 1 - alpha, np.float32)]


def volatility_chunk(x, start, end, state, span=128):
    alpha = np.float32(2 / (span + 1))
    values = x[:, start:end]
    mean, state[0] = lfilter([alpha], [1, alpha - 1], values, axis=1, zi=state[0])
    second, state[1] = lfilter([alpha], [1, alpha - 1], values * values,
                               axis=1, zi=state[1])
    return np.log1p(np.maximum(second - mean * mean, 1e-3)).mean(axis=2)


def causal_volatility(x, span=128):
    return volatility_chunk(x[None], 0, len(x), adaptive_state(1, span), span)[0]


def regime_threshold(parquet, fit, mean, std):
    values = []
    for group in fit[np.linspace(0, len(fit) - 1, 32, dtype=int)]:
        x, _ = read_group(parquet, group)
        x = np.clip((x - mean) / std, -10, 10)
        values.append(causal_volatility(x)[WARMUP::32])
    return float(np.median(np.concatenate(values)))


def train(parquet, fold, fit, name):
    ema_spans = OPTIONS[name]
    folder = OUT / f"fold_{fold}" / name
    folder.mkdir(parents=True, exist_ok=True)
    artifact = folder / "model.pt"
    if artifact.exists():
        print("using completed", artifact, flush=True)
        return
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    mean, std = standardizer(parquet, fit)
    np.savez(folder / "standardizer.npz", mean=mean, std=std)
    model = MultiScaleGRU(len(LAGS) + len(ema_spans))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(EPOCHS):
        if epoch == 3:
            for group in optimizer.param_groups:
                group["lr"] = 0.0003
        losses = []
        order = rng.permutation(fit)
        for offset in range(0, len(order), BATCH):
            indices = order[offset:offset + BATCH]
            x = np.empty((len(indices), N_ROWS, N_FEATURES), np.float32)
            y = np.empty((len(indices), N_ROWS, 2), np.float32)
            for j, group in enumerate(indices):
                x[j], y[j] = read_group(parquet, group)
            x = np.clip((x - mean) / std, -10, 10)
            ema_state = [((1 - 2 / (span + 1)) * x[:, :1]).copy() for span in ema_spans]
            hidden = None
            for start in range(0, N_ROWS, GRU_CHUNK):
                end = min(start + GRU_CHUNK, N_ROWS)
                current, history = history_chunk(x, start, end, ema_spans, ema_state)
                xb = torch.from_numpy(np.ascontiguousarray(current))
                hb = torch.from_numpy(np.ascontiguousarray(history))
                yb = torch.from_numpy(np.clip(np.ascontiguousarray(y[:, start:end]), -2, 2))
                optimizer.zero_grad()
                prediction, hidden = model(xb, hb, hidden)
                skip = WARMUP if start == 0 else 0
                loss = weighted_loss(prediction[:, skip:], yb[:, skip:])
                loss.backward()
                if GRAD_CLIP[name] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP[name])
                optimizer.step()
                hidden = (hidden.detach() if isinstance(hidden, torch.Tensor)
                          else [state.detach() for state in hidden])
                losses.append(loss.item())
            if (offset + len(indices)) % 256 < BATCH:
                print(name, "fold", fold, "epoch", epoch + 1,
                      "groups", offset + len(indices), flush=True)
        print(name, "fold", fold, "epoch", epoch + 1,
              "loss", float(np.mean(losses)), flush=True)
    torch.save(model.state_dict(), artifact)


def train_plain(parquet, fold, fit, name):
    folder = OUT / f"fold_{fold}" / name
    folder.mkdir(parents=True, exist_ok=True)
    artifact = folder / "model.pt"
    if artifact.exists():
        print("using completed", artifact, flush=True)
        return
    with np.load(CACHE / f"improve/fold_{fold}/gru96_norm5ep/gru_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    model = make_plain(name)
    spec = SPECS[name]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(spec["epochs"]):
        if epoch == 3:
            for group in optimizer.param_groups:
                group["lr"] = 0.0003
        if epoch == 5:
            for group in optimizer.param_groups:
                group["lr"] = 0.0001
        losses = []
        order = rng.permutation(fit)
        for offset in range(0, len(order), GRU_BATCH):
            indices = order[offset:offset + GRU_BATCH]
            x = np.empty((len(indices), N_ROWS, N_FEATURES), np.float32)
            y = np.empty((len(indices), N_ROWS, 2), np.float32)
            for j, group in enumerate(indices):
                x[j], y[j] = read_group(parquet, group)
            x = np.clip((x - mean) / std, -10, 10)
            if spec.get("windows"):
                losses.extend(window_epoch(model, optimizer, x, y, rng, spec.get("clip")))
                continue
            hidden = None
            for start in range(0, N_ROWS, GRU_CHUNK):
                xb = torch.from_numpy(np.ascontiguousarray(x[:, start:start + GRU_CHUNK]))
                yb = torch.from_numpy(np.clip(
                    np.ascontiguousarray(y[:, start:start + GRU_CHUNK]), -2, 2))
                optimizer.zero_grad()
                prediction, hidden = model(xb, hidden)
                skip = WARMUP if start == 0 else 0
                loss = weighted_loss(prediction[:, skip:], yb[:, skip:])
                loss.backward()
                if spec.get("clip") is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), spec["clip"])
                optimizer.step()
                hidden = (hidden.detach() if isinstance(hidden, torch.Tensor)
                          else [state.detach() for state in hidden])
                losses.append(loss.item())
            if (offset + len(indices)) % 1_024 < GRU_BATCH:
                print(name, "fold", fold, "epoch", epoch + 1,
                      "groups", offset + len(indices), flush=True)
        print(name, "fold", fold, "epoch", epoch + 1,
              "loss", float(np.mean(losses)), flush=True)
    torch.save(model.state_dict(), artifact)


def score(parquet, fold, validation, name):
    ema_spans = OPTIONS[name]
    folder = OUT / f"fold_{fold}" / name
    with np.load(folder / "standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    model = MultiScaleGRU(len(LAGS) + len(ema_spans))
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    metric = GlobalWP()
    required = np.arange(N_ROWS) >= WARMUP
    with torch.no_grad():
        for n, group in enumerate(validation, 1):
            x, y = read_group(parquet, group)
            x = np.clip((x - mean) / std, -10, 10)[None]
            ema_state = [((1 - 2 / (span + 1)) * x[:, :1]).copy() for span in ema_spans]
            hidden = None
            predictions = []
            for start in range(0, N_ROWS, GRU_CHUNK):
                end = min(start + GRU_CHUNK, N_ROWS)
                current, history = history_chunk(x, start, end, ema_spans, ema_state)
                values, hidden = model(torch.from_numpy(np.ascontiguousarray(current)),
                                       torch.from_numpy(np.ascontiguousarray(history)), hidden)
                predictions.append(values[0].numpy())
            values = np.concatenate(predictions)
            metric.add(y, values, required)
            if n % 100 == 0:
                print(name, "fold", fold, "scored", n, flush=True)
    result = metric.result()
    (folder / "score.json").write_text(json.dumps(result, indent=2) + "\n")
    print(name, "fold", fold, result["weighted_pearson"], flush=True)


def score_plain(parquet, fold, validation, name):
    folder = OUT / f"fold_{fold}" / name
    with np.load(CACHE / f"improve/fold_{fold}/gru96_norm5ep/gru_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    model = make_plain(name)
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    metric = GlobalWP()
    required = np.arange(N_ROWS) >= WARMUP
    with torch.no_grad():
        for n, group in enumerate(validation, 1):
            x, y = read_group(parquet, group)
            x = np.clip((x - mean) / std, -10, 10)
            hidden = None
            predictions = []
            for start in range(0, N_ROWS, GRU_CHUNK):
                values, hidden = model(torch.from_numpy(
                    np.ascontiguousarray(x[start:start + GRU_CHUNK])[None]), hidden)
                predictions.append(values[0].numpy())
            metric.add(y, np.concatenate(predictions), required)
            if n % 100 == 0:
                print(name, "fold", fold, "scored", n, flush=True)
    result = metric.result()
    (folder / "score.json").write_text(json.dumps(result, indent=2) + "\n")
    print(name, "fold", fold, result["weighted_pearson"], flush=True)


def train_mlp_offset(parquet, fold, fit):
    name = "mlp_offset"
    folder = OUT / f"fold_{fold}" / name
    folder.mkdir(parents=True, exist_ok=True)
    artifact = folder / "model.pt"
    if artifact.exists():
        print("using completed", artifact, flush=True)
        return
    with np.load(CACHE / f"improve/fold_{fold}/mlp_norm_lag32_4/mlp_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    model = make_mlp(4 * N_FEATURES)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(4):
        losses = []
        for n, group in enumerate(rng.permutation(fit), 1):
            x, y = read_group(parquet, group)
            x = history_features(x, (1, 8, 32))
            offset = int(rng.integers(32))
            rows = slice(WARMUP + offset, None, 32)
            xb = torch.from_numpy(np.ascontiguousarray(np.clip((x[rows] - mean) / std, -10, 10)))
            yb = torch.from_numpy(np.clip(np.ascontiguousarray(y[rows]), -2, 2))
            optimizer.zero_grad()
            loss = weighted_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if n % 1_000 == 0:
                print(name, "fold", fold, "epoch", epoch + 1, "groups", n, flush=True)
        print(name, "fold", fold, "epoch", epoch + 1,
              "loss", float(np.mean(losses)), flush=True)
    torch.save(model.state_dict(), artifact)


def score_mlp_offset(parquet, fold, validation):
    name = "mlp_offset"
    folder = OUT / f"fold_{fold}" / name
    with np.load(CACHE / f"improve/fold_{fold}/mlp_norm_lag32_4/mlp_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    model = make_mlp(4 * N_FEATURES)
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    metric = GlobalWP()
    required = np.arange(N_ROWS) >= WARMUP
    with torch.no_grad():
        for n, group in enumerate(validation, 1):
            x, y = read_group(parquet, group)
            x = history_features(x, (1, 8, 32))
            values = model(torch.from_numpy(np.ascontiguousarray(
                np.clip((x - mean) / std, -10, 10)))).numpy()
            metric.add(y, values, required)
            if n % 100 == 0:
                print(name, "fold", fold, "scored", n, flush=True)
    result = metric.result()
    (folder / "score.json").write_text(json.dumps(result, indent=2) + "\n")
    print(name, "fold", fold, result["weighted_pearson"], flush=True)


def train_adaptive(parquet, fold, fit, name):
    folder = OUT / f"fold_{fold}" / name
    folder.mkdir(parents=True, exist_ok=True)
    artifact = folder / "model.pt"
    if artifact.exists():
        print("using completed", artifact, flush=True)
        return
    with np.load(CACHE / f"improve/fold_{fold}/gru96_norm5ep/gru_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    model = make_gru(96)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5):
        if epoch == 3:
            for group in optimizer.param_groups:
                group["lr"] = 0.0003
        losses = []
        order = rng.permutation(fit)
        for offset in range(0, len(order), GRU_BATCH):
            indices = order[offset:offset + GRU_BATCH]
            x = np.empty((len(indices), N_ROWS, N_FEATURES), np.float32)
            y = np.empty((len(indices), N_ROWS, 2), np.float32)
            for j, group in enumerate(indices):
                x[j], y[j] = read_group(parquet, group)
            x = np.clip((x - mean) / std, -10, 10)
            state, hidden = adaptive_state(len(indices)), None
            for start in range(0, N_ROWS, GRU_CHUNK):
                end = min(start + GRU_CHUNK, N_ROWS)
                values = adaptive_chunk(x, start, end, state, name)
                xb = torch.from_numpy(np.ascontiguousarray(values))
                yb = torch.from_numpy(np.clip(np.ascontiguousarray(y[:, start:end]), -2, 2))
                optimizer.zero_grad()
                prediction, hidden = model(xb, hidden)
                skip = WARMUP if start == 0 else 0
                loss = weighted_loss(prediction[:, skip:], yb[:, skip:])
                loss.backward()
                optimizer.step()
                hidden = hidden.detach()
                losses.append(loss.item())
            if (offset + len(indices)) % 1_024 < GRU_BATCH:
                print(name, "fold", fold, "epoch", epoch + 1,
                      "groups", offset + len(indices), flush=True)
        print(name, "fold", fold, "epoch", epoch + 1,
              "loss", float(np.mean(losses)), flush=True)
    torch.save(model.state_dict(), artifact)


def score_adaptive(parquet, fold, validation, name):
    folder = OUT / f"fold_{fold}" / name
    with np.load(CACHE / f"improve/fold_{fold}/gru96_norm5ep/gru_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    model = make_gru(96)
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    metric = GlobalWP()
    required = np.arange(N_ROWS) >= WARMUP
    with torch.no_grad():
        for n, group in enumerate(validation, 1):
            x, y = read_group(parquet, group)
            x = np.clip((x - mean) / std, -10, 10)[None]
            state, hidden, predictions = adaptive_state(1), None, []
            for start in range(0, N_ROWS, GRU_CHUNK):
                end = min(start + GRU_CHUNK, N_ROWS)
                values = adaptive_chunk(x, start, end, state, name)
                prediction, hidden = model(torch.from_numpy(np.ascontiguousarray(values)), hidden)
                predictions.append(prediction[0].numpy())
            metric.add(y, np.concatenate(predictions), required)
            if n % 100 == 0:
                print(name, "fold", fold, "scored", n, flush=True)
    result = metric.result()
    (folder / "score.json").write_text(json.dumps(result, indent=2) + "\n")
    print(name, "fold", fold, result["weighted_pearson"], flush=True)


def train_regime(parquet, fold, fit):
    name = "regime_head"
    folder = OUT / f"fold_{fold}" / name
    folder.mkdir(parents=True, exist_ok=True)
    artifact = folder / "model.pt"
    if artifact.exists():
        print("using completed", artifact, flush=True)
        return
    with np.load(CACHE / f"improve/fold_{fold}/gru96_norm5ep/gru_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    threshold = regime_threshold(parquet, fit, mean, std)
    (folder / "threshold.json").write_text(json.dumps({"threshold": threshold}) + "\n")
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    model = RegimeGRU()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5):
        if epoch == 3:
            for group in optimizer.param_groups:
                group["lr"] = 0.0003
        losses = []
        order = rng.permutation(fit)
        for offset in range(0, len(order), GRU_BATCH):
            indices = order[offset:offset + GRU_BATCH]
            x = np.empty((len(indices), N_ROWS, N_FEATURES), np.float32)
            y = np.empty((len(indices), N_ROWS, 2), np.float32)
            for j, group in enumerate(indices):
                x[j], y[j] = read_group(parquet, group)
            x = np.clip((x - mean) / std, -10, 10)
            state, hidden = adaptive_state(len(indices)), None
            for start in range(0, N_ROWS, GRU_CHUNK):
                end = min(start + GRU_CHUNK, N_ROWS)
                high = volatility_chunk(x, start, end, state) > threshold
                xb = torch.from_numpy(np.ascontiguousarray(x[:, start:end]))
                hb = torch.from_numpy(np.ascontiguousarray(high))
                yb = torch.from_numpy(np.clip(np.ascontiguousarray(y[:, start:end]), -2, 2))
                optimizer.zero_grad()
                prediction, hidden = model(xb, hb, hidden)
                skip = WARMUP if start == 0 else 0
                loss = weighted_loss(prediction[:, skip:], yb[:, skip:])
                loss.backward()
                optimizer.step()
                hidden = hidden.detach()
                losses.append(loss.item())
            if (offset + len(indices)) % 1_024 < GRU_BATCH:
                print(name, "fold", fold, "epoch", epoch + 1,
                      "groups", offset + len(indices), flush=True)
        print(name, "fold", fold, "epoch", epoch + 1,
              "loss", float(np.mean(losses)), flush=True)
    torch.save(model.state_dict(), artifact)


def score_regime(parquet, fold, validation):
    name = "regime_head"
    folder = OUT / f"fold_{fold}" / name
    with np.load(CACHE / f"improve/fold_{fold}/gru96_norm5ep/gru_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    threshold = json.loads((folder / "threshold.json").read_text())["threshold"]
    model = RegimeGRU()
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    metric = GlobalWP()
    required = np.arange(N_ROWS) >= WARMUP
    with torch.no_grad():
        for n, group in enumerate(validation, 1):
            x, y = read_group(parquet, group)
            x = np.clip((x - mean) / std, -10, 10)[None]
            state, hidden, predictions = adaptive_state(1), None, []
            for start in range(0, N_ROWS, GRU_CHUNK):
                end = min(start + GRU_CHUNK, N_ROWS)
                high = volatility_chunk(x, start, end, state) > threshold
                values, hidden = model(torch.from_numpy(np.ascontiguousarray(x[:, start:end])),
                                       torch.from_numpy(np.ascontiguousarray(high)), hidden)
                predictions.append(values[0].numpy())
            metric.add(y, np.concatenate(predictions), required)
            if n % 100 == 0:
                print(name, "fold", fold, "scored", n, flush=True)
    result = metric.result()
    (folder / "score.json").write_text(json.dumps(result, indent=2) + "\n")
    print(name, "fold", fold, result["weighted_pearson"], flush=True)


def summarize(name):
    values = [json.loads((OUT / f"fold_{fold}" / name / "score.json").read_text())
              ["weighted_pearson"] for fold in range(1, 4)]
    baseline = 0.3657529475208234 if name in MLP else 0.415425098635339
    result = {"model": name, "folds": values, "mean": float(np.mean(values)),
              "baseline": baseline, "gain": float(np.mean(values) - baseline)}
    (OUT / f"{name}_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def regime_diagnostic(parquet, fold, fit, validation):
    folder = OUT / f"fold_{fold}" / "regime"
    folder.mkdir(parents=True, exist_ok=True)
    model_folder = CACHE / f"improve/fold_{fold}/gru96_norm5ep"
    with np.load(model_folder / "gru_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    threshold = regime_threshold(parquet, fit, mean, std)
    model = make_gru(96)
    model.load_state_dict(torch.load(model_folder / "scaled_gru.pt",
                                     map_location="cpu", weights_only=True))
    model.eval()
    scores = {key: GlobalWP() for key in ("low", "high")}
    required = np.arange(N_ROWS) >= WARMUP
    with torch.no_grad():
        for n, group in enumerate(validation, 1):
            x, y = read_group(parquet, group)
            x = np.clip((x - mean) / std, -10, 10)
            volatility = causal_volatility(x)
            hidden, predictions = None, []
            for start in range(0, N_ROWS, GRU_CHUNK):
                values, hidden = model(torch.from_numpy(
                    np.ascontiguousarray(x[start:start + GRU_CHUNK])[None]), hidden)
                predictions.append(values[0].numpy())
            predictions = np.concatenate(predictions)
            scores["low"].add(y, predictions, required & (volatility <= threshold))
            scores["high"].add(y, predictions, required & (volatility > threshold))
            if n % 100 == 0:
                print("regime fold", fold, "scored", n, flush=True)
    result = {"threshold": threshold,
              "fit_threshold_groups": 32,
              "scores": {key: value.result() for key, value in scores.items()}}
    (folder / "scores.json").write_text(json.dumps(result, indent=2) + "\n")
    print("regime fold", fold, json.dumps(result), flush=True)


def predict_sequence(model, x):
    hidden, predictions = None, []
    for start in range(0, N_ROWS, GRU_CHUNK):
        values, hidden = model(torch.from_numpy(
            np.ascontiguousarray(x[start:start + GRU_CHUNK])[None]), hidden)
        predictions.append(values[0].numpy())
    return np.concatenate(predictions)


def blend_screen(parquet, fold, validation):
    folder = OUT / f"fold_{fold}" / "blend"
    folder.mkdir(parents=True, exist_ok=True)
    improve = CACHE / f"improve/fold_{fold}"
    with np.load(improve / "gru96_norm5ep/gru_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    models = {}
    for name, width, path in (
        ("norm5", 96, improve / "gru96_norm5ep/scaled_gru.pt"),
        ("seed17", 96, improve / "gru96_seed17/scaled_gru.pt"),
        ("seed37", 96, improve / "gru96_seed37/scaled_gru.pt"),
    ):
        model = make_gru(width)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        models[name] = model.eval()
    split = SplitEncoderGRU()
    split.load_state_dict(torch.load(OUT / f"fold_{fold}/split_encoder/model.pt",
                                     map_location="cpu", weights_only=True))
    models["split"] = split.eval()
    wide = make_gru(128)
    wide.load_state_dict(torch.load(OUT / f"fold_{fold}/gru128_5ep/model.pt",
                                    map_location="cpu", weights_only=True))
    models["wide"] = wide.eval()
    multiscale_path = OUT / f"fold_{fold}/multiscale96/model.pt"
    if multiscale_path.exists():
        multiscale = MultiScaleMixer()
        multiscale.load_state_dict(torch.load(multiscale_path, map_location="cpu", weights_only=True))
        models["multiscale"] = multiscale.eval()
    difference_path = CACHE / f"difference_cv/fold_{fold}/model.pt"
    if difference_path.exists():
        difference = make_gru(96)
        difference.head = torch.nn.Linear(96, 1)
        difference.load_state_dict(torch.load(difference_path, map_location="cpu", weights_only=True))
        models["difference"] = difference.eval()
    names = ("ensemble", "ensemble_split25", "ensemble_wide25",
             "ensemble_both10", "equal_three")
    if "multiscale" in models:
        names += ("ensemble_multiscale10", "ensemble_multiscale25",
                  "equal_three_multiscale10")
    if "difference" in models:
        names += ("ensemble_difference50", "equal_three_difference50")
    scores = {name: GlobalWP() for name in names}
    required = np.arange(N_ROWS) >= WARMUP
    with torch.no_grad():
        for n, group in enumerate(validation, 1):
            raw, y = read_group(parquet, group)
            normalized = np.clip((raw - mean) / std, -10, 10)
            p = {
                "norm5": predict_sequence(models["norm5"], normalized),
                "seed17": predict_sequence(models["seed17"], raw),
                "seed37": predict_sequence(models["seed37"], raw),
                "split": predict_sequence(models["split"], normalized),
                "wide": predict_sequence(models["wide"], normalized),
            }
            if "multiscale" in models:
                p["multiscale"] = predict_sequence(models["multiscale"], normalized)
            if "difference" in models:
                p["difference"] = predict_sequence(models["difference"], normalized)[:, :1]
            ensemble = (p["norm5"] + p["seed17"] + p["seed37"]) / 3
            equal_three = (ensemble + p["split"] + p["wide"]) / 3
            predictions = {
                "ensemble": ensemble,
                "ensemble_split25": 0.75 * ensemble + 0.25 * p["split"],
                "ensemble_wide25": 0.75 * ensemble + 0.25 * p["wide"],
                "ensemble_both10": 0.8 * ensemble + 0.1 * p["split"] + 0.1 * p["wide"],
                "equal_three": equal_three,
            }
            if "multiscale" in models:
                predictions.update({
                    "ensemble_multiscale10": 0.9 * ensemble + 0.1 * p["multiscale"],
                    "ensemble_multiscale25": 0.75 * ensemble + 0.25 * p["multiscale"],
                    "equal_three_multiscale10": 0.9 * equal_three + 0.1 * p["multiscale"],
                })
            if "difference" in models:
                for name, base in (("ensemble_difference50", ensemble),
                                   ("equal_three_difference50", equal_three)):
                    spread = -(base[:, :1] + base[:, 1:])
                    direction = 0.5 * (base[:, :1] - base[:, 1:] + p["difference"])
                    predictions[name] = np.column_stack(
                        ((direction - spread)[:, 0] / 2, (-direction - spread)[:, 0] / 2))
            for name, values in predictions.items():
                scores[name].add(y, values, required)
            if n % 100 == 0:
                print("blend fold", fold, "scored", n, flush=True)
    result = {name: score.result() for name, score in scores.items()}
    expected = (0.41376846879991985, 0.4355001264892581, 0.4175703181321274)[fold - 1]
    if not np.isclose(result["ensemble"]["weighted_pearson"], expected, atol=1e-9):
        raise ValueError("existing ensemble did not reproduce")
    (folder / "scores.json").write_text(json.dumps(result, indent=2) + "\n")
    print("blend fold", fold, {name: value["weighted_pearson"]
                               for name, value in result.items()}, flush=True)


def blend_summary():
    results = [json.loads((OUT / f"fold_{fold}/blend/scores.json").read_text())
               for fold in range(1, 4)]
    summary = {}
    for name in results[0]:
        values = [fold[name]["weighted_pearson"] for fold in results]
        summary[name] = {"folds": values, "mean": float(np.mean(values))}
    reference = summary["ensemble"]["mean"]
    for value in summary.values():
        value["gain_vs_ensemble"] = value["mean"] - reference
    (OUT / "blend_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def selfcheck():
    x = np.arange(2 * 140 * N_FEATURES, dtype=np.float32).reshape(2, 140, N_FEATURES)
    states = [((1 - 2 / 9) * x[:, :1]).copy()]
    current, history = history_chunk(x, 0, 64, (8,), states)
    assert current.shape == (2, 64, N_FEATURES)
    assert history.shape == (2, 64, N_FEATURES, 5)
    assert np.all(history[:, :1, :, :4] == 0)
    assert np.array_equal(history[:, 8, :, 1], x[:, 8] - x[:, 0])
    _, second = history_chunk(x, 64, 140, (8,), states)
    assert np.all(second[:, 0, :, 3] == 0)
    assert np.array_equal(second[:, 64, :, 3], x[:, 128] - x[:, 0])
    assert np.isfinite(second).all()
    model = MultiScaleGRU(5)
    prediction, _ = model(torch.from_numpy(current), torch.from_numpy(history))
    assert prediction.shape == (2, 64, 2) and np.isfinite(prediction.detach().numpy()).all()
    for name in PLAIN:
        prediction, _ = make_plain(name)(torch.from_numpy(current))
        assert prediction.shape == (2, 64, 2)
    prediction, _ = RegimeGRU()(torch.from_numpy(current),
                                torch.zeros((2, 64), dtype=torch.bool))
    assert prediction.shape == (2, 64, 2)
    tcn = CausalTCN().eval()
    values = torch.from_numpy(x[:, :128])
    with torch.no_grad():
        full, _ = tcn(values)
        first, state = tcn(values[:, :64])
        second, _ = tcn(values[:, 64:], state)
    assert torch.allclose(full, torch.cat((first, second), dim=1), atol=1e-6)
    state = adaptive_state(2)
    whole = adaptive_chunk(x, 0, 140, state, "causal_norm")
    state = adaptive_state(2)
    chunked = np.concatenate((adaptive_chunk(x, 0, 64, state, "causal_norm"),
                              adaptive_chunk(x, 64, 140, state, "causal_norm")), axis=1)
    assert np.allclose(whole, chunked, atol=1e-6)
    print("advance causal feature and model checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("selfcheck", "run", "summary", "regime",
                                            "blend", "blend_summary"))
    parser.add_argument("--fold", type=int, choices=(1, 2, 3))
    parser.add_argument("--name", choices=(*OPTIONS, *PLAIN, *MLP, *ADAPTIVE, *REGIME))
    args = parser.parse_args()
    if args.command == "selfcheck":
        selfcheck()
        return
    if args.command == "blend_summary":
        blend_summary()
        return
    if args.command not in ("selfcheck", "regime", "blend") and args.name is None:
        raise ValueError("--name is required")
    if args.command == "summary":
        summarize(args.name)
        return
    torch.set_num_threads(1)
    parquet = pq.ParquetFile(ROOT / "datasets/train.parquet")
    check_train(parquet)
    for fold, fit, validation in folds():
        if args.fold is None or args.fold == fold:
            if args.command == "regime":
                regime_diagnostic(parquet, fold, fit, validation)
            elif args.command == "blend":
                blend_screen(parquet, fold, validation)
            elif args.name in REGIME:
                train_regime(parquet, fold, fit)
                score_regime(parquet, fold, validation)
            elif args.name in ADAPTIVE:
                train_adaptive(parquet, fold, fit, args.name)
                score_adaptive(parquet, fold, validation, args.name)
            elif args.name in MLP:
                train_mlp_offset(parquet, fold, fit)
                score_mlp_offset(parquet, fold, validation)
            elif args.name in PLAIN:
                train_plain(parquet, fold, fit, args.name)
                score_plain(parquet, fold, validation, args.name)
            else:
                train(parquet, fold, fit, args.name)
                score(parquet, fold, validation, args.name)


if __name__ == "__main__":
    main()
