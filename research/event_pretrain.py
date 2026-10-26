"""Pilot: pretrain a GRU on the next i0 price event, then fine-tune on targets."""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import torch

from global_wp import GlobalWP
from maskft import detach_state, next_event_deltas
from pilot import CACHE, DEV_GROUPS, FIT_GROUPS, N_FEATURES, N_ROWS, SEED, WARMUP, make_gru

OUT = CACHE / "event_pretrain"
CHUNK = 512
BATCH = 16
WIDTH = 96


class EventGRU(torch.nn.Module):
    def __init__(self, recurrent):
        super().__init__()
        self.rnn = recurrent
        self.head = torch.nn.Linear(recurrent.hidden_size, 22)

    def forward(self, values, hidden=None):
        values, hidden = self.rnn(values, hidden)
        return self.head(values), hidden


class EventBottleneck(torch.nn.Module):
    """Force target predictions through a supervised next-event forecast."""

    def __init__(self, event_model):
        super().__init__()
        self.rnn = event_model.rnn
        self.event_head = event_model.head
        self.head = torch.nn.Linear(44, 2)

    def forward(self, values, hidden=None):
        states, hidden = self.rnn(values, hidden)
        event = self.event_head(states)
        return self.head(torch.cat((values[..., :22], event), -1)), hidden, event


def normalize(raw, mean, std):
    return np.clip((raw - mean) / std, -10, 10).astype(np.float32)


def weighted_loss(prediction, target):
    weight = target.abs()
    return ((prediction - target).square() * weight).sum() / weight.sum().clamp_min(1e-6)


def score(model, x, y, mean, std, device):
    acc = GlobalWP()
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(x), BATCH):
            raw = np.array(x[offset:offset + BATCH])
            values = normalize(raw, mean, std)
            hidden = None
            for start in range(0, N_ROWS, CHUNK):
                stop = min(start + CHUNK, N_ROWS)
                output = model(torch.from_numpy(np.ascontiguousarray(values[:, start:stop])).to(device), hidden)
                prediction, hidden = output[:2]
                prediction = prediction.cpu().numpy()
                for i in range(len(raw)):
                    mask = np.arange(start, stop) >= WARMUP
                    acc.add(y[offset + i, start:stop], prediction[i], mask)
    model.train()
    return acc.result()


def pretrain(model, x, mean, std, device, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(SEED + 1)
    for epoch in range(epochs):
        losses = []
        order = rng.permutation(len(x))
        for offset in range(0, len(x), BATCH):
            indices = order[offset:offset + BATCH]
            values = normalize(np.array(x[indices]), mean, std)
            event_y, valid = next_event_deltas(values)
            valid[:, :WARMUP] = False
            hidden = None
            for start in range(0, N_ROWS, CHUNK):
                stop = min(start + CHUNK, N_ROWS)
                xb = torch.from_numpy(np.ascontiguousarray(values[:, start:stop])).to(device)
                yb = torch.from_numpy(np.ascontiguousarray(event_y[:, start:stop])).to(device)
                vb = torch.from_numpy(np.ascontiguousarray(valid[:, start:stop])).to(device)
                optimizer.zero_grad()
                prediction, hidden = model(xb, hidden)
                error = (prediction - yb).square() * vb[..., None]
                loss = error.sum() / (vb.sum() * prediction.shape[-1]).clamp_min(1)
                loss.backward()
                optimizer.step()
                hidden = detach_state(hidden)
                losses.append(loss.item())
        print("event epoch", epoch + 1, "loss", round(float(np.mean(losses)), 6), flush=True)


def finetune(name, model, x, y, dx, dy, mean, std, device, epochs, event_aux=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(SEED)
    history = [score(model, dx, dy, mean, std, device)]
    print(name, "epoch 0", history[-1], flush=True)
    for epoch in range(epochs):
        if epoch == 3:
            for group in optimizer.param_groups:
                group["lr"] = 3e-4
        losses = []
        order = rng.permutation(len(x))
        for offset in range(0, len(x), BATCH):
            indices = order[offset:offset + BATCH]
            values = normalize(np.array(x[indices]), mean, std)
            targets = np.clip(np.array(y[indices]), -2, 2).astype(np.float32)
            event_y, valid = next_event_deltas(values) if event_aux else (None, None)
            if event_aux:
                valid[:, :WARMUP] = False
            hidden = None
            for start in range(0, N_ROWS, CHUNK):
                stop = min(start + CHUNK, N_ROWS)
                xb = torch.from_numpy(np.ascontiguousarray(values[:, start:stop])).to(device)
                yb = torch.from_numpy(np.ascontiguousarray(targets[:, start:stop])).to(device)
                optimizer.zero_grad()
                output = model(xb, hidden)
                prediction, hidden = output[:2]
                begin = WARMUP if start == 0 else 0
                loss = weighted_loss(prediction[:, begin:], yb[:, begin:])
                if event_aux:
                    eb = torch.from_numpy(np.ascontiguousarray(event_y[:, start:stop])).to(device)
                    vb = torch.from_numpy(np.ascontiguousarray(valid[:, start:stop])).to(device)
                    event_error = (output[2] - eb).square() * vb[..., None]
                    loss = loss + event_aux * event_error.sum() / (
                        vb.sum() * event_error.shape[-1]).clamp_min(1)
                loss.backward()
                optimizer.step()
                hidden = detach_state(hidden)
                losses.append(loss.item())
        history.append(score(model, dx, dy, mean, std, device))
        print(name, "epoch", epoch + 1, "loss", round(float(np.mean(losses)), 6),
              history[-1], flush=True)
    return history


def selfcheck():
    values = np.zeros((1, 6, N_FEATURES), np.float32)
    values[0, 2:, :22] = 1
    values[0, 5:, :22] = 3
    target, valid = next_event_deltas(values)
    assert valid.tolist() == [[True, True, True, True, True, False]]
    assert np.all(target[0, :2] == 1) and np.all(target[0, 2:5] == 2)
    torch.manual_seed(1)
    model = make_gru(4).eval()
    x = torch.randn(2, 11, N_FEATURES)
    full = model(x)[0]
    first, hidden = model(x[:, :6])
    second, _ = model(x[:, 6:], hidden)
    assert torch.allclose(full, torch.cat((first, second), 1), atol=1e-7)
    bottleneck = EventBottleneck(EventGRU(copy.deepcopy(model.rnn))).eval()
    full = bottleneck(x)[0]
    first, hidden = bottleneck(x[:, :6])[:2]
    second = bottleneck(x[:, 6:], hidden)[0]
    assert torch.allclose(full, torch.cat((first, second), 1), atol=1e-7)
    print("event labels and stateful chunk inference agree")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-epochs", type=int, default=3)
    parser.add_argument("--target-epochs", type=int, default=5)
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
    del sampled

    torch.manual_seed(SEED)
    initial = make_gru(WIDTH)
    control = copy.deepcopy(initial).to(args.device)
    pretrained = EventGRU(copy.deepcopy(initial.rnn)).to(args.device)
    pretrain(pretrained, x, mean, std, args.device, args.event_epochs)
    candidate = copy.deepcopy(initial)
    candidate.rnn.load_state_dict(pretrained.rnn.state_dict())
    candidate = candidate.to(args.device)
    bottleneck = EventBottleneck(copy.deepcopy(pretrained)).to(args.device)

    control_history = finetune("control", control, x, y, dx, dy, mean, std, args.device,
                               args.target_epochs)
    candidate_history = finetune("pretrained", candidate, x, y, dx, dy, mean, std, args.device,
                                 args.target_epochs)
    bottleneck_history = finetune("bottleneck", bottleneck, x, y, dx, dy, mean, std, args.device,
                                  args.target_epochs, event_aux=0.1)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"event_epochs": args.event_epochs, "target_epochs": args.target_epochs,
              "device": args.device, "control": control_history, "pretrained": candidate_history,
              "bottleneck": bottleneck_history,
              "pretrained_best_gain": max(v["weighted_pearson"] for v in candidate_history[1:]) -
                                      max(v["weighted_pearson"] for v in control_history[1:]),
              "bottleneck_best_gain": max(v["weighted_pearson"] for v in bottleneck_history[1:]) -
                                      max(v["weighted_pearson"] for v in control_history[1:])}
    (OUT / "pilot.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(candidate.cpu().state_dict(), OUT / "pilot_pretrained.pt")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
