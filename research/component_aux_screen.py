"""Locked pilot: compress the useful dedicated-D objective into one GRU."""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import torch

from event_pretrain import BATCH, CHUNK, normalize, weighted_loss
from global_wp import GlobalWP
from maskft import detach_state
from pilot import CACHE, DEV_GROUPS, FIT_GROUPS, N_FEATURES, N_ROWS, SEED, WARMUP, make_gru

OUT = CACHE / "component_aux_screen"
AUXILIARY = 0.5
MIXES = (0.0, 0.25, 0.5, 1.0)
CONTROL = 0.3160643776683788


class DirectionAuxGRU(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.rnn, self.head = base.rnn, base.head
        self.direction = torch.nn.Linear(self.rnn.hidden_size, 1)
        with torch.no_grad():
            self.direction.weight.copy_(self.head.weight[:1] - self.head.weight[1:])
            self.direction.bias.copy_(self.head.bias[:1] - self.head.bias[1:])

    def forward(self, values, hidden=None):
        states, hidden = self.rnn(values, hidden)
        return self.head(states), self.direction(states), hidden


def train_epoch(model, optimizer, x, y, mean, std, device, rng):
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
            prediction, direction, hidden = model(xb, hidden)
            begin = WARMUP if start == 0 else 0
            base_loss = weighted_loss(prediction[:, begin:], yb[:, begin:])
            d = yb[..., :1] - yb[..., 1:]
            weight = yb.abs().sum(-1, keepdim=True)
            d_loss = ((direction[:, begin:] - d[:, begin:]).square() *
                      weight[:, begin:]).sum() / weight[:, begin:].sum().clamp_min(1e-6)
            loss = base_loss + AUXILIARY * d_loss
            loss.backward()
            optimizer.step()
            hidden = detach_state(hidden)
            losses.append(loss.item())
    return float(np.mean(losses))


def score(model, x, y, mean, std, device):
    metrics = {mix: GlobalWP() for mix in MIXES}
    required = np.arange(N_ROWS) >= WARMUP
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(x), BATCH):
            values = normalize(np.array(x[offset:offset + BATCH]), mean, std)
            hidden, predictions = None, {mix: [] for mix in MIXES}
            for start in range(0, N_ROWS, CHUNK):
                xb = torch.from_numpy(np.ascontiguousarray(values[:, start:start + CHUNK])).to(device)
                base, d_hat, hidden = model(xb, hidden)
                spread = -(base[..., :1] + base[..., 1:])
                base_d = base[..., :1] - base[..., 1:]
                for mix in MIXES:
                    direction = (1 - mix) * base_d + mix * d_hat
                    predictions[mix].append(torch.cat(
                        ((direction - spread) / 2, (-direction - spread) / 2), -1).cpu().numpy())
            for i in range(len(values)):
                for mix, parts in predictions.items():
                    metrics[mix].add(y[offset + i], np.concatenate(parts, 1)[i], required)
    model.train()
    return {str(mix): metric.result() for mix, metric in metrics.items()}


def selfcheck():
    torch.manual_seed(1)
    base = make_gru(8)
    model = DirectionAuxGRU(copy.deepcopy(base)).eval()
    x = torch.randn(2, 17, N_FEATURES)
    raw = base(x)[0]
    joint, direction, _ = model(x)
    assert torch.allclose(raw, joint)
    assert torch.allclose(direction, joint[..., :1] - joint[..., 1:], atol=1e-6)
    print("direction head initialization exactly reconstructs the joint head")


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
    model = DirectionAuxGRU(make_gru(96)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(args.epochs):
        if epoch == 3:
            for group in optimizer.param_groups:
                group["lr"] = 3e-4
        loss = train_epoch(model, optimizer, x, y, mean, std, args.device, rng)
        scores = score(model, dx, dy, mean, std, args.device)
        history.append({"epoch": epoch + 1, "loss": loss, "scores": scores})
        print("component_aux epoch", epoch + 1, "loss", round(loss, 6),
              {key: round(value["weighted_pearson"], 6) for key, value in scores.items()},
              flush=True)
    frozen = history[-1]["scores"]["0.5"]["weighted_pearson"]
    result = {"seed": args.seed, "auxiliary": AUXILIARY, "history": history,
              "control": CONTROL, "frozen_epoch5_mix05": frozen,
              "gain": frozen - CONTROL, "passes_gate": frozen >= CONTROL + 0.005}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"seed{args.seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(model.cpu().state_dict(), OUT / f"seed{args.seed}.pt")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
