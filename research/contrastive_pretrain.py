"""Locked pilot: causal predictive-contrastive pretraining for the GRU encoder."""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import torch

from event_pretrain import BATCH, CHUNK, finetune, normalize
from maskft import detach_state
from pilot import CACHE, DEV_GROUPS, FIT_GROUPS, N_FEATURES, N_ROWS, SEED, WARMUP, make_gru

OUT = CACHE / "contrastive_pretrain"
HORIZONS = (1, 8, 32)


class PredictiveContrast(torch.nn.Module):
    def __init__(self, recurrent, embedding=32):
        super().__init__()
        self.rnn = recurrent
        self.target = torch.nn.Linear(N_FEATURES, embedding, bias=False)
        self.predictors = torch.nn.ModuleList(
            torch.nn.Linear(recurrent.hidden_size, embedding) for _ in HORIZONS)

    def loss(self, values, hidden=None, stride=32, temperature=0.1):
        states, hidden = self.rnn(values, hidden)
        positions = torch.arange(0, values.shape[1] - max(HORIZONS), stride,
                                 device=values.device)
        labels = torch.arange(values.shape[0] * len(positions), device=values.device)
        losses = []
        for horizon, predictor in zip(HORIZONS, self.predictors):
            context = torch.nn.functional.normalize(
                predictor(states[:, positions]).flatten(0, 1), dim=1)
            target = torch.nn.functional.normalize(
                self.target(values[:, positions + horizon]).flatten(0, 1), dim=1)
            losses.append(torch.nn.functional.cross_entropy(context @ target.T / temperature,
                                                             labels))
        return torch.stack(losses).mean(), hidden


def pretrain(model, x, mean, std, device, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(SEED + 11)
    history = []
    for epoch in range(epochs):
        losses = []
        order = rng.permutation(len(x))
        for offset in range(0, len(x), BATCH):
            values = normalize(np.array(x[order[offset:offset + BATCH]]), mean, std)
            hidden = None
            for start in range(0, N_ROWS, CHUNK):
                stop = min(start + CHUNK, N_ROWS)
                if stop - start <= max(HORIZONS):
                    continue
                xb = torch.from_numpy(np.ascontiguousarray(values[:, start:stop])).to(device)
                optimizer.zero_grad()
                loss, hidden = model.loss(xb, hidden)
                loss.backward()
                optimizer.step()
                hidden = detach_state(hidden)
                losses.append(loss.item())
        history.append(float(np.mean(losses)))
        print("contrastive epoch", epoch + 1, "loss", round(history[-1], 6), flush=True)
    return history


def selfcheck(device="cpu"):
    torch.manual_seed(1)
    base = make_gru(8)
    model = PredictiveContrast(base.rnn, 4).to(device)
    values = torch.randn(3, 80, N_FEATURES, device=device)
    loss, hidden = model.loss(values, stride=8)
    loss.backward()
    assert loss.isfinite() and hidden.shape == (1, 3, 8)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    with torch.no_grad():
        full = model.rnn(values)[0]
        first, state = model.rnn(values[:, :40])
        second = model.rnn(values[:, 40:], state)[0]
    assert torch.allclose(full, torch.cat((first, second), 1), atol=1e-6)
    print("contrastive loss and causal chunk state checks passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--target-epochs", type=int, default=5)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck(args.device)
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

    torch.manual_seed(SEED)
    initial = make_gru(96)
    control = copy.deepcopy(initial).to(args.device)
    contrastive = PredictiveContrast(copy.deepcopy(initial.rnn)).to(args.device)
    pretrain_history = pretrain(contrastive, x, mean, std, args.device, args.pretrain_epochs)
    candidate = copy.deepcopy(initial)
    candidate.rnn.load_state_dict(contrastive.rnn.state_dict())
    candidate = candidate.to(args.device)
    control_history = finetune("control", control, x, y, dx, dy, mean, std, args.device,
                               args.target_epochs)
    candidate_history = finetune("contrastive", candidate, x, y, dx, dy, mean, std, args.device,
                                 args.target_epochs)
    control_best = max(row["weighted_pearson"] for row in control_history[1:])
    candidate_best = max(row["weighted_pearson"] for row in candidate_history[1:])
    result = {"device": args.device, "horizons": HORIZONS,
              "pretrain_loss": pretrain_history, "control": control_history,
              "contrastive": candidate_history, "control_best": control_best,
              "contrastive_best": candidate_best, "gain": candidate_best - control_best,
              "passes_gate": candidate_best >= control_best + 0.005}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pilot.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(candidate.cpu().state_dict(), OUT / "pilot.pt")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
