"""Cheap screens for alternative output heads on a frozen split-encoder GRU."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pyarrow.parquet as pq
import torch

from global_wp import GlobalWP
from maskft import BATCH, CHUNK, OUT as FT_OUT, batch_arrays, load, weighted_loss
from pilot import N_ROWS, ROOT, WARMUP

OUT = FT_OUT.parent / "head_screen"
N_BINS = 33


class Heads(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        width = source.head.in_features
        self.control = torch.nn.Linear(width, 2)
        self.distribution = torch.nn.Linear(width, 2 * N_BINS)
        self.mask = torch.nn.Linear(width, 1)
        self.low = torch.nn.Linear(width, 2)
        self.high = torch.nn.Linear(width, 2)
        for head in (self.control, self.low, self.high):
            head.load_state_dict(source.head.state_dict())
        self.register_buffer("centers", torch.linspace(-2, 2, N_BINS))

    def predictions(self, hidden):
        logits = self.distribution(hidden).reshape(*hidden.shape[:2], 2, N_BINS)
        q = torch.sigmoid(self.mask(hidden))
        return {"control": self.control(hidden),
                "distribution": (logits.softmax(-1) * self.centers).sum(-1),
                "distribution_hard": self.centers[logits.argmax(-1)],
                "mask_gated": (1 - q) * self.low(hidden) + q * self.high(hidden)}, logits, q


def recurrent(model, values, state=None):
    i0, i1 = model.instrument(values[..., :52]), model.instrument(values[..., 52:104])
    encoded = torch.cat((i0, i1, i0 - i1, i0 * i1, model.additional(values[..., 104:112])), -1)
    return model.rnn(encoded, state)


def score(model, heads, parquet, groups, scaler, cache=None):
    scores = {name: GlobalWP() for name in ("base", "control", "distribution",
                                             "distribution_hard", "mask_gated")}
    selected = {name: [] for name in (*scores, "y")} if cache else None
    model.eval(); heads.eval()
    with torch.no_grad():
        for offset in range(0, len(groups), BATCH):
            x, y, mask = batch_arrays(parquet, groups[offset:offset + BATCH], scaler)
            x = torch.from_numpy(x)
            state, pieces = None, {name: [] for name in scores}
            for start in range(0, N_ROWS, CHUNK):
                hidden, state = recurrent(model, x[:, start:start + CHUNK], state)
                prediction, _, _ = heads.predictions(hidden)
                pieces["base"].append(model.head(hidden).numpy())
                for name, values in prediction.items():
                    pieces[name].append(values.numpy())
            for name, values in pieces.items():
                values = np.concatenate(values, 1)
                for i in range(len(x)):
                    scores[name].add(y[i], values[i], mask[i])
                if selected is not None:
                    selected[name].append(values[mask])
            if selected is not None:
                selected["y"].append(y[mask])
    if cache:
        np.savez(cache, **{name: np.concatenate(values) for name, values in selected.items()})
    heads.train()
    return {name: value.result()["weighted_pearson"] for name, value in scores.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--half", type=int, required=True, choices=(-1, 0, 1))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--score-swa", action="store_true")
    parser.add_argument("--finetuned-base", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.manual_seed(91)
    rng = np.random.default_rng(91)
    groups = np.arange(1_873)
    fit = groups if args.half < 0 else groups[groups % 2 == args.half]
    evaluate = None if args.half < 0 else groups[groups % 2 != args.half]
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    model, scaler = load("split128_a0.75_s81")
    half_name = "all" if args.half < 0 else f"h{args.half}"
    tag = "tbbase_" if args.finetuned_base else ""
    if args.finetuned_base:
        states = [torch.load(FT_OUT / f"split128_a0.75_s81_u0.06_lr0.0002_s7_tb_{half_name}_e{e}.pt",
                             weights_only=True) for e in (2, 3)]
        model.load_state_dict({key: (states[0][key] + states[1][key]) / 2 for key in states[0]})
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    heads = Heads(model)
    if args.score_swa:
        if evaluate is None:
            raise ValueError("--score-swa requires a held-out half")
        states = [torch.load(OUT / f"{tag}{half_name}_e{epoch}.pt", weights_only=True) for epoch in (2, 3)]
        heads.load_state_dict({key: (states[0][key] + states[1][key]) / 2 for key in states[0]})
        print(score(model, heads, parquet, evaluate, scaler, OUT / f"pred_{tag}{half_name}.npz"))
        return
    slow = [*heads.control.parameters(), *heads.mask.parameters(),
            *heads.low.parameters(), *heads.high.parameters()]
    optimizer = torch.optim.Adam(({"params": slow, "lr": 2e-4},
                                  {"params": heads.distribution.parameters(), "lr": 1e-3}))
    history, states = ([] if evaluate is None else [score(model, heads, parquet, evaluate, scaler)]), []
    print("half", args.half, "epoch 0", history[-1] if history else None, flush=True)
    for epoch in range(args.epochs):
        losses = []
        order = rng.permutation(fit)
        for offset in range(0, len(fit), BATCH):
            batch = order[offset:offset + BATCH]
            x, y, mask = batch_arrays(parquet, batch, scaler)
            x = torch.from_numpy(x)
            y = torch.from_numpy(np.clip(y, -2, 2))
            row_weight = torch.from_numpy(np.where(mask, 1.0, 0.06).astype(np.float32))
            row_weight[:, :WARMUP] = 0
            state = None
            for start in range(0, N_ROWS, CHUNK):
                sl = slice(start, start + CHUNK)
                with torch.no_grad():
                    hidden, state = recurrent(model, x[:, sl], state)
                prediction, logits, q = heads.predictions(hidden)
                target = y[:, sl]
                weight = row_weight[:, sl]
                classes = ((target + 2) * ((N_BINS - 1) / 4)).round().long().clamp(0, N_BINS - 1)
                ce = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 2), classes.flatten(), reduction="none").reshape_as(target)
                weighted = weight[..., None] * target.abs()
                distribution = (ce.mul(weighted).sum((0, 1)) /
                                weighted.sum((0, 1)).clamp_min(1e-6)).mean()
                gate = torch.nn.functional.binary_cross_entropy(q[..., 0],
                                                                 torch.from_numpy(mask[:, sl].astype(np.float32)))
                loss = (weighted_loss(prediction["control"], target, weight, True) + distribution +
                        weighted_loss(prediction["mask_gated"], target, weight, True) + 0.1 * gate)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                state = state.detach(); losses.append(float(loss.detach()))
        states.append({key: value.detach().clone() for key, value in heads.state_dict().items()})
        torch.save(states[-1], OUT / f"{tag}{half_name}_e{epoch + 1}.pt")
        if evaluate is not None:
            history.append(score(model, heads, parquet, evaluate, scaler))
        print("half", args.half, "epoch", epoch + 1, "loss", float(np.mean(losses)),
              history[-1] if history else None, flush=True)
    heads.load_state_dict({key: (states[-2][key] + states[-1][key]) / 2 for key in states[-1]})
    swa = score(model, heads, parquet, evaluate, scaler) if evaluate is not None else None
    result = {"half": args.half, "history": history, "swa_e2_e3": swa}
    (OUT / f"{tag}{half_name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print("half", args.half, "SWA", swa, flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
