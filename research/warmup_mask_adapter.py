"""Cross-fit a warm-up context adapter on the v24 C71 lineage.

The recurrent model starts as an exact copy of the SWA of C71 continuation
epochs 1--3.  Its old parameters and the first 112 input columns are frozen;
only a new 16-dimensional context path derived from rows 0--98 is trained.
Each validation parity is fitted and scored on the opposite parity.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pyarrow.parquet as pq
import torch

from ensemble_screen import MEMBERS, load as load_ensemble, wp
from maskft import batch_arrays, detach_state, weighted_loss
from oos_global import read_group as read_valid_group
from pilot import CACHE, N_FEATURES, N_ROWS, ROOT, WARMUP
from stack import FULL


OUT = CACHE / "warmup_adapter"
BATCH = 16
CHUNK = 512
WIDTH = 128
CONTEXT = 16


class WarmupAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.context_net = torch.nn.Sequential(
            torch.nn.Linear(3 * N_FEATURES, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, CONTEXT),
            torch.nn.Tanh(),
        )
        self.rnn = torch.nn.GRU(N_FEATURES + CONTEXT, WIDTH, batch_first=True)
        self.head = torch.nn.Linear(WIDTH, 2)

    def context(self, values):
        warmup = values[:, :WARMUP]
        return self.context_net(torch.cat((warmup.mean(1), warmup.std(1, unbiased=False),
                                           (warmup[:, 1:] - warmup[:, :-1]).abs().mean(1)), -1))

    def forward(self, values, context, hidden=None):
        repeated = context[:, None].expand(-1, values.shape[1], -1)
        states, hidden = self.rnn(torch.cat((values, repeated), -1), hidden)
        return self.head(states), hidden


def continued_state(half):
    prefix = (CACHE / "ensdistill" /
              f"scratch128g2_a0.75_t6000_p6000_lr0.0002_s71_cont_h{half}")
    states = [torch.load(f"{prefix}_e{epoch}.pt", map_location="cpu", weights_only=True)
              for epoch in (1, 2, 3)]
    return {key: sum(state[key] for state in states) / len(states) for key in states[0]}


def initialize(half):
    base = continued_state(half)
    model = WarmupAdapter()
    with torch.no_grad():
        model.rnn.weight_ih_l0[:, :N_FEATURES].copy_(base["rnn.weight_ih_l0"])
        model.rnn.weight_ih_l0[:, N_FEATURES:].zero_()
        for name in ("weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
            getattr(model.rnn, name).copy_(base[f"rnn.{name}"])
        model.head.weight.copy_(base["head.weight"])
        model.head.bias.copy_(base["head.bias"])
    for parameter in (model.rnn.weight_hh_l0, model.rnn.bias_ih_l0,
                      model.rnn.bias_hh_l0, model.head.weight, model.head.bias):
        parameter.requires_grad_(False)
    model.rnn.weight_ih_l0.register_hook(
        lambda gradient: torch.cat((torch.zeros_like(gradient[:, :N_FEATURES]),
                                    gradient[:, N_FEATURES:]), 1))
    return model


def predict(model, values):
    context = model.context(values)
    hidden, pieces = None, []
    for start in range(0, N_ROWS, CHUNK):
        output, hidden = model(values[:, start:start + CHUNK], context, hidden)
        pieces.append(output)
    return torch.cat(pieces, 1)


def score_and_cache(model, parquet, groups, scaler, path=None):
    predictions, targets = [], []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(groups), BATCH):
            x, y, mask = batch_arrays(parquet, groups[offset:offset + BATCH], scaler)
            output = predict(model, torch.from_numpy(x)).numpy()
            predictions.append(output[mask])
            targets.append(y[mask])
            if offset and offset % 160 == 0:
                print("warmup adapter scored groups", offset, flush=True)
    model.train()
    p, y = np.concatenate(predictions), np.concatenate(targets)
    if path is not None:
        np.savez(path, p=p, y=y)
    return wp(y, p), p, y


def blend_report(half, candidate, y):
    cached, cached_y, _ = load_ensemble(half)
    if not np.array_equal(y, cached_y):
        raise ValueError("warm-up adapter and cached ensemble rows do not align")
    names = list(MEMBERS)
    c71, c72, split = (names.index(name) for name in ("c71", "c72", "split"))
    base = 0.4 * cached[:, c71] + 0.4 * cached[:, c72] + 0.2 * cached[:, split]
    result = {"base": wp(y, base), "candidate": wp(y, candidate),
              "replace_c71": wp(y, 0.4 * candidate + 0.4 * cached[:, c72] +
                                  0.2 * cached[:, split])}
    for weight in (0.05, 0.10, 0.20, 0.25):
        result[f"add_{weight:.2f}"] = wp(y, (1 - weight) * base + weight * candidate)
    return result


def fit(half, epochs, lr, seed):
    with np.load(FULL / "gru96_norm5ep/gru_standardizer.npz") as source:
        scaler = (source["mean"], source["std"], False)
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    groups = np.arange(1_873)
    fit_groups, eval_groups = groups[groups % 2 == half], groups[groups % 2 != half]
    torch.manual_seed(seed)
    model = initialize(half)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    rng = np.random.default_rng(seed)
    folder = OUT / f"half_{half}"
    folder.mkdir(parents=True, exist_ok=True)
    history = []
    score, prediction, y = score_and_cache(model, parquet, eval_groups, scaler,
                                            folder / "pred_e0.npz")
    history.append({"epoch": 0, "score": score, "blend": blend_report(half, prediction, y)})
    print("warmup adapter half", half, "epoch 0", history[-1], flush=True)
    for epoch in range(1, epochs + 1):
        losses = []
        for offset in range(0, len(fit_groups), BATCH):
            indices = rng.permutation(fit_groups) if offset == 0 else order
            if offset == 0:
                order = indices
            batch = order[offset:offset + BATCH]
            x, target, mask = batch_arrays(parquet, batch, scaler)
            values = torch.from_numpy(x)
            target = torch.from_numpy(np.clip(target, -2, 2))
            row_weight = torch.from_numpy(np.where(mask, 1.0, 0.06).astype(np.float32))
            row_weight[:, :WARMUP] = 0
            hidden = None
            for start in range(0, N_ROWS, CHUNK):
                sl = slice(start, start + CHUNK)
                optimizer.zero_grad()
                context = model.context(values)
                output, hidden = model(values[:, sl], context, hidden)
                loss = weighted_loss(output, target[:, sl], row_weight[:, sl])
                loss.backward()
                optimizer.step()
                hidden = detach_state(hidden)
                losses.append(float(loss.detach()))
            if offset and offset % 160 == 0:
                print("warmup adapter half", half, "epoch", epoch,
                      "trained groups", offset, flush=True)
        path = folder / f"pred_e{epoch}.npz"
        score, prediction, y = score_and_cache(model, parquet, eval_groups, scaler, path)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "score": score,
               "blend": blend_report(half, prediction, y)}
        history.append(row)
        torch.save(model.state_dict(), folder / f"model_e{epoch}.pt")
        print("warmup adapter half", half, "epoch", epoch, row, flush=True)
    result = {"half": half, "epochs": epochs, "lr": lr, "seed": seed,
              "fitted_groups": fit_groups.tolist(), "evaluated_groups": eval_groups.tolist(),
              "history": history,
              "causality": "context uses normalized input rows 0..98 only"}
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def selfcheck():
    torch.manual_seed(3)
    base = {"rnn.weight_ih_l0": torch.randn(384, 112),
            "rnn.weight_hh_l0": torch.randn(384, 128),
            "rnn.bias_ih_l0": torch.randn(384), "rnn.bias_hh_l0": torch.randn(384),
            "head.weight": torch.randn(2, 128), "head.bias": torch.randn(2)}
    model = WarmupAdapter()
    with torch.no_grad():
        model.rnn.weight_ih_l0[:, :112].copy_(base["rnn.weight_ih_l0"])
        model.rnn.weight_ih_l0[:, 112:].zero_()
        model.rnn.weight_hh_l0.copy_(base["rnn.weight_hh_l0"])
        model.rnn.bias_ih_l0.copy_(base["rnn.bias_ih_l0"])
        model.rnn.bias_hh_l0.copy_(base["rnn.bias_hh_l0"])
        model.head.weight.copy_(base["head.weight"])
        model.head.bias.copy_(base["head.bias"])
    control = torch.nn.GRU(112, 128, batch_first=True)
    head = torch.nn.Linear(128, 2)
    control.load_state_dict({key.removeprefix("rnn."): value for key, value in base.items()
                             if key.startswith("rnn.")})
    head.load_state_dict({key.removeprefix("head."): value for key, value in base.items()
                          if key.startswith("head.")})
    x = torch.randn(2, 140, 112)
    context = model.context(x)
    assert torch.equal(model(x, context)[0], head(control(x)[0]))
    changed = x.clone(); changed[:, WARMUP:] += 50
    assert torch.equal(context, model.context(changed))
    print("warm-up adapter exact initialization and causal context checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--half", type=int, choices=(0, 1))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(3)
    if args.selfcheck:
        selfcheck()
    elif args.half is None:
        parser.error("--half is required")
    else:
        fit(args.half, args.epochs, args.lr, args.seed)


if __name__ == "__main__":
    main()
