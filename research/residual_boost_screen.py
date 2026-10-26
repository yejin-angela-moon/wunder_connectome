"""Cross-fitted recurrent residual booster for the strong v24-like proxy.

For each validation parity, the base prediction comes from models fine-tuned
only on the opposite parity.  A residual GRU is trained on one parity and
evaluated on the other, so neither the base prediction nor the correction has
seen an evaluated sequence.

The first screen deliberately trains only on rows selected by ``is_scored``.
The hidden state still consumes every causal input row; labels and the hidden
scoring mask are never inference inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from global_wp import GlobalWP
from maskft import detach_state, load as load_base
from maskmix import normalize, read_valid
from pilot import N_FEATURES, N_ROWS, ROOT, SEED, WARMUP, make_gru


OUT = Path(__file__).resolve().parent / "cache" / "residual_boost"
PRED = Path(__file__).resolve().parent / "cache" / "maskft"
BATCH = 8
CHUNK = 512
BLENDS = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
MEMBERS = {
    "c71": "pred_scratch128g2c71_uC_e1-2-3_h{half}.npz",
    "c72": "pred_scratch128g2c72_uC_e1-2-3_h{half}.npz",
    "split": "pred_split128_a0.75_s81_u0.06_lr0.0002_s7_tb_e2-3_h{half}.npz",
}
MEMBER_WEIGHTS = {"c71": 0.4, "c72": 0.4, "split": 0.2}


def fixed_subset(groups, count):
    groups = np.asarray(groups, dtype=int)
    if count <= 0 or count >= len(groups):
        return groups
    return groups[np.linspace(0, len(groups) - 1, count, dtype=int)]


def scored_counts(parquet, groups):
    counts = {}
    for group in groups:
        table = parquet.read_row_group(
            int(group), columns=["need_prediction", "is_scored"], use_threads=False)
        need = table["need_prediction"].to_numpy().astype(bool)
        scored = table["is_scored"].to_numpy().astype(bool)
        counts[int(group)] = int(np.sum(need & scored))
    return counts


def load_strong(parquet, fitted_half):
    """Load predictions evaluated on the parity opposite ``fitted_half``."""
    prediction = None
    target = None
    for name, filename in MEMBERS.items():
        with np.load(PRED / filename.format(half=fitted_half)) as cached:
            if target is None:
                target = cached["y"].copy()
                prediction = MEMBER_WEIGHTS[name] * cached["p"]
            else:
                if not np.array_equal(target, cached["y"]):
                    raise ValueError("strong-member target rows do not align")
                prediction += MEMBER_WEIGHTS[name] * cached["p"]
    groups = np.arange(parquet.num_row_groups)
    groups = groups[groups % 2 != fitted_half]
    counts = scored_counts(parquet, groups)
    slices, offset = {}, 0
    for group in groups:
        stop = offset + counts[int(group)]
        slices[int(group)] = slice(offset, stop)
        offset = stop
    if offset != len(target):
        raise ValueError(f"cached selected rows {len(target)} != reconstructed rows {offset}")
    return prediction.astype(np.float32), target.astype(np.float32), slices


def selected_base(group, y, mask, prediction, target, slices):
    sl = slices[int(group)]
    base = prediction[sl]
    expected = target[sl]
    actual = y[mask]
    if len(base) != int(mask.sum()) or not np.array_equal(expected, actual):
        raise ValueError(f"cached rows do not align for group {group}")
    return base


def residual_batch(parquet, groups, scaler, prediction, target, slices):
    x, y, mask = read_valid(parquet, groups)
    residual = np.zeros((*y.shape[:2], 2), np.float32)
    weight = np.zeros((*y.shape[:2], 2), np.float32)
    clipped_y = np.clip(y, -2, 2).astype(np.float32)
    for i, group in enumerate(groups):
        base = selected_base(group, y[i], mask[i], prediction, target, slices)
        residual[i, mask[i]] = clipped_y[i, mask[i]] - np.clip(base, -2, 2)
        weight[i, mask[i]] = np.abs(clipped_y[i, mask[i]])
    weight[:, :WARMUP] = 0
    return normalize(x, scaler), y, mask, residual, weight


def train_epoch(model, optimizer, parquet, groups, scaler, prediction, target, slices,
                device, rng):
    model.train()
    losses = []
    order = groups[rng.permutation(len(groups))]
    for offset in range(0, len(groups), BATCH):
        batch = order[offset:offset + BATCH]
        x, _, _, residual, weight = residual_batch(
            parquet, batch, scaler, prediction, target, slices)
        x = torch.from_numpy(x).to(device)
        residual = torch.from_numpy(residual).to(device)
        weight = torch.from_numpy(weight).to(device)
        hidden = None
        for start in range(0, N_ROWS, CHUNK):
            sl = slice(start, start + CHUNK)
            estimate, hidden = model(x[:, sl], hidden)
            total = weight[:, sl].sum()
            if total.item() > 0:
                loss = ((estimate - residual[:, sl]).square() * weight[:, sl]).sum() / total
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            hidden = detach_state(hidden)
    if not losses:
        raise ValueError("no scored rows reached residual training")
    return float(np.mean(losses))


def evaluate(model, parquet, groups, scaler, prediction, target, slices, device):
    scores = {"strong": GlobalWP(), "residual": GlobalWP()}
    scores.update({f"blend_{weight:g}": GlobalWP() for weight in BLENDS})
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(groups), BATCH):
            batch = groups[offset:offset + BATCH]
            x, y, mask = read_valid(parquet, batch)
            values = torch.from_numpy(normalize(x, scaler)).to(device)
            hidden, pieces = None, []
            for start in range(0, N_ROWS, CHUNK):
                output, hidden = model(values[:, start:start + CHUNK], hidden)
                pieces.append(output.cpu().numpy())
            correction = np.concatenate(pieces, axis=1)
            for i, group in enumerate(batch):
                base = selected_base(group, y[i], mask[i], prediction, target, slices)
                corr = correction[i, mask[i]]
                selected_y = y[i, mask[i]]
                selected = np.ones(len(base), bool)
                scores["strong"].add(selected_y, base, selected)
                scores["residual"].add(selected_y, corr, selected)
                for weight in BLENDS:
                    scores[f"blend_{weight:g}"].add(selected_y, base + weight * corr, selected)
    return {name: metric.result()["weighted_pearson"] for name, metric in scores.items()}


def selfcheck():
    assert abs(sum(MEMBER_WEIGHTS.values()) - 1) < 1e-12
    assert set(fixed_subset(np.arange(10), 4)).issubset(set(range(10)))
    torch.manual_seed(1)
    model = make_gru(8).eval()
    torch.nn.init.zeros_(model.head.weight)
    torch.nn.init.zeros_(model.head.bias)
    x = torch.randn(2, 73, N_FEATURES)
    full = model(x)[0]
    first, hidden = model(x[:, :31])
    second = model(x[:, 31:], hidden)[0]
    assert torch.equal(full, torch.zeros_like(full))
    assert torch.allclose(full, torch.cat((first, second), 1), atol=1e-7)
    print("residual booster initialization, split, and recurrent continuity checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-half", type=int, choices=(0, 1), default=0,
                        help="validation parity evaluated by a booster trained on the other parity")
    parser.add_argument("--train-groups", type=int, default=128,
                        help="fixed evenly spaced training subset; <=0 means the complete parity")
    parser.add_argument("--eval-groups", type=int, default=128,
                        help="fixed evenly spaced evaluation subset; <=0 means the complete parity")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    torch.set_num_threads(3)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    if parquet.num_row_groups != 1_873:
        raise ValueError("unexpected validation file")
    train_half = 1 - args.test_half
    train_groups = fixed_subset(np.flatnonzero(np.arange(parquet.num_row_groups) % 2 == train_half),
                                args.train_groups)
    eval_groups = fixed_subset(np.flatnonzero(np.arange(parquet.num_row_groups) % 2 == args.test_half),
                               args.eval_groups)
    if set(train_groups) & set(eval_groups):
        raise ValueError("sequence leakage between residual train and evaluation groups")

    # A parity's OOF base was fitted on the opposite parity.
    train_prediction, train_target, train_slices = load_strong(parquet, args.test_half)
    eval_prediction, eval_target, eval_slices = load_strong(parquet, train_half)
    _, scaler = load_base("v1")

    model = make_gru(args.width).to(args.device)
    torch.nn.init.zeros_(model.head.weight)
    torch.nn.init.zeros_(model.head.bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = [{"epoch": 0, "loss": None,
                "scores": evaluate(model, parquet, eval_groups, scaler, eval_prediction,
                                   eval_target, eval_slices, args.device)}]
    print("residual boost epoch 0", history[-1]["scores"], flush=True)
    for epoch in range(args.epochs):
        loss = train_epoch(model, optimizer, parquet, train_groups, scaler, train_prediction,
                           train_target, train_slices, args.device, rng)
        scores = evaluate(model, parquet, eval_groups, scaler, eval_prediction, eval_target,
                          eval_slices, args.device)
        history.append({"epoch": epoch + 1, "loss": loss, "scores": scores})
        print("residual boost epoch", epoch + 1, "loss", round(loss, 6), scores, flush=True)

    base = history[0]["scores"]["strong"]
    candidates = [(name, row["scores"][name], row["epoch"])
                  for row in history[1:]
                  for name in row["scores"] if name.startswith("blend_")]
    best = max(candidates, key=lambda item: item[1])
    result = {
        "args": vars(args),
        "train_half": train_half,
        "train_group_ids": train_groups.tolist(),
        "eval_group_ids": eval_groups.tolist(),
        "base_cache_fitted_half": train_half,
        "train_cache_fitted_half": args.test_half,
        "history": history,
        "best": {"blend": best[0], "score": best[1], "epoch": best[2],
                 "gain": best[1] - base},
        "passes_pilot_gate": best[1] >= base + 0.005,
        "methodology": "sequence-disjoint cross-fit base and residual correction; is_scored is loss-only",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"testh{args.test_half}_tr{len(train_groups)}_ev{len(eval_groups)}_s{args.seed}"
    (OUT / f"{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(model.cpu().state_dict(), OUT / f"{tag}.pt")
    print(json.dumps(result["best"], indent=2), flush=True)


if __name__ == "__main__":
    main()
