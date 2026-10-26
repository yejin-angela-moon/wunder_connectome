"""Distil the fine-tuned ensemble into one GRU so its gain fits the runtime budget.

Official runtime grows with per-row compute (v10, four members padded to
width 192, took 56 minutes), so ensemble members are expensive. Here the
teacher is the average of fine-tuned members (SWA over the given epochs) for
one validation half. The student starts from a distilled base and trains on
that half's validation sequences (true scored-row weights) plus train
sequences (mask-net weights). The loss is
(1 - alpha) * label loss + alpha * imitation of the teacher, both |y|-weighted.
The other half's scored rows are scored after every epoch. `--half -1` trains
on all validation with teachers fine-tuned on all validation (no held-out
score).

    python research/ensdistill.py --half 0 --student gru128_a0.75_s58
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from features import event_features
from maskft import OUT as FT_OUT, detach_state, load
import maskmix
from maskmix import (BATCH, CHUNK, N_TRAIN, WARMUP, evaluate, fit_mask_net, mask_probability, normalize,
                     split)
from oos_global import read_group as read_valid_group
from pilot import CACHE, N_ROWS, ROOT, make_gru
from scale import read_group as read_train_group

OUT = CACHE / "ensdistill"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TEACHERS = ("v1", "gru128_a0.5_s57", "gru192_a0.5_s57", "gru128_a0.75_s58",
            "gru192ev_a0.5_s59", "gru192_a0.75_s60", "gru128_a0.75_s61")


class PLEGRU(torch.nn.Module):
    """GRU after a learned per-feature piecewise-linear calibration of the standardized inputs,
    z_j = x_j + sum_k slope_jk relu(x_j - knot_jk); slopes start at zero, so training starts from a plain GRU."""

    def __init__(self, width, knots):
        super().__init__()
        self.register_buffer("knots", knots)                           # (112, K) per-feature quantiles
        self.slope = torch.nn.Parameter(torch.zeros_like(knots))
        self.rnn = torch.nn.GRU(112, width, batch_first=True)
        self.head = torch.nn.Linear(width, 2)

    def forward(self, values, hidden=None):
        z = values
        for k in range(self.knots.shape[1]):
            z = z + torch.relu(values - self.knots[:, k]) * self.slope[:, k]
        values, hidden = self.rnn(z, hidden)
        return self.head(values), hidden


class TwinGRU(torch.nn.Module):
    """Separate recurrent states for i0 (columns 0-51) and i1 (52-103), each also reading a0-a7 (104-111),
    then a joint GRU over both states."""

    def __init__(self, width):
        super().__init__()
        self.i0 = torch.nn.GRU(60, width, batch_first=True)
        self.i1 = torch.nn.GRU(60, width // 2, batch_first=True)
        self.rnn = torch.nn.GRU(width + width // 2, width, batch_first=True)
        self.head = torch.nn.Linear(width, 2)

    def forward(self, values, hidden=None):
        h0, h1, h = hidden if hidden is not None else (None, None, None)
        extra = values[..., 104:112]
        o0, h0 = self.i0(torch.cat((values[..., :52], extra), -1), h0)
        o1, h1 = self.i1(torch.cat((values[..., 52:104], extra), -1), h1)
        values, h = self.rnn(torch.cat((o0, o1), -1), h)
        return self.head(values), (h0, h1, h)


A_COLUMNS = list(range(104, 112))
STREAM_COLUMNS = {"i0": list(range(52)) + A_COLUMNS, "i1": list(range(52, 104)) + A_COLUMNS,
                  "p0": list(range(22)) + list(range(44, 48)) + A_COLUMNS,       # i0 book and trade prices
                  "v0": list(range(22, 44)) + list(range(48, 52)) + A_COLUMNS}   # i0 book and trade volumes


class TripleGRU(torch.nn.Module):
    """Separate recurrent states for i0 prices, i0 volumes and i1 (each also reading a0-a7), then a joint GRU
    over the three states. Widths: 2/3, 2/3 and 1/2 of the joint width."""

    def __init__(self, width):
        super().__init__()
        self.p0 = torch.nn.GRU(len(STREAM_COLUMNS["p0"]), 2 * width // 3, batch_first=True)
        self.v0 = torch.nn.GRU(len(STREAM_COLUMNS["v0"]), 2 * width // 3, batch_first=True)
        self.i1 = torch.nn.GRU(len(STREAM_COLUMNS["i1"]), width // 2, batch_first=True)
        self.rnn = torch.nn.GRU(2 * (2 * width // 3) + width // 2, width, batch_first=True)
        self.head = torch.nn.Linear(width, 2)

    def forward(self, values, hidden=None):
        hp, hv, hi, h = hidden if hidden is not None else (None, None, None, None)
        op, hp = self.p0(values[..., STREAM_COLUMNS["p0"]], hp)
        ov, hv = self.v0(values[..., STREAM_COLUMNS["v0"]], hv)
        oi, hi = self.i1(values[..., STREAM_COLUMNS["i1"]], hi)
        values, h = self.rnn(torch.cat((op, ov, oi), -1), h)
        return self.head(values), (hp, hv, hi, h)


class SiameseGRU(torch.nn.Module):
    """One GRU, shared by i0 and i1 (each with a0-a7; two separate states), then a joint GRU over both
    states. Stream width 2/3 of the joint width."""

    def __init__(self, width):
        super().__init__()
        self.shared = torch.nn.GRU(len(STREAM_COLUMNS["i0"]), 2 * width // 3, batch_first=True)
        self.rnn = torch.nn.GRU(2 * (2 * width // 3), width, batch_first=True)
        self.head = torch.nn.Linear(width, 2)

    def forward(self, values, hidden=None):
        h0, h1, h = hidden if hidden is not None else (None, None, None)
        o0, h0 = self.shared(values[..., STREAM_COLUMNS["i0"]], h0)
        o1, h1 = self.shared(values[..., STREAM_COLUMNS["i1"]], h1)
        values, h = self.rnn(torch.cat((o0, o1), -1), h)
        return self.head(values), (h0, h1, h)


BOOK_GROUPS = {"i0": [range(0, 11), range(11, 22), range(22, 33), range(33, 44)],
               "i1": [range(52, 63), range(63, 74), range(74, 85), range(85, 96)]}


class TwinEncGRU(TwinGRU):
    """Twin GRU whose streams also read a learned nonlinear encoding of each order-book group (bid/ask price
    and volume levels, 11 columns each): a small per-group MLP 11 -> 32 -> 16 that can model relations among
    the group's columns, which a GRU sees only through its linear input layer."""

    def __init__(self, width):
        torch.nn.Module.__init__(self)
        self.encoders = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Linear(11, 32), torch.nn.Tanh(), torch.nn.Linear(32, 16)) for _ in range(8))
        self.i0 = torch.nn.GRU(60 + 64, width, batch_first=True)
        self.i1 = torch.nn.GRU(60 + 64, width // 2, batch_first=True)
        self.rnn = torch.nn.GRU(width + width // 2, width, batch_first=True)
        self.head = torch.nn.Linear(width, 2)

    def forward(self, values, hidden=None):
        h0, h1, h = hidden if hidden is not None else (None, None, None)
        codes = [enc(values[..., list(cols)]) for enc, cols in zip(self.encoders, BOOK_GROUPS["i0"] + BOOK_GROUPS["i1"])]
        extra = values[..., 104:112]
        o0, h0 = self.i0(torch.cat([values[..., :52], extra] + codes[:4], -1), h0)
        o1, h1 = self.i1(torch.cat([values[..., 52:104], extra] + codes[4:], -1), h1)
        values, h = self.rnn(torch.cat((o0, o1), -1), h)
        return self.head(values), (h0, h1, h)


SIDE_GROUPS = {"i0": [list(range(0, 11)) + list(range(22, 33)), list(range(11, 22)) + list(range(33, 44))],
               "i1": [list(range(52, 63)) + list(range(74, 85)), list(range(63, 74)) + list(range(85, 96))]}


class TwinEnc2GRU(TwinEncGRU):
    """TwinEncGRU plus side encoders: each book side's 11 price and 11 volume levels together (22 columns)
    through an MLP 22 -> 48 -> 16, so a price level and its volume can interact before the GRU. Stream input:
    raw (60), group codes (64), side codes (32)."""

    def __init__(self, width):
        torch.nn.Module.__init__(self)
        self.encoders = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Linear(11, 32), torch.nn.Tanh(), torch.nn.Linear(32, 16)) for _ in range(8))
        self.sides = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Linear(22, 48), torch.nn.Tanh(), torch.nn.Linear(48, 16)) for _ in range(4))
        self.i0 = torch.nn.GRU(60 + 64 + 32, width, batch_first=True)
        self.i1 = torch.nn.GRU(60 + 64 + 32, width // 2, batch_first=True)
        self.rnn = torch.nn.GRU(width + width // 2, width, batch_first=True)
        self.head = torch.nn.Linear(width, 2)

    def forward(self, values, hidden=None):
        h0, h1, h = hidden if hidden is not None else (None, None, None)
        codes = [enc(values[..., list(cols)]) for enc, cols in zip(self.encoders, BOOK_GROUPS["i0"] + BOOK_GROUPS["i1"])]
        sides = [enc(values[..., cols]) for enc, cols in zip(self.sides, SIDE_GROUPS["i0"] + SIDE_GROUPS["i1"])]
        extra = values[..., 104:112]
        o0, h0 = self.i0(torch.cat([values[..., :52], extra] + codes[:4] + sides[:2], -1), h0)
        o1, h1 = self.i1(torch.cat([values[..., 52:104], extra] + codes[4:] + sides[2:], -1), h1)
        values, h = self.rnn(torch.cat((o0, o1), -1), h)
        return self.head(values), (h0, h1, h)


def causal_features(x, half_life):
    """Each standardized feature's deviation from its causal EWMA mean, in units of the causal EWMA
    deviation (prior mean 0, variance 1), clipped like the inputs. x: (sequences, rows, features)."""
    from scipy.signal import lfilter
    decay = 0.5 ** (1 / half_life)
    mean = lfilter([1 - decay], [1, -decay], x, axis=1)
    deviation = x - np.concatenate([np.zeros_like(mean[:, :1]), mean[:, :-1]], 1)
    variance = lfilter([1 - decay], [1, -decay], deviation ** 2, axis=1,
                       zi=np.full((x.shape[0], 1, x.shape[2]), decay))[0]
    previous = np.concatenate([np.ones_like(variance[:, :1]), variance[:, :-1]], 1)
    return np.clip(deviation / np.sqrt(previous + 1e-4), -10, 10).astype(np.float32)


PLE_QUANTILES = (0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98)
PLE_KNOTS = len(PLE_QUANTILES)


def make_student(cell, width, layers=1, dropout=0.0, inputs=112, knots=None):
    if cell == "gru":
        return make_gru(width, input_size=inputs, num_layers=layers, dropout=dropout)
    if cell == "ple":
        return PLEGRU(width, knots if knots is not None else torch.zeros(112, PLE_KNOTS))
    if cell == "twin":
        return TwinGRU(width)
    if cell == "triple":
        return TripleGRU(width)
    if cell == "siamese":
        return SiameseGRU(width)
    if cell == "twinenc":
        return TwinEncGRU(width)
    if cell == "twinenc2":
        return TwinEnc2GRU(width)
    if cell == "warmupgru":
        if layers != 1 or dropout:
            raise ValueError("warmup GRU currently supports one recurrent layer without dropout")
        from warmup_model import WarmupGRU
        return WarmupGRU(width)
    if cell == "mingru":
        from mingru import MinGRUStack
        return MinGRUStack(width, layers)

    class LSTM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.LSTM(112, width, num_layers=layers, batch_first=True)
            self.head = torch.nn.Linear(width, 2)

        def forward(self, values, hidden=None):
            values, hidden = self.rnn(values, hidden)
            return self.head(values), hidden

    return LSTM()


def prep(raw, scaler):
    x = np.clip((raw - scaler[0]) / scaler[1], -10, 10).astype(np.float32)
    if scaler[2]:
        x = np.concatenate([x, np.stack([event_features(r) for r in raw])], axis=2)
    return x


def load_teachers(half, swa):
    teachers = []
    for base in TEACHERS:
        model, scaler = load(base)
        split_tag = f"h{half}" if half >= 0 else "all"
        states = [torch.load(FT_OUT / f"{base}_u0.06_lr0.0002_s7_{split_tag}_e{e}.pt", map_location="cpu",
                             weights_only=True) for e in swa]
        model.load_state_dict({k: sum(s[k] for s in states) / len(states) for k in states[0]})
        teachers.append((model.to(DEVICE).eval(), scaler, 1 / len(TEACHERS)))
    return teachers


def load_gen2_extra(half):
    """Second-generation teacher = 0.5 * mean(gen-1 members, target-balanced split encoder,
    distilled GRU128) + 0.5 * mean(scratch students). Returns the models added to the
    cached gen-1 average, with weights, and the gen-1 average's weight."""
    tag = f"h{half}" if half >= 0 else "all"
    seeds = (71, 72) if half >= 0 else (71, 72, 73, 74)
    n_ft = len(TEACHERS) + 2
    specs = [("split128_a0.75_s81", [FT_OUT / f"split128_a0.75_s81_u0.06_lr0.0002_s7_tb_{tag}_e{e}.pt"
                                     for e in (2, 3)], 0.5 / n_ft),
             ("gru128_a0.75_s58", [OUT / f"gru128_a0.75_s58_a0.5_t1872_lr0.0002_s7_{tag}_e{e}.pt"
                                   for e in (1, 2, 3, 4)], 0.5 / n_ft)]
    specs += [(None, [OUT / f"scratch128_a0.75_t6000_p6000_lr0.001_s{seed}_{tag}_e{e}.pt" for e in (3, 4, 5)],
               0.5 / len(seeds)) for seed in seeds]
    extra = []
    for base, paths, weight in specs:
        model, scaler = load(base or "v1")
        if base is None:
            model = make_gru(128)
        states = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
        model.load_state_dict({k: sum(s[k] for s in states) / len(states) for k in states[0]})
        extra.append((model.to(DEVICE).eval(), scaler, weight))
    return extra, 0.5 * len(TEACHERS) / n_ft


def load_gen3_extra(half):
    """Third-generation teacher = 0.5 * gen-2 teacher + 0.5 * mean(twin-family students): per half twin s71,
    twin s72 and twin-encoder s71; for all validation twin-encoders s71-75 and twins s71-75 (each SWA over the
    three continuation epochs). Returns the models added to the cached gen-2 teacher and its weight."""
    tag = f"h{half}" if half >= 0 else "all"
    if half >= 0:
        specs = [(TwinGRU, "scratchtwin96", 71), (TwinGRU, "scratchtwin96", 72), (TwinEncGRU, "scratchtwinenc96", 71)]
    else:
        specs = ([(TwinEncGRU, "scratchtwinenc96", s) for s in (71, 72, 73, 74, 75)]
                 + [(TwinGRU, "scratchtwin96", s) for s in (71, 72, 73, 74, 75)])
    _, scaler = load("v1")
    extra = []
    for cls, name, seed in specs:
        states = [torch.load(OUT / f"{name}g2_a0.75_t6000_p6000_lr0.0002_s{seed}_cont_{tag}_e{e}.pt",
                             map_location="cpu", weights_only=True) for e in (1, 2, 3)]
        model = cls(96)
        model.load_state_dict({k: sum(st[k] for st in states) / len(states) for k in states[0]})
        extra.append((model.to(DEVICE).eval(), scaler, 0.5 / len(specs)))
    return extra, 0.5


def teacher_predict(teachers, raw):
    with torch.no_grad():
        return sum(w * m(torch.from_numpy(prep(raw, s)).to(DEVICE))[0].cpu().numpy()
                   for m, s, w in teachers)


BASE_POOL = 2_000


def cache_predictions(teachers, groups, parquet, path, from_valid, base=None, base_weight=0.0):
    out = np.lib.format.open_memmap(path.with_suffix(".tmp.npy"), mode="w+", dtype=np.float16,
                                   shape=(len(groups), N_ROWS, 2))
    for offset in range(0, len(groups), 16):
        chunk = groups[offset:offset + 16]
        raw = np.stack([(read_valid_group(parquet, int(g))[2] if from_valid
                         else read_train_group(parquet, int(g))[0]) for g in chunk])
        prediction = teacher_predict(teachers, raw)
        if base is not None:
            prediction += base_weight * np.asarray(base[offset:offset + len(chunk)], np.float32)
        out[offset:offset + len(chunk)] = prediction
    out.flush()
    del out
    path.with_suffix(".tmp.npy").replace(path)
    print("cached teacher", path.name, len(groups), flush=True)


def teacher_cache(half, swa, valid, train, fit_groups, pool_size=BASE_POOL, gen=1):
    """Teacher predictions (float16) for the fitting half and a fixed train pool, computed once.

    The pool is the original 2,000 sequences plus, for larger pools, extra
    sequences drawn from the rest, so cached predictions are reused. The
    second-generation teacher adds models to the cached gen-1 average."""
    base = np.sort(np.random.default_rng(123).choice(N_TRAIN, BASE_POOL, replace=False))
    extra = np.random.default_rng(124).choice(np.setdiff1d(np.arange(N_TRAIN), base),
                                              min(pool_size, 6_000) - BASE_POOL, replace=False)
    rest = np.setdiff1d(np.arange(N_TRAIN), np.concatenate([base, extra]))[:max(0, pool_size - 6_000)]
    pool = np.concatenate([base, np.sort(extra), rest])
    tag = f"h{half}" if half >= 0 else "all"
    prefix = "teacher" if gen == 1 else f"teacher{gen}"
    paths = {"v": OUT / f"{prefix}_{tag}_v.npy", "t": OUT / f"{prefix}_{tag}_t.npy",
             "x": OUT / f"{prefix}_{tag}_x{len(extra)}.npy", "y": OUT / f"{prefix}_{tag}_y{len(rest)}.npy"}
    todo = [(k, g, p, k == "v") for k, g, p in (("v", fit_groups, valid), ("t", base, train),
                                                 ("x", np.sort(extra), train), ("y", rest, train))
            if len(g) and not paths[k].exists()]
    if todo and gen == 1:
        teachers = load_teachers(half, swa)
        for key, groups, parquet, from_valid in todo:
            cache_predictions(teachers, groups, parquet, paths[key], from_valid)
        del teachers
    elif todo:                                                         # generation g adds models to g - 1
        _, previous, _ = teacher_cache(half, swa, valid, train, fit_groups, pool_size, gen - 1)
        teachers, base_weight = (load_gen2_extra if gen == 2 else load_gen3_extra)(half)
        for key, groups, parquet, from_valid in todo:
            cache_predictions(teachers, groups, parquet, paths[key], from_valid, previous[key], base_weight)
        del teachers
    arrays = {k: np.load(p, mmap_mode="r") for k, p in paths.items() if p.exists()}
    index = {"v": {g: ("v", i) for i, g in enumerate(fit_groups.tolist())},
             "t": {g: ("t", i) for i, g in enumerate(base.tolist())}}
    index["t"].update({g: ("x", i) for i, g in enumerate(np.sort(extra).tolist())})
    index["t"].update({g: ("y", i) for i, g in enumerate(rest.tolist())})
    return index, arrays, pool


def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--half", type=int, default=0, choices=(-1, 0, 1))
    parser.add_argument("--student", default="gru128_a0.75_s58")
    parser.add_argument("--swa", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--train-per-epoch", type=int, default=1_872)
    parser.add_argument("--unscored", type=float, default=0.06)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pool", type=int, default=BASE_POOL, help="train sequences with cached teacher")
    parser.add_argument("--scratch", type=int, default=0, help="train a fresh GRU of this width instead")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--gen", type=int, default=1, choices=(1, 2, 3), help="teacher generation")
    parser.add_argument("--init", nargs="*", default=[], help="start from the average of these checkpoints")
    parser.add_argument("--valid-repeat", type=int, default=1, help="validation sequences per epoch, times")
    parser.add_argument("--suffix", default="", help="appended to the output tag")
    parser.add_argument("--epoch-lrs", type=float, nargs="*", default=[], help="per-epoch learning rates")
    parser.add_argument("--ema", type=float, nargs="*", default=[], help="also keep weight EMAs with these decays")
    parser.add_argument("--cell", choices=("gru", "warmupgru", "lstm", "mingru", "ple", "twin", "triple", "siamese", "twinenc", "twinenc2"), default="gru")
    parser.add_argument("--causal", type=float, default=0.0,
                        help="append causal EWMA z-scores of every input with this half-life in rows")
    parser.add_argument("--layers", type=int, default=1, help="recurrent layers for --scratch")
    parser.add_argument("--chunk", type=int, default=CHUNK, help="truncated-BPTT length")
    parser.add_argument("--batch", type=int, default=BATCH // 2, help="sequences per update")
    parser.add_argument("--adamw", type=float, default=0.0, help="use AdamW with this weight decay")
    parser.add_argument("--onecycle", action="store_true", help="per-step OneCycle cosine schedule, 10%% warm-up")
    parser.add_argument("--clip", type=float, default=0.0, help="gradient-norm clip")
    parser.add_argument("--dropout", type=float, default=0.0, help="between recurrent layers")
    parser.add_argument("--chrono", type=float, default=0.0, help="chrono-initialize GRU update gates up to this many rows")
    parser.add_argument("--random-offset", action="store_true", help="random first truncated-BPTT boundary per batch")
    parser.add_argument("--alpha-train", type=float, default=-1.0, help="teacher weight on train rows (default: --alpha)")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        torch.manual_seed(1)
        model = make_student("lstm", 4).eval()
        values = torch.randn(2, 110, 112)
        full = model(values)[0]
        first, state = model(values[:, :103])
        second, _ = model(values[:, 103:], state)
        assert torch.allclose(full, torch.cat((first, second), 1), atol=1e-7)
        warmup = make_student("warmupgru", 4).eval()
        full = warmup(values)[0]
        first, state = warmup(values[:, :103])
        second, _ = warmup(values[:, 103:], state)
        assert torch.allclose(full, torch.cat((first, second), 1), atol=1e-6)
        changed = values.clone(); changed[:, WARMUP:] += 20
        assert torch.equal(warmup.context(values), warmup.context(changed))
        for cell in ("ple", "twin", "triple", "siamese", "twinenc", "twinenc2"):
            model = make_student(cell, 8, knots=torch.randn(112, PLE_KNOTS)).eval()
            if cell == "ple":
                torch.nn.init.normal_(model.slope)
            full = model(values)[0]
            first, state = model(values[:, :103])
            second, _ = model(values[:, 103:], state)
            assert torch.allclose(full, torch.cat((first, second), 1), atol=1e-6), cell
        z = values.numpy()
        changed = z.copy(); changed[:, 60:] += 5
        assert np.array_equal(causal_features(z, 10)[:, :60], causal_features(changed, 10)[:, :60])
        print("LSTM, warm-up, PLE, twin, triple, siamese, twin-encoder and side-encoder GRU full/chunk inference agree; warm-up context and causal z-scores are causal")
        return
    DEVICE = maskmix.DEVICE = args.device
    if args.threads:
        torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    valid = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    train = pq.ParquetFile(ROOT / "datasets/train.parquet")
    fit_groups, eval_groups = split(args.half)
    student, scaler = load(args.student if not args.scratch else "v1")
    if args.causal:                                                   # training and evaluate() both see them
        global normalize
        plain = normalize
        normalize = maskmix.normalize = lambda x, s: (lambda z: np.concatenate(
            [z, causal_features(z, args.causal)], axis=2))(plain(x, s))
    knots = None
    if args.cell == "ple":
        sample = np.stack([read_train_group(train, int(g))[0] for g in
                           np.random.default_rng(5).choice(N_TRAIN, 32, replace=False)])
        sample = np.clip((sample - scaler[0]) / scaler[1], -10, 10)
        knots = torch.from_numpy(np.quantile(sample.reshape(-1, 112), PLE_QUANTILES, axis=0).T.astype(np.float32))
    if args.scratch:
        torch.manual_seed(args.seed)
        student = make_student(args.cell, args.scratch, args.layers, args.dropout,
                               224 if args.causal else 112, knots)
        args.student = (f"scratch{'' if args.cell == 'gru' else args.cell}{args.scratch}"
                        f"{f'L{args.layers}' if args.layers > 1 else ''}")
    if args.init:
        states = [torch.load(path, map_location="cpu", weights_only=True) for path in args.init]
        student.load_state_dict({k: sum(s[k] for s in states) / len(states) for k in states[0]})
    if args.epoch_lrs:
        args.epochs = len(args.epoch_lrs)
    split_tag = f"h{args.half}" if args.half >= 0 else "all"
    net = fit_mask_net(valid, fit_groups, scaler, CACHE / "maskmix" / f"masknet_{split_tag}.pt")
    index, cached, pool = teacher_cache(args.half, args.swa, valid, train, fit_groups, args.pool, args.gen)
    tag = (f"{args.student}{f'g{args.gen}' if args.gen > 1 else ''}_a{args.alpha:g}_t{args.train_per_epoch}_p{args.pool}_lr{args.lr:g}"
           f"_s{args.seed}{f'_c{args.chunk}' if args.chunk != CHUNK else ''}{f'_b{args.batch}' if args.batch != BATCH // 2 else ''}{'_ema' if args.ema else ''}{'_cont' if args.init else ''}"
           f"{f'_aw{args.adamw:g}' if args.adamw else ''}{'_oc' if args.onecycle else ''}"
           f"{f'_cl{args.clip:g}' if args.clip else ''}{f'_do{args.dropout:g}' if args.dropout else ''}"
           f"{f'_chr{args.chrono:g}' if args.chrono else ''}{'_ro' if args.random_offset else ''}"
           f"{f'_at{args.alpha_train:g}' if args.alpha_train >= 0 else ''}{f'_cz{args.causal:g}' if args.causal else ''}"
           f"{args.suffix}_{split_tag}")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.chrono and hasattr(student, "rnn") and isinstance(student.rnn, torch.nn.GRU):
        # chrono initialization (Tallec & Ollivier 2018): update-gate biases log U[1, T-1], so the keep gate
        # sigmoid(b) starts spread over memory horizons up to T rows; gate order in torch GRU is (r, z, n)
        g = torch.Generator().manual_seed(args.seed)
        with torch.no_grad():
            for layer in range(student.rnn.num_layers):
                hidden_size = student.rnn.hidden_size
                getattr(student.rnn, f"bias_ih_l{layer}")[hidden_size:2 * hidden_size] = 0
                u = 1 + torch.rand(hidden_size, generator=g) * (args.chrono - 2)
                getattr(student.rnn, f"bias_hh_l{layer}")[hidden_size:2 * hidden_size] = torch.log(u)
    student = student.to(DEVICE).train()
    optimizer = (torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.adamw) if args.adamw
                 else torch.optim.Adam(student.parameters(), lr=args.lr))
    scheduler = None
    if args.onecycle:
        per_epoch = -(-(len(fit_groups) * args.valid_repeat + min(args.train_per_epoch, len(pool))) // args.batch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, pct_start=0.1,
                                                        total_steps=args.epochs * per_epoch * -(-N_ROWS // args.chunk))
    names, params = zip(*student.named_parameters())
    emas = {d: [p.detach().clone() for p in params] for d in args.ema}
    history, ema_history = [], {d: [] for d in args.ema}
    if eval_groups is not None:
        history.append(evaluate(student, valid, eval_groups, scaler))
        print(tag, "epoch 0 held-out scored WP", round(history[-1], 6), flush=True)
    for epoch in range(args.epochs):
        if scheduler is not None:
            pass
        elif args.epoch_lrs:
            for group in optimizer.param_groups:
                group["lr"] = args.epoch_lrs[epoch]
        elif args.scratch and epoch == 3:                              # distill_mps.py schedule
            for group in optimizer.param_groups:
                group["lr"] = args.lr * 0.3
        started, losses = time.time(), []
        items = [("v", g) for g in fit_groups] * args.valid_repeat
        items += [("t", g) for g in rng.choice(pool, min(args.train_per_epoch, len(pool)), replace=False)]
        items = [items[i] for i in rng.permutation(len(items))]
        for offset in range(0, len(items), args.batch):
            batch = items[offset:offset + args.batch]
            raws, ys, masks = [], [], []
            for source, group in batch:
                if source == "v":
                    _, _, x, y, mask = read_valid_group(valid, int(group))
                    masks.append(np.where(mask, 1.0, args.unscored))
                else:
                    x, y = read_train_group(train, int(group))
                    masks.append(None)
                raws.append(x)
                ys.append(y)
            raw = np.stack(raws)
            x = normalize(raw, scaler)
            if any(m is None for m in masks):
                q = mask_probability(net, x[..., :112])                   # the mask net reads the plain inputs
                masks = [m if m is not None else q[i] + args.unscored * (1 - q[i]) for i, m in enumerate(masks)]
            t = torch.from_numpy(np.stack([np.asarray(cached[index[src][int(g)][0]][index[src][int(g)][1]],
                                                      np.float32) for src, g in batch])).to(DEVICE)
            y = torch.from_numpy(np.clip(np.stack(ys), -2, 2)).to(DEVICE)
            m = np.stack(masks).astype(np.float32)
            m[:, :WARMUP] = 0
            w = torch.from_numpy(m).to(DEVICE)[..., None] * y.abs()
            x = torch.from_numpy(x).to(DEVICE)
            hidden = None
            first = int(rng.integers(args.chunk // 4, args.chunk + 1)) if args.random_offset else args.chunk
            starts = [0] + list(range(first, N_ROWS, args.chunk))
            alpha_rows = None
            if args.alpha_train >= 0:
                alpha_rows = torch.tensor([args.alpha_train if src == "t" else args.alpha for src, _ in batch],
                                          dtype=torch.float32, device=DEVICE)[:, None, None]
            for i, start in enumerate(starts):
                sl = slice(start, starts[i + 1] if i + 1 < len(starts) else N_ROWS)
                optimizer.zero_grad()
                if hasattr(student, "forward_context"):
                    context = student.context(x)
                    prediction, hidden = student.forward_context(x[:, sl], context, hidden)
                else:
                    prediction, hidden = student(x[:, sl], hidden)
                total = w[:, sl].sum().clamp_min(1e-6)
                label = ((prediction - y[:, sl]).square() * w[:, sl]).sum() / total
                mimic = ((prediction - t[:, sl]).square() * w[:, sl]).sum() / total
                if alpha_rows is None:
                    loss = (1 - args.alpha) * label + args.alpha * mimic
                else:                                                  # per-source teacher weight
                    err = (1 - alpha_rows) * (prediction - y[:, sl]).square() + alpha_rows * (prediction - t[:, sl]).square()
                    loss = (err * w[:, sl]).sum() / total
                loss.backward()
                if args.clip:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                with torch.no_grad():
                    for decay, shadow in emas.items():
                        torch._foreach_lerp_(shadow, params, 1 - decay)
                hidden = detach_state(hidden)
                losses.append(loss.item())
        for decay, shadow in emas.items():
            state = {k: v.cpu() for k, v in zip(names, shadow)}
            torch.save(state, OUT / f"{tag}_ema{decay:g}_e{epoch + 1}.pt")
            if eval_groups is not None:
                averaged = copy.deepcopy(student)
                averaged.load_state_dict(state)
                ema_history[decay].append(evaluate(averaged, valid, eval_groups, scaler))
                print(tag, "epoch", epoch + 1, f"EMA {decay:g} held-out scored WP",
                      round(ema_history[decay][-1], 6), flush=True)
        torch.save({k: v.cpu() for k, v in student.state_dict().items()}, OUT / f"{tag}_e{epoch + 1}.pt")
        if eval_groups is not None:
            history.append(evaluate(student, valid, eval_groups, scaler))
        print(tag, "epoch", epoch + 1, "loss", round(float(np.mean(losses)), 6), "held-out scored WP",
              round(history[-1], 6) if history else None, f"{time.time() - started:.0f}s", flush=True)
    (OUT / f"{tag}.json").write_text(json.dumps({"args": vars(args), "history": history,
                                                    "ema_history": {f"{d:g}": h for d, h in ema_history.items()}}, indent=2) + "\n")


if __name__ == "__main__":
    main()
