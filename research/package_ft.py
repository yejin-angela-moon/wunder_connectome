"""Export fine-tuned standardized-input GRUs into one ZIP and check the callback.

    python research/package_ft.py NAME member1.pt [member2.pt ...]

A member written as a.pt+b.pt is the weight average of those checkpoints (SWA);
a trailing @w (e.g. a.pt+b.pt@3) sets its ensemble weight, default 1.

Checks: row-wise callback (NumPy path and float16 kernel) vs batched PyTorch on one complete validation
sequence, extracted-ZIP smoke test (warm-up None, shape, reset), size limit,
and single-sequence callback timing on this host.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from features import event_features
from oos_global import read_group as read_valid_group
from pilot import CACHE, ROOT, make_gru
from stack import FULL

sys.path.insert(0, str(ROOT))
from utils import DataPoint  # noqa: E402

PACKAGES = CACHE / "packages"
SOLUTION = Path(__file__).with_name("multi_solution.py")
KERNEL_TOLERANCE = 2e-2                                                # float16 weights


def load_member(spec):
    """One checkpoint path, or several joined by '+' to average their weights (SWA).
    A trailing '@w' sets the member's ensemble weight (default 1)."""
    spec = str(spec).rsplit("@", 1)[0]
    parts = [torch.load(p, map_location="cpu", weights_only=True) for p in spec.split("+")]
    return {key: sum(p[key] for p in parts) / len(parts) for key in parts[0]}


def streams(state):
    """Stream names of a multi-stream GRU state (ensdistill.TwinGRU and SiameseGRU: i0, i1; TripleGRU: p0, v0,
    i1), else []."""
    if "shared.weight_ih_l0" in state:
        return ["i0", "i1"]
    return [name for name in ("i0", "p0", "v0", "i1") if f"{name}.weight_ih_l0" in state]   # joint input order


def stream_model(state):
    from ensdistill import SiameseGRU, TripleGRU, TwinEnc2GRU, TwinEncGRU, TwinGRU
    model = (TwinEnc2GRU if "sides.0.0.weight" in state else TwinEncGRU if "encoders.0.0.weight" in state
             else SiameseGRU if "shared.weight_ih_l0" in state else TwinGRU if "i0.weight_ih_l0" in state
             else TripleGRU)
    return model(state["rnn.weight_hh_l0"].shape[1])


def stream_members(state, inputs=112, codes=None):
    """A multi-stream GRU as separate first-layer members plus a joint second layer. Each stream reads its own
    columns of the raw inputs (zero elsewhere; `inputs` input columns in all), has a zero head and is padded
    with zero units to a multiple of 8 (they stay at zero). With group encoders (TwinEncGRU), a stream's next
    64 inputs are its instrument's group codes, at input columns codes + (0 for i0, 64 for i1); with side
    encoders (TwinEnc2GRU) its last 32 are its side codes, at codes + 128 + (0 for i0, 32 for i1). The joint
    layer's input is the streams' [h, 1] in order; its input bias rides on the last stream's constant 1."""
    from ensdistill import STREAM_COLUMNS
    names, joint_ih = streams(state), state["rnn.weight_ih_l0"]
    get = lambda name, key: state[f"{name}.{key}"] if f"{name}.{key}" in state else state[f"shared.{key}"]
    parts, blocks, offset = [], [], 0
    for k, name in enumerate(names):
        n = get(name, "weight_hh_l0").shape[1]
        width = -(-n // 8) * 8
        part = {"rnn.weight_ih_l0": torch.zeros(3 * width, inputs), "rnn.weight_hh_l0": torch.zeros(3 * width, width),
                "rnn.bias_ih_l0": torch.zeros(3 * width), "rnn.bias_hh_l0": torch.zeros(3 * width),
                "head.weight": torch.zeros(2, width), "head.bias": torch.zeros(2), "layer0": True}
        for gate in range(3):
            rows, source = slice(gate * width, gate * width + n), slice(gate * n, (gate + 1) * n)
            w_ih = get(name, "weight_ih_l0")[source]
            part["rnn.weight_ih_l0"][rows, STREAM_COLUMNS[name]] = w_ih[:, :len(STREAM_COLUMNS[name])]
            if codes is not None:
                column = len(STREAM_COLUMNS[name])
                spans = [(64, codes + (0 if name == "i0" else 64))]
                if "sides.0.0.weight" in state:
                    spans.append((32, codes + 128 + (0 if name == "i0" else 32)))
                for size, start in spans:
                    part["rnn.weight_ih_l0"][rows, start:start + size] = w_ih[:, column:column + size]
                    column += size
            part["rnn.weight_hh_l0"][rows, :n] = get(name, "weight_hh_l0")[source]
            part["rnn.bias_ih_l0"][rows] = get(name, "bias_ih_l0")[source]
            part["rnn.bias_hh_l0"][rows] = get(name, "bias_hh_l0")[source]
        parts.append(part)
        block = torch.zeros(joint_ih.shape[0], width + (k < len(names) - 1))   # + this stream's constant 1
        block[:, :n] = joint_ih[:, offset:offset + n]
        blocks.append(block)
        offset += n
    joint = {"rnn.weight_ih_l0": torch.cat(blocks, 1), "rnn.bias_ih_l0": state["rnn.bias_ih_l0"],
             "rnn.weight_hh_l0": state["rnn.weight_hh_l0"], "rnn.bias_hh_l0": state["rnn.bias_hh_l0"],
             "head.weight": state["head.weight"], "head.bias": state["head.bias"]}
    return parts, joint


def export(members, output, distribution=None, distribution_mix=0.0):
    """Group plain members by width (one shared input matmul across them); each
    split-encoder member is its own group with its encoder weights, and so is the
    second layer of a two-layer member.

    Only input columns are padded (to the largest plain input size, for
    event-feature members). Widths are not padded: official runtime follows
    matmul work."""
    originals = [load_member(m) for m in members]
    weights = np.array([float(str(m).rsplit("@", 1)[1]) if "@" in str(m) else 1.0 for m in members])
    weights = weights / weights.sum()
    split = ["instrument.0.weight" in s for s in originals]
    split_index = split.index(True) if distribution is not None and split.count(True) == 1 else None
    if distribution is not None and split_index is None:
        raise ValueError("distribution head requires exactly one split member")
    output_weights = weights.copy()
    if split_index is not None:
        output_weights[split_index] *= 1 - distribution_mix
    raw_inputs = max([112 if streams(s) else s["rnn.weight_ih_l0"].shape[1]         # stream models read the raw 112
                      for s, sp in zip(originals, split) if not sp] or [112])
    encoded = [i for i, s in enumerate(originals) if "encoders.0.0.weight" in s]   # their codes follow the inputs
    code_sizes = [192 if "sides.0.0.weight" in originals[i] else 128 for i in encoded]
    code_starts = dict(zip(encoded, raw_inputs + np.concatenate([[0], np.cumsum(code_sizes)[:-1]]).astype(int)))
    plain_inputs = raw_inputs + sum(code_sizes)
    groups, encoders = {}, []
    for i, (state, weight) in enumerate(zip(originals, output_weights)):
        state = dict(state)
        state["weight"] = torch.tensor(weight)
        if streams(state):
            codes = int(code_starts[i]) if i in encoded else None
            parts, joint = stream_members(state, plain_inputs, codes)
            if codes is not None:
                encoders.append((state, codes))
            feeds = []
            for part in parts:
                key = ("plain", part["rnn.weight_hh_l0"].shape[1])
                groups.setdefault(key, []).append(part | {"weight": state["weight"]})
                feeds.append((list(groups).index(key), len(groups[key]) - 1))
            groups[("deep", i)] = [joint | {"weight": state["weight"], "source": feeds}]
            continue
        if split[i]:
            key = ("split", i)
        else:
            state["rnn.weight_ih_l0"] = torch.nn.functional.pad(
                state["rnn.weight_ih_l0"], (0, plain_inputs - state["rnn.weight_ih_l0"].shape[1]))
            key = ("plain", state["rnn.weight_hh_l0"].shape[1])
            if "rnn.weight_ih_l1" in state:
                # two layers: layer 0 joins its width group with a zero head; layer 1 is its own group that reads
                # that member's hidden state
                head = {"head.weight": torch.zeros(2, state["rnn.weight_hh_l0"].shape[1]),
                        "head.bias": torch.zeros(2), "layer0": True}
                groups.setdefault(key, []).append(state | head)
                deep = {f"rnn.{name}_l0": state[f"rnn.{name}_l1"]
                        for name in ("weight_ih", "bias_ih", "weight_hh", "bias_hh")}
                deep |= {name: state[name] for name in ("head.weight", "head.bias", "weight")}
                deep["source"] = (list(groups).index(key), len(groups[key]) - 1)
                groups[("deep", i)] = [deep]
                continue
        groups.setdefault(key, []).append(state)
    with np.load(FULL / "gru96_norm5ep/gru_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    arrays = {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "n_groups": len(groups),
              "event_inputs": raw_inputs - 112, "n_encoders": len(encoders),
              "b_out": np.sum([w * s["head.bias"].numpy() for s, w in zip(originals, output_weights)], 0)
              .astype(np.float32)}
    for j, ((kind, _), states) in enumerate(groups.items()):
        if j == 0 and kind == "plain" and "layer0" not in states[0]:  # runtime-guard member: its own head
            arrays["fallback_out"] = np.concatenate([states[0]["head.weight"].numpy(),
                                                     states[0]["head.bias"].numpy()[:, None]], 1).astype(np.float32)
        get = lambda key: np.stack([s[key].numpy().astype(np.float32) for s in states])
        arrays[f"g{j}_w_ih"] = np.concatenate(list(get("rnn.weight_ih_l0")))
        arrays[f"g{j}_b_ih"] = get("rnn.bias_ih_l0")
        arrays[f"g{j}_w_hh"] = get("rnn.weight_hh_l0")
        arrays[f"g{j}_b_hh"] = get("rnn.bias_hh_l0")
        arrays[f"g{j}_w_out"] = np.concatenate([s["weight"].item() * s["head.weight"].numpy().astype(np.float32)
                                                for s in states], axis=1)
        if kind == "deep":
            arrays[f"g{j}_source"] = np.array(states[0]["source"])
        if kind == "split":
            for name in ("instrument.0.weight", "instrument.0.bias", "additional.0.weight", "additional.0.bias"):
                arrays[f"g{j}_{name.replace('.0.', '_')}"] = states[0][name].numpy().astype(np.float32)
            if distribution is not None:
                arrays[f"g{j}_distribution_weight"] = distribution["distribution.weight"].numpy().astype(np.float32)
                arrays[f"g{j}_distribution_bias"] = distribution["distribution.bias"].numpy().astype(np.float32)
                arrays[f"g{j}_distribution_centers"] = distribution["centers"].numpy().astype(np.float32)
                arrays[f"g{j}_distribution_scale"] = np.float32(weights[split_index] * distribution_mix)
    from ensdistill import BOOK_GROUPS, SIDE_GROUPS
    blocks = []                                                         # (state, module prefix, groups, code column)
    for state, codes in encoders:
        blocks.append((state, "encoders", [list(c) for c in BOOK_GROUPS["i0"] + BOOK_GROUPS["i1"]], codes))
        if "sides.0.0.weight" in state:
            blocks.append((state, "sides", SIDE_GROUPS["i0"] + SIDE_GROUPS["i1"], codes + 128))
    arrays["n_encoders"] = len(blocks)
    for k, (state, prefix, cols, codes) in enumerate(blocks):          # x index of the codes: column + 1 (bias)
        stack = lambda key: np.stack([state[f"{prefix}.{g}.{key}"].numpy().astype(np.float32)
                                      for g in range(len(cols))])
        arrays |= {f"e{k}_w1": stack("0.weight"), f"e{k}_b1": stack("0.bias"), f"e{k}_w2": stack("2.weight"),
                   f"e{k}_b2": stack("2.bias"), f"e{k}_slot": codes + 1, f"e{k}_cols": np.array(cols)}
    np.savez(output, **arrays)
    return originals, output_weights, mean, std, ((split_index, distribution, weights[split_index] * distribution_mix)
                                                    if distribution is not None else None)


def expected(states, weights, mean, std, raw, distribution=None):
    from distill_mps import build
    normalized = np.clip((raw - mean) / std, -10, 10).astype(np.float32)
    full = np.concatenate([normalized, event_features(raw)], axis=1)
    outputs = []
    with torch.no_grad():
        for state in states:
            if streams(state):
                model = stream_model(state).eval()
            else:
                model = build(state).eval()
            model.load_state_dict(state)
            inputs = 112 if "instrument.0.weight" in state or streams(state) \
                else state["rnn.weight_ih_l0"].shape[1]
            outputs.append(model(torch.from_numpy(full[None, :, :inputs]))[0][0].numpy())
    result = np.tensordot(weights, np.stack(outputs), axes=1)
    if distribution is not None:
        index, head, scale = distribution
        model = build(states[index]).eval()
        model.load_state_dict(states[index])
        values = torch.from_numpy(normalized[None])
        with torch.no_grad():
            i0, i1 = model.instrument(values[..., :52]), model.instrument(values[..., 52:104])
            encoded = torch.cat((i0, i1, i0 - i1, i0 * i1, model.additional(values[..., 104:112])), -1)
            hidden = model.rnn(encoded)[0]
            logits = torch.nn.functional.linear(hidden, head["distribution.weight"],
                                                 head["distribution.bias"]).reshape(1, len(raw), 2, -1)
            prediction = (logits.softmax(-1) * head["centers"]).sum(-1)[0].numpy()
        result += scale * prediction
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("members", nargs="+")
    parser.add_argument("--distribution-head")
    parser.add_argument("--distribution-mix", type=float, default=0.25)
    args = parser.parse_args()
    torch.set_num_threads(1)
    PACKAGES.mkdir(exist_ok=True)
    folder = PACKAGES / args.name
    folder.mkdir(exist_ok=True)
    distribution = load_member(args.distribution_head) if args.distribution_head else None
    states, weights, mean, std, distribution = export(
        args.members, folder / "model.npz", distribution, args.distribution_mix)
    output = PACKAGES / f"{args.name}.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(SOLUTION, "solution.py")
        archive.write(folder / "model.npz", "model.npz")
        archive.write(Path(__file__).with_name("features.py"), "features.py")
        archive.write(Path(__file__).with_name("fast_kernel.c"), "fast_kernel.c")
    size = output.stat().st_size
    if size >= 20_000_000:
        raise ValueError("ZIP exceeds 20 MB")

    with tempfile.TemporaryDirectory() as directory:
        with zipfile.ZipFile(output) as archive:
            archive.extractall(directory)
        subprocess.run(
            [sys.executable, "-c", "from types import SimpleNamespace as S; import numpy as np; "
             "from solution import PredictionModel; m=PredictionModel(); "
             "p=lambda q,s,n: S(seq_ix=q,step_in_seq=s,need_prediction=n,state=np.ones(112,np.float32)); "
             "assert m.predict(p(0,0,False)) is None; a=m.predict(p(0,1,True)); "
             "assert a.shape==(2,) and np.isfinite(a).all(); "
             "b=m.predict(p(1,0,True)); c=m.predict(p(1,1,True)); assert np.allclose(a,c), (a,c)"],
            cwd=directory, check=True, timeout=60)
        spec = importlib.util.spec_from_file_location("packaged_solution", Path(directory) / "solution.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        paths = {"numpy": (module.PredictionModel(kernel=False), module.PredictionModel(kernel=False)),
                 "kernel": (module.PredictionModel(), module.PredictionModel())}

    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    seq, need, raw, _, _ = read_valid_group(parquet, 0)
    reference = expected(states, weights, mean, std, raw, distribution)[99:]
    report = {"members": [str(m) for m in args.members], "zip_bytes": size}
    for name, (callback, guarded) in paths.items():
        if name == "kernel" and callback.kernel is None:
            report[name] = "not available"
            continue
        tolerance = 5e-5 if name == "numpy" else KERNEL_TOLERANCE
        started = time.perf_counter()
        rows = [callback.predict(DataPoint(seq, i, bool(need[i]), raw[i])) for i in range(len(raw))]
        seconds = time.perf_counter() - started
        if any(r is not None for r in rows[:99]) or any(r is None for r in rows[99:]):
            raise ValueError("warm-up handling is wrong")
        got = np.asarray(rows[99:], np.float32)
        difference = float(np.max(np.abs(got - reference)))
        if not np.isfinite(got).all() or difference > tolerance:
            raise ValueError(f"{name} callback differs from PyTorch: {difference}")

        # runtime guard: mid-sequence switch to the first member alone, then zeros
        guard_difference = None
        if guarded.stage == 0:
            half = len(raw) // 2
            for i in range(half):
                guarded.predict(DataPoint(seq, i, bool(need[i]), raw[i]))
            guarded.reduce()
            guarded.stage = 1
            alone = np.asarray([guarded.predict(DataPoint(seq, i, bool(need[i]), raw[i]))
                                for i in range(half, len(raw))])
            guard_difference = float(np.max(np.abs(alone - expected(states[:1], np.ones(1), mean, std, raw)[half:])))
            if guard_difference > tolerance:
                raise ValueError(f"{name} reduced callback differs from its member: {guard_difference}")
        guarded.stage = 2
        if guarded.predict(DataPoint(seq, 5, False, raw[5])) is not None or \
                not np.array_equal(guarded.predict(DataPoint(seq, 200, True, raw[200])), np.zeros(2)):
            raise ValueError("skip stage is wrong")
        report[name] = {"max_abs_difference_vs_torch": difference, "guard_member_difference": guard_difference,
                        "seconds_per_20k_rows_this_host": seconds}
    (folder / "check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
