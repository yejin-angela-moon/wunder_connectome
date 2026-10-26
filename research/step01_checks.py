"""Step 0/1 checks on the per-half v24-like set (C71, C72 0.4 each; split encoder 0.2).

0a  fp16 round trip of all weights (would a half-precision kernel be safe?)
0b  truncated SVD of the two students' stacked input matrices (runtime insurance)
1   WiSE-FT: split encoder and GRU192 teacher member at theta = base + a (fine-tuned - base),
    scored on the held-out validation half and on DEV (train-period sequences the bases never saw).

    python research/step01_checks.py
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import torch

from global_wp import GlobalWP
from maskft import BATCH, batch_arrays, load
from pilot import CACHE, ROOT, WARMUP, make_gru
from scale import DEV, read_group as read_train_group

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = CACHE / "step01"
ED, FT = CACHE / "ensdistill", CACHE / "maskft"
valid = pq.ParquetFile(ROOT / "datasets/valid.parquet")
train = pq.ParquetFile(ROOT / "datasets/train.parquet")
DEV_GROUPS = DEV[::4]                                                   # 266 train-period sequences


def mean_state(paths):
    states = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    return {k: sum(s[k] for s in states) / len(states) for k in states[0]}


def predict_valid(model, scaler, groups):
    model = model.to(DEVICE).eval()
    ps, ys = [], []
    with torch.no_grad():
        for offset in range(0, len(groups), BATCH):
            x, y, m = batch_arrays(valid, groups[offset:offset + BATCH], scaler)
            p = model(torch.from_numpy(x).to(DEVICE))[0].cpu().numpy()
            ps.append(p[m]); ys.append(y[m])
    return np.concatenate(ps).astype(np.float64), np.concatenate(ys)


def predict_dev(model, scaler):
    model = model.to(DEVICE).eval()
    acc = GlobalWP()
    with torch.no_grad():
        for offset in range(0, len(DEV_GROUPS), 16):
            data = [read_train_group(train, int(g)) for g in DEV_GROUPS[offset:offset + 16]]
            x = np.stack([np.clip((d[0] - scaler[0]) / scaler[1], -10, 10) for d in data]).astype(np.float32)
            p = model(torch.from_numpy(x).to(DEVICE))[0].cpu().numpy()
            for (_, y), pi in zip(data, p):
                need = np.arange(len(y)) >= WARMUP
                acc.add(y, pi, need)
    return acc.result()["weighted_pearson"]


def wp(y, p):
    a = GlobalWP(); a.add(y, p.astype(np.float32), np.ones(len(y), bool)); return a.result()["weighted_pearson"]


def student(seed, h):
    _, scaler = load("v1")
    model = make_gru(128)
    model.load_state_dict(mean_state([ED / f"scratch128g2_a0.75_t6000_p6000_lr0.0002_s{seed}_cont_h{h}_e{e}.pt" for e in (1, 2, 3)]))
    return model, scaler


results = {}
for h in (0, 1):
    groups = np.arange(1873); groups = groups[groups % 2 != h]
    r = results[f"h{h}"] = {}
    c71, sc = student(71, h); c72, _ = student(72, h)
    split, ssc = load("split128_a0.75_s81")
    split_base = {k: v.clone() for k, v in split.state_dict().items()}
    split_ft = mean_state([FT / f"split128_a0.75_s81_u0.06_lr0.0002_s7_tb_h{h}_e{e}.pt" for e in (2, 3)])
    split.load_state_dict(split_ft)
    P = {}
    P["C71"], y = predict_valid(c71, sc, groups)
    P["C72"], _ = predict_valid(c72, sc, groups)
    P["split"], _ = predict_valid(split, ssc, groups)
    base_set = 0.4 * P["C71"] + 0.4 * P["C72"] + 0.2 * P["split"]
    r["base_set"] = wp(y, base_set)
    print(f"h{h} base set {r['base_set']:.6f}", flush=True)

    # 0a fp16 round trip
    q = {}
    for name, model, scaler in (("C71", c71, sc), ("C72", c72, sc), ("split", split, ssc)):
        clone = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.half().float() for k, v in clone.items()})
        q[name], _ = predict_valid(model, scaler, groups)
        model.load_state_dict(clone)
    r["fp16_set"] = wp(y, 0.4 * q["C71"] + 0.4 * q["C72"] + 0.2 * q["split"])
    r["fp16_max_abs_pred_diff"] = float(max(np.abs(q[k] - P[k]).max() for k in q))
    print(f"h{h} fp16 set {r['fp16_set']:.6f} (max |dp| {r['fp16_max_abs_pred_diff']:.2e})", flush=True)

    # 0b truncated SVD of the stacked student input matrices
    w71 = c71.state_dict()["rnn.weight_ih_l0"].detach().cpu().clone()
    w72 = c72.state_dict()["rnn.weight_ih_l0"].detach().cpu().clone()
    stacked = torch.cat([w71, w72]).double()                             # SVD on CPU (MPS has no float64)
    U, S, Vt = torch.linalg.svd(stacked, full_matrices=False)
    for rank in (32, 48, 64, 80, 96):
        approx = (U[:, :rank] * S[:rank]) @ Vt[:rank]
        preds = {}
        for name, model, w, part in (("C71", c71, w71, approx[:len(w71)]), ("C72", c72, w72, approx[len(w71):])):
            sd = model.state_dict(); sd["rnn.weight_ih_l0"] = part.float(); model.load_state_dict(sd)
            preds[name], _ = predict_valid(model, sc, groups)
            sd["rnn.weight_ih_l0"] = w; model.load_state_dict(sd)
        r[f"svd{rank}_set"] = wp(y, 0.4 * preds["C71"] + 0.4 * preds["C72"] + 0.2 * P["split"])
        print(f"h{h} svd rank {rank} set {r[f'svd{rank}_set']:.6f}", flush=True)

    # 1 WiSE-FT on the split encoder and on the GRU192 teacher member
    g192, gsc = load("gru192_a0.5_s57")
    g_base = {k: v.clone() for k, v in g192.state_dict().items()}
    g_ft = mean_state([FT / f"gru192_a0.5_s57_u0.06_lr0.0002_s7_h{h}_e{e}.pt" for e in (2, 3)])
    for a in (0.0, 0.6, 0.8, 0.9, 1.0, 1.1):
        split.load_state_dict({k: split_base[k] + a * (split_ft[k] - split_base[k]) for k in split_base})
        ps, _ = predict_valid(split, ssc, groups)
        r[f"wise_split_{a:g}"] = {"split_alone": wp(y, ps), "set": wp(y, 0.4 * P["C71"] + 0.4 * P["C72"] + 0.2 * ps),
                                  "dev": predict_dev(split, ssc)}
        g192.load_state_dict({k: g_base[k] + a * (g_ft[k] - g_base[k]) for k in g_base})
        pg, _ = predict_valid(g192, gsc, groups)
        r[f"wise_g192_{a:g}"] = {"alone": wp(y, pg), "dev": predict_dev(g192, gsc)}
        print(f"h{h} WiSE a={a:g}: split alone {r[f'wise_split_{a:g}']['split_alone']:.6f} set "
              f"{r[f'wise_split_{a:g}']['set']:.6f} dev {r[f'wise_split_{a:g}']['dev']:.6f} | g192 alone "
              f"{r[f'wise_g192_{a:g}']['alone']:.6f} dev {r[f'wise_g192_{a:g}']['dev']:.6f}", flush=True)
    split.load_state_dict(split_ft)
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
print("done", flush=True)
