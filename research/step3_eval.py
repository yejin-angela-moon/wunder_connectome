"""Evaluate the step-3 fine-tune regularizer grid on GRU128 s61 with SWA(e2, e3), paired with the seed-7
baseline, plus a 3-seed fine-tune soup (weights averaged over seeds 7, 8, 9)."""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import torch

from global_wp import GlobalWP
from maskft import OUT, member_predictions, batch_arrays, load, BATCH
from pilot import ROOT

torch.set_num_threads(8)
parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
BASE = "gru128_a0.75_s61_u0.06_lr0.0002_s7"
ARMS = {"baseline": BASE, "DropConnect 0.15": BASE + "_dc0.15", "DropConnect 0.3": BASE + "_dc0.3",
        "SAM 0.05": BASE + "_sam0.05", "beta2 0.98": BASE + "_b20.98", "input noise 0.05": BASE + "_in0.05",
        "seed 8": "gru128_a0.75_s61_u0.06_lr0.0002_s8", "seed 9": "gru128_a0.75_s61_u0.06_lr0.0002_s9"}


def wp(y, p):
    a = GlobalWP(); a.add(y, np.asarray(p, np.float32), np.ones(len(y), bool)); return a.result()["weighted_pearson"]


res = {k: [] for k in [*ARMS, "soup (seeds 7-9)", "output average (seeds 7-9)"]}
for h in (0, 1):
    groups = np.arange(1873); groups = groups[groups % 2 != h]
    P = {}
    for name, member in ARMS.items():
        P[name], y = member_predictions(member, h, [2, 3], parquet, groups)
        res[name].append(wp(y, P[name]))
    model, scaler = load("gru128_a0.75_s61")
    states = [torch.load(OUT / f"{ARMS[k]}_h{h}_e{e}.pt", map_location="cpu", weights_only=True)
              for k in ("baseline", "seed 8", "seed 9") for e in (2, 3)]
    model.load_state_dict({k: sum(s[k] for s in states) / len(states) for k in states[0]}); model.eval()
    ps = []
    with torch.no_grad():
        for offset in range(0, len(groups), BATCH):
            x, yy, m = batch_arrays(parquet, groups[offset:offset + BATCH], scaler)
            ps.append(model(torch.from_numpy(x))[0].numpy()[m])
    res["soup (seeds 7-9)"].append(wp(y, np.concatenate(ps)))
    res["output average (seeds 7-9)"].append(wp(y, (P["baseline"] + P["seed 8"] + P["seed 9"]) / 3))
for name, v in res.items():
    d = [a - b for a, b in zip(v, res["baseline"])]
    print(f"{name:28s} h0 {v[0]:.6f} h1 {v[1]:.6f} mean {np.mean(v):.6f}  vs baseline {d[0]:+.5f} / {d[1]:+.5f}")
