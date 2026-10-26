"""Clustered permutation importance (AFML ch. 8, MDA by feature clusters) for the continued students.

For a sequence model, a cluster is "permuted" by swapping its standardized columns with those of another
sequence in the batch (cyclic shift), which keeps the cluster's own time structure but breaks its relation to
the targets and to the other inputs. Importance = held-out pooled WP drop. A negative importance means the
cluster hurts held-out accuracy.

    python research/cluster_mda.py
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import torch

from global_wp import GlobalWP
from maskft import BATCH, batch_arrays, load
from pilot import CACHE, ROOT, make_gru

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CLUSTERS = {"i0 bid price": range(0, 11), "i0 ask price": range(11, 22), "i0 bid volume": range(22, 33),
            "i0 ask volume": range(33, 44), "i0 trade price": range(44, 48), "i0 trade volume": range(48, 52),
            "i1 bid price": range(52, 63), "i1 ask price": range(63, 74), "i1 bid volume": range(74, 85),
            "i1 ask volume": range(85, 96), "i1 trade price": range(96, 100), "i1 trade volume": range(100, 104),
            "a0-a4": range(104, 109), "a5-a7": range(109, 112)}
valid = pq.ParquetFile(ROOT / "datasets/valid.parquet")
ED = CACHE / "ensdistill"


def student(seed, h):
    states = [torch.load(ED / f"scratch128g2_a0.75_t6000_p6000_lr0.0002_s{seed}_cont_h{h}_e{e}.pt", map_location="cpu",
                         weights_only=True) for e in (1, 2, 3)]
    model = make_gru(128)
    model.load_state_dict({k: sum(s[k] for s in states) / 3 for k in states[0]})
    return model.to(DEVICE).eval()


results = {}
_, scaler = load("v1")
for h in (0, 1):
    groups = np.arange(1873); groups = groups[groups % 2 != h]
    models = [student(71, h), student(72, h)]
    accs = {name: GlobalWP() for name in ["base", *CLUSTERS]}
    with torch.no_grad():
        for offset in range(0, len(groups), BATCH):
            x, y, mask = batch_arrays(valid, groups[offset:offset + BATCH], scaler)
            variants = {"base": x}
            for name, cols in CLUSTERS.items():
                xs = x.copy()
                cols = list(cols)
                xs[:, :, cols] = np.roll(x[:, :, cols], 1, axis=0)        # another sequence's cluster
                variants[name] = xs
            for name, xv in variants.items():
                xt = torch.from_numpy(xv).to(DEVICE)
                p = np.mean([m(xt)[0].cpu().numpy() for m in models], axis=0)
                for i in range(len(xv)):
                    accs[name].add(y[i], p[i], mask[i])
    base = accs["base"].result()["weighted_pearson"]
    results[f"h{h}"] = {"base": base, **{n: base - accs[n].result()["weighted_pearson"] for n in CLUSTERS}}
    print(f"h{h} base pair WP {base:.6f}", flush=True)
print("\nimportance = WP drop when the cluster is swapped (negative = cluster hurts):")
for n in CLUSTERS:
    a, b = results["h0"][n], results["h1"][n]
    print(f"   {n:16s} h0 {a:+.5f} h1 {b:+.5f} mean {(a + b) / 2:+.5f}")
(CACHE / "step01" / "cluster_mda.json").write_text(json.dumps(results, indent=2) + "\n")
