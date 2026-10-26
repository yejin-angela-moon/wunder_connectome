"""Selection overfitting of our configuration choices (AFML ch. 11; Bailey et al. CSCV).

Every configuration evaluated on both validation halves is a trial. Held-out predictions from the two halves
cover all 1,873 validation sequences (each predicted by models that never saw it). Sequences are split into
S blocks; for every half/half split of the blocks, the configuration with the best pooled WP on one side is
ranked among all configurations on the other side. PBO = share of splits where that winner falls in the
bottom half.

    python research/selection_pbo.py
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pyarrow.parquet as pq

from pilot import CACHE, ROOT

C = CACHE / "maskft"
SINGLE = {"v1": "v1_u0.06_lr0.0002_s7_e2-3", "g128c": "gru128_a0.75_s61_u0.06_lr0.0002_s7_e2-3",
          "ED": "gru128_a0.75_s58_uED_e1-2-3-4", "g192": "gru192_a0.5_s57_u0.06_lr0.0002_s7_e2-3",
          "g192b": "gru192_a0.75_s60_u0.06_lr0.0002_s7_e2-3", "split": "split128_a0.75_s81_u0.06_lr0.0002_s7_tb_e2-3",
          "SD71": "scratch128_uSD_e3-4-5", "SD72": "scratch128s72_uSD_e3-4-5", "G71": "scratch128g2_uSD_e3-4-5",
          "G72": "scratch128g2s72_uSD_e3-4-5", "C71": "scratch128g2c71_uC_e1-2-3", "C72": "scratch128g2c72_uC_e1-2-3",
          "L2": "gruL2_uL2_e3-4-5", "LB": "gruC2048_uLB_e3-4-5", "S71": "gruS128_uS_e8", "S72": "gruS128s72_uS_e8",
          "B192": "gruB192_uB_e8"}
SETS = {"v19-like": {"SD71": .4, "SD72": .4, "split": .2}, "v24-like": {"C71": .4, "C72": .4, "split": .2},
        "v22-like": {"G71": .4, "G72": .4, "split": .2}, "v26-like": {"C71": .25, "C72": .25, "g192": .25, "split": .25},
        "SD pair": {"SD71": .5, "SD72": .5}, "C pair": {"C71": .5, "C72": .5}, "S pair": {"S71": .5, "S72": .5},
        "4 students": {"C71": .2, "C72": .2, "S71": .2, "S72": .2, "split": .2},
        "big mix": {"B192": .3, "C71": .25, "C72": .25, "split": .2},
        "v12-like": {"v1": .25, "ED": .25, "g192": .25, "g192b": .25}}


def load_half(h):
    preds, y = {}, None
    for k, f in SINGLE.items():
        z = np.load(C / f"pred_{f}_h{h}.npz")
        preds[k] = z["p"].astype(np.float64)
        y = z["y"].astype(np.float64) if y is None else y
    return preds, y


def sequence_ids(h):
    mask_pf = pq.ParquetFile(ROOT / "datasets/valid_mask.parquet")
    groups = np.arange(1873); groups = groups[groups % 2 != h]
    return np.concatenate([np.full(int(mask_pf.read_row_group(int(g), columns=["is_scored"])["is_scored"].to_numpy().sum()), g)
                           for g in groups])


def block_stats(y, p, block, n_blocks):
    """Per-block sufficient statistics of the clipped |y|-weighted correlation, shape (blocks, 2 targets, 6)."""
    y = np.clip(y, -2, 2); p = np.clip(p, -2, 2); w = np.abs(y)
    cols = [w, w * y, w * p, w * y * y, w * p * p, w * y * p]
    return np.stack([np.stack([np.bincount(block, c[:, k], n_blocks) for c in cols], -1) for k in range(2)], 1)


def wp_from(stats):
    sw, sy, sp, syy, spp, syp = np.moveaxis(stats, -1, 0)
    cov = syp - sy * sp / sw
    return float(np.mean(cov / np.sqrt((syy - sy ** 2 / sw) * (spp - sp ** 2 / sw))))


def main(n_blocks=16):
    names, stats = [], []
    halves = [load_half(h) for h in (0, 1)]
    seq = np.concatenate([sequence_ids(0), sequence_ids(1)])
    order = np.random.default_rng(20261006).permutation(1873)
    block = (order[seq] * n_blocks) // 1873
    y = np.concatenate([halves[0][1], halves[1][1]])
    configs = {k: {k: 1.0} for k in SINGLE} | SETS
    for name, parts in configs.items():
        p = np.concatenate([sum(w * halves[h][0][k] for k, w in parts.items()) for h in (0, 1)])
        names.append(name); stats.append(block_stats(y, p, block, n_blocks))
    stats = np.stack(stats)                                             # (configs, blocks, 2, 6)
    full = np.array([wp_from(s.sum(0)) for s in stats])
    print("full-validation pooled WP (held-out predictions):")
    for i in np.argsort(-full):
        print(f"   {names[i]:12s} {full[i]:.6f}")
    logits, gaps = [], []
    for chosen in itertools.combinations(range(n_blocks), n_blocks // 2):
        rest = [b for b in range(n_blocks) if b not in chosen]
        is_score = np.array([wp_from(s[list(chosen)].sum(0)) for s in stats])
        oos_score = np.array([wp_from(s[rest].sum(0)) for s in stats])
        best = int(np.argmax(is_score))
        rank = (oos_score < oos_score[best]).sum() + 0.5 * ((oos_score == oos_score[best]).sum() - 1) + 1
        omega = rank / (len(names) + 1)
        logits.append(math.log(omega / (1 - omega)))
        gaps.append((names[best], is_score[best] - np.sort(is_score)[-2], oos_score[best] - oos_score.max()))
    logits = np.array(logits)
    winners = {}
    for n, _, _ in gaps:
        winners[n] = winners.get(n, 0) + 1
    print(f"\nCSCV over {len(logits)} splits of {n_blocks} sequence blocks, {len(names)} configurations:")
    print(f"   PBO {np.mean(logits <= 0):.3f}; median logit {np.median(logits):.2f}")
    print(f"   in-sample winners: {sorted(winners.items(), key=lambda kv: -kv[1])}")
    print(f"   mean OOS shortfall of the IS winner vs the OOS best: {np.mean([g[2] for g in gaps]):.6f}")


if __name__ == "__main__":
    main()
