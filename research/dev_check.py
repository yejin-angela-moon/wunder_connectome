"""Out-of-period check of deployable member sets on train sequences that no member has seen.

DEV (every 10th train sequence) minus the 6,000-sequence student pool (ensdistill) and the split base's
FIT groups (distill_mps) leaves 451 train-period sequences. Scored like step01's DEV: Global WP over all
rows after warm-up. Member predictions are cached, so further sets cost nothing.

    python research/dev_check.py
"""
from __future__ import annotations

import os

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from distill_mps import build
from package_ft import load_member, stream_model, streams
from global_wp import GlobalWP
from pilot import CACHE, ROOT, WARMUP
from scale import DEV, read_group
from stack import DISTILL, FULL

DEVICE = os.environ.get("DEV_CHECK_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
OUT = CACHE / "dev_check"
E, F = CACHE / "ensdistill", CACHE / "maskft"
C = "scratch128g2_a0.75_t6000_p6000_lr0.0002_s{}_cont_all_e{}.pt"
S = "scratch128g2_a0.75_t6000_p6000_lr0.0015_s{}_aw0.0001_oc_cl1_all_e8.pt"
SPLIT = "split128_a0.75_s81_u0.06_lr0.0002_s7_tb_all"
MEMBERS = {f"C{s}": "+".join(str(E / C.format(s, e)) for e in (1, 2, 3)) for s in (75, 76, 77, 78)} | {
    "S71": str(E / S.format(71)), "S72": str(E / S.format(72)),
    "B192": str(E / "scratch192L2g2_a0.75_t6000_p6000_lr0.0015_s71_aw0.0001_oc_cl1_do0.1_all_e8.pt"),
    "split": f"{F / (SPLIT + '_e2.pt')}+{F / (SPLIT + '_e3.pt')}",
    "split_wise": str(F / (SPLIT + "_swa23_wise0.9.pt")),
    "twinC": "+".join(str(E / f"scratchtwin96g2_a0.75_t6000_p6000_lr0.0002_s71_cont_all_e{e}.pt") for e in (1, 2, 3)),
    "twinC72": "+".join(str(E / f"scratchtwin96g2_a0.75_t6000_p6000_lr0.0002_s72_cont_all_e{e}.pt") for e in (1, 2, 3)),
    "twinC73": "+".join(str(E / f"scratchtwin96g2_a0.75_t6000_p6000_lr0.0002_s73_cont_all_e{e}.pt") for e in (1, 2, 3)),
    "twinC74": "+".join(str(E / f"scratchtwin96g2_a0.75_t6000_p6000_lr0.0002_s74_cont_all_e{e}.pt") for e in (1, 2, 3)),
    "twinC75": "+".join(str(E / f"scratchtwin96g2_a0.75_t6000_p6000_lr0.0002_s75_cont_all_e{e}.pt") for e in (1, 2, 3)),
    "twinC76": "+".join(str(E / f"scratchtwin96g2_a0.75_t6000_p6000_lr0.0002_s76_cont_all_e{e}.pt") for e in (1, 2, 3)),
    "twinC77": "+".join(str(E / f"scratchtwin96g2_a0.75_t6000_p6000_lr0.0002_s77_cont_all_e{e}.pt") for e in (1, 2, 3)),
    "twinC78": "+".join(str(E / f"scratchtwin96g2_a0.75_t6000_p6000_lr0.0002_s78_cont_all_e{e}.pt") for e in (1, 2, 3))} | {
    f"encC{s}": "+".join(str(E / f"scratchtwinenc96g2_a0.75_t6000_p6000_lr0.0002_s{s}_cont_all_e{e}.pt") for e in (1, 2, 3))
    for s in (71, 72, 73, 74, 75, 76)} | {
    f"encG{g}h{h}": "+".join(str(E / f"scratchtwinenc96g{g}_a0.75_t6000_p6000_lr0.0002_s71_cont_h{h}_e{e}.pt") for e in (1, 2, 3))
    for g in (2, 3) for h in (0, 1)}                                  # seed-71 half models, gen-2 and gen-3 teacher
C4 = {f"C{s}": 0.2 for s in (75, 76, 77, 78)}
TWINS = ("twinC", "twinC72", "twinC73", "twinC74")
SETS = {"v28 (v24 members)": C4 | {"split": 0.2},
        "v28 with WiSE split": C4 | {"split_wise": 0.2},
        "v30": {k: v / 2 for k, v in C4.items()} | {"S71": .1, "S72": .1, "B192": .2, "split_wise": .2},
        "v30 with plain split": {k: v / 2 for k, v in C4.items()} | {"S71": .1, "S72": .1, "B192": .2, "split": .2},
        "v31": {k: v / 2 for k, v in C4.items()} | {"twinC": .2, "B192": .2, "split_wise": .2},
        "v32": {k: .075 for k in C4} | {"twinC": .3, "B192": .2, "split_wise": .2},
        "two twins .3, B192 .2": {k: .075 for k in C4} | {"twinC": .15, "twinC72": .15, "B192": .2, "split_wise": .2},
        "two twins .4, B192 .2": {k: .05 for k in C4} | {"twinC": .2, "twinC72": .2, "B192": .2, "split_wise": .2},
        "two twins .4, no B192": {k: .1 for k in C4} | {"twinC": .2, "twinC72": .2, "split_wise": .2},
        "v33": {"C75": .1, "C76": .1, "twinC": .2, "twinC72": .2, "B192": .2, "split_wise": .2},
        "v34: four twins .6, B192 .2": {t: .15 for t in TWINS} | {"B192": .2, "split_wise": .2},
        "four twins .5, B192 .25": {t: .125 for t in TWINS} | {"B192": .25, "split_wise": .25},
        "four twins .7, B192 .15": {t: .175 for t in TWINS} | {"B192": .15, "split_wise": .15},
        "v35": {t: .2 for t in TWINS} | {"split_wise": .2},
        "v36a: five twins .8, split .2": {t: .16 for t in TWINS + ("twinC75",)} | {"split_wise": .2},
        "v36b: six twins 1.0, no split": {t: 1 / 6 for t in TWINS + ("twinC75", "twinC76")},
        "six twins .8, split .2": {t: .8 / 6 for t in TWINS + ("twinC75", "twinC76")} | {"split_wise": .2},
        "v36": {t: 1 / 6 for t in TWINS + ("twinC75", "twinC76")},
        "v37: seven twins": {t: 1 / 7 for t in TWINS + ("twinC75", "twinC76", "twinC77")},
        "eight twins": {t: 1 / 8 for t in TWINS + ("twinC75", "twinC76", "twinC77", "twinC78")},
        "pair twin s71 + twin s72": {"twinC": .5, "twinC72": .5},
        "pair enc s71 + twin s72": {"encC71": .5, "twinC72": .5},
        "v37: five twin-encoders": {f"encC{s}": 1 / 5 for s in (71, 72, 73, 74, 75)},
        "v38: six twin-encoders": {f"encC{s}": 1 / 6 for s in (71, 72, 73, 74, 75, 76)},
        "v39: gen-3 halves + four": {k: 1 / 6 for k in ("encG3h0", "encG3h1", "encC71", "encC72", "encC73", "encC74")},
        "v39 with gen-2 halves": {k: 1 / 6 for k in ("encG2h0", "encG2h1", "encC71", "encC72", "encC73", "encC74")}}


def groups():
    n_train = 10_607
    base = np.sort(np.random.default_rng(123).choice(n_train, 2_000, replace=False))     # ensdistill pool
    extra = np.random.default_rng(124).choice(np.setdiff1d(np.arange(n_train), base), 4_000, replace=False)
    return np.setdiff1d(DEV, np.concatenate([base, extra, np.load(DISTILL / "groups.npy")]))


def main():
    OUT.mkdir(exist_ok=True)
    train = pq.ParquetFile(ROOT / "datasets/train.parquet")
    with np.load(FULL / "gru96_norm5ep/gru_standardizer.npz") as z:
        mean, std = z["mean"], z["std"]
    dev = groups()
    data = []
    for g in dev:                                                     # standardized inputs only (4 GB)
        x, y = read_group(train, int(g))
        data.append((np.clip((x - mean) / std, -10, 10).astype(np.float32), y))
    print(f"{len(dev)} unseen DEV sequences", flush=True)
    for name, spec in MEMBERS.items():
        path = OUT / f"{name}.npy"
        if path.exists() or not all(Path(p).exists() for p in spec.split("+")):
            continue
        state = load_member(spec)
        model = (stream_model(state) if streams(state) else build(state)).to(DEVICE).eval()
        model.load_state_dict(state)
        out = []
        with torch.no_grad():
            for i in range(0, len(data), 16):
                x = np.stack([d[0] for d in data[i:i + 16]])
                out.append(model(torch.from_numpy(x).to(DEVICE))[0].cpu().numpy().astype(np.float16))
        np.save(path, np.concatenate(out))
        print("predicted", name, flush=True)
    preds = {name: np.load(OUT / f"{name}.npy").astype(np.float32) for name in MEMBERS if (OUT / f"{name}.npy").exists()}
    need = np.arange(len(data[0][1])) >= WARMUP
    for name, parts in ({k: {k: 1.0} for k in preds} | SETS).items():
        if not all(k in preds for k in parts):
            continue
        acc = GlobalWP()
        for i, (_, y) in enumerate(data):
            acc.add(y, sum(w * preds[k][i] for k, w in parts.items()), need)
        print(f"{name:24s} {acc.result()['weighted_pearson']:.6f}", flush=True)


if __name__ == "__main__":
    main()
