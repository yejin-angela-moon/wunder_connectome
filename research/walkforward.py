"""Chronological expanding-window evaluation within the locked fitting pool.

Requires an independently supplied sequence chronology; parquet order and
shuffled seq_ix values are not a valid time axis.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from pilot import CACHE, ROOT, check_train
from scale import FIT, score_dev, train_gru, train_mlp, train_ridge


def ordered_fit_groups(parquet, rows, fit_groups=FIT):
    group_by_id = {}
    for group in range(parquet.num_row_groups):
        seq = int(parquet.read_row_group(group, columns=["seq_ix"], use_threads=False)["seq_ix"][0].as_py())
        if seq in group_by_id:
            raise ValueError(f"duplicate train seq_ix {seq}")
        group_by_id[seq] = group
    if not {"seq_ix", "rank"}.issubset(rows.fieldnames or []):
        raise ValueError("order CSV needs seq_ix,rank columns")
    ordered = [None] * len(group_by_id)
    seen = set()
    for row in rows:
        seq, rank = int(row["seq_ix"]), int(row["rank"])
        if seq not in group_by_id or seq in seen or not 0 <= rank < len(ordered) or ordered[rank] is not None:
            raise ValueError(f"invalid or repeated sequence/rank: {seq}, {rank}")
        ordered[rank] = group_by_id[seq]
        seen.add(seq)
    if len(seen) != len(group_by_id):
        raise ValueError(f"chronology must contain all {len(group_by_id)} train sequences exactly once")
    # The pre-existing development holdout never becomes a walk-forward training group.
    fit_set = set(fit_groups)
    return np.asarray([g for g in ordered if g in fit_set], dtype=np.int32)


def expanding_folds(ordered, initial_fraction=0.5, n_folds=3, embargo=0):
    if not 0 < initial_fraction < 1 or n_folds < 1 or embargo < 0:
        raise ValueError("expected 0 < initial_fraction < 1, n_folds >= 1, embargo >= 0")
    initial = int(len(ordered) * initial_fraction)
    windows = np.array_split(np.arange(initial, len(ordered)), n_folds)
    if initial <= embargo or any(len(window) == 0 for window in windows):
        raise ValueError("not enough chronological groups for the requested folds and embargo")
    for number, window in enumerate(windows, 1):
        start, stop = int(window[0]), int(window[-1]) + 1
        yield number, ordered[:start - embargo], ordered[start:stop], start, stop


def selfcheck():
    class TinyParquet:
        num_row_groups = 12

        def read_row_group(self, group, **_):
            return {"seq_ix": pa.array([group + 100])}

    fit_pool = np.arange(1, 11, dtype=np.int32)
    reader = csv.DictReader(io.StringIO("seq_ix,rank\n" + "".join(f"{g + 100},{11 - g}\n" for g in range(12))))
    ordered = ordered_fit_groups(TinyParquet(), reader, fit_pool)
    folds = list(expanding_folds(ordered, 0.4, 3, 1))
    assert len(folds) == 3 and set(ordered) == set(fit_pool)
    for _, train, validation, start, _ in folds:
        assert len(train) > 0 and len(validation) > 0
        assert set(train).isdisjoint(validation)
        assert set(train).issubset(fit_pool) and set(validation).issubset(fit_pool)
        assert np.array_equal(train, ordered[:start - 1])
        assert np.array_equal(validation, ordered[start:start + len(validation)])
    incomplete = csv.DictReader(io.StringIO("seq_ix,rank\n100,0\n"))
    try:
        ordered_fit_groups(TinyParquet(), incomplete, fit_pool)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete chronology was accepted")
    print("walk-forward chronology, embargo, and disjoint-fold checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["selfcheck", "plan", "run"])
    parser.add_argument("--order-file", type=Path, help="CSV: seq_ix,rank; rank 0 is earliest")
    parser.add_argument("--initial-fraction", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--embargo-groups", type=int, default=0)
    parser.add_argument("--fold", type=int, help="run one 1-based fold; default runs all")
    parser.add_argument("--output-dir", type=Path, default=CACHE / "walkforward")
    args = parser.parse_args()
    if args.command == "selfcheck":
        selfcheck()
        return
    if args.order_file is None:
        parser.error("--order-file is required; seq_ix and parquet row-group order are not chronological")

    torch.set_num_threads(1)
    parquet = pq.ParquetFile(ROOT / "datasets/train.parquet")
    check_train(parquet)
    with args.order_file.open(newline="") as source:
        ordered = ordered_fit_groups(parquet, csv.DictReader(source))
    if len(ordered) != len(FIT) or set(ordered) != set(FIT):
        raise ValueError("walk-forward pool must equal the locked fitting groups")
    digest = hashlib.sha256(args.order_file.read_bytes()).hexdigest()
    folds = list(expanding_folds(ordered, args.initial_fraction, args.folds, args.embargo_groups))
    if args.fold is not None and not 1 <= args.fold <= len(folds):
        parser.error("--fold must be between 1 and --folds")
    for number, train, validation, start, stop in folds:
        print(f"fold {number}: train={len(train)} embargo={args.embargo_groups} "
              f"validation={len(validation)} validation_ranks=[{start},{stop})", flush=True)
        if args.command == "plan" or args.fold is not None and args.fold != number:
            continue
        folder = args.output_dir / f"fold_{number:02d}"
        folder.mkdir(parents=True, exist_ok=False)
        metadata = {"order_sha256": digest, "train_groups": train.tolist(),
                    "validation_groups": validation.tolist(), "validation_rank_range": [start, stop],
                    "embargo_groups": args.embargo_groups}
        (folder / "fold.json").write_text(json.dumps(metadata) + "\n")
        for fit in (train_ridge, train_mlp, train_gru):
            fit(parquet, train, folder)
        score_dev(parquet, validation, folder)
    if args.command == "run" and args.fold is None:
        fold_scores = [json.loads((args.output_dir / f"fold_{number:02d}" / "scaled_dev_scores.json").read_text())
                       for number, *_ in folds]
        summary = {}
        for model in ("ridge", "mlp", "gru"):
            count = sum(scores[model]["eligible_blocks"] for scores in fold_scores)
            t0, t1 = (sum(scores[model][target] * scores[model]["eligible_blocks"]
                          for scores in fold_scores) / count for target in ("t0", "t1"))
            summary[model] = {"t0": t0, "t1": t1, "weighted_pearson": (t0 + t1) / 2,
                              "eligible_blocks": count}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
