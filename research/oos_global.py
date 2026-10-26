"""Rescore frozen models on the fixed masked validation set with Global WP."""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from global_wp import GlobalWP
from pilot import (CACHE, ROOT, N_ROWS, VALID_BYTES, VALID_ID_DIGEST,
                   load_frozen_models, predict_frozen, sequence_id_digest)
from refit_selected import predictors as selected_predictors
from rescore import expanded_models

sys.path.insert(0, str(ROOT))
from utils import DataPoint, FEATURE_COLUMNS, TARGET_COLUMNS, validate_sequence

EXPECTED_SELECTED = 3_528_273


def read_group(parquet, group):
    table = parquet.read_row_group(group, columns=["seq_ix", "step_in_seq", "need_prediction",
                                                     "is_scored", *FEATURE_COLUMNS, *TARGET_COLUMNS],
                                   use_threads=False)
    seq, need = validate_sequence(table)
    x = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    y = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
    mask = table["is_scored"].to_numpy().astype(bool) & need
    return seq, need, x, y, mask


def add(accumulator, y, values, need, mask):
    values = np.asarray(values, np.float32)
    if values.shape != (N_ROWS, 2) or not np.isfinite(values[need]).all():
        raise ValueError("model must predict two finite values on every required row")
    accumulator.add(y, values, mask)


def score(parquet, mode):
    if mode == "baseline":
        from baseline.solution import PredictionModel
        baseline = PredictionModel()
        accumulators = {"baseline": GlobalWP()}
    else:
        frozen = {"pilot": load_frozen_models(), "expanded": expanded_models()}
        selected = selected_predictors()
        names = [f"{stage}_{name}" for stage in frozen for name in ("ridge", "mlp", "gru")]
        names += [f"selected_{name}" for name in selected]
        accumulators = {name: GlobalWP() for name in names}

    seen = set()
    selected_rows = 0
    started = time.monotonic()
    with torch.no_grad():
        for group in range(parquet.num_row_groups):
            seq, need, x, y, mask = read_group(parquet, group)
            if seq in seen:
                raise ValueError("duplicate validation sequence ID")
            seen.add(seq)
            selected_rows += int(mask.sum())
            if mode == "baseline":
                values = np.full((N_ROWS, 2), np.nan, np.float32)
                for step, row in enumerate(x):
                    prediction = baseline.predict(DataPoint(seq, step, bool(need[step]), row))
                    if need[step]:
                        values[step] = prediction
                    elif prediction is not None:
                        raise ValueError("baseline returned a warm-up prediction")
                add(accumulators["baseline"], y, values, need, mask)
            else:
                for stage, models in frozen.items():
                    for name, values in predict_frozen(models, x).items():
                        add(accumulators[f"{stage}_{name}"], y, values, need, mask)
                for name, predict in selected.items():
                    with np.errstate(all="ignore"):
                        values = predict(x)
                    add(accumulators[f"selected_{name}"], y, values, need, mask)
            if (group + 1) % 200 == 0:
                print(mode, "scored sequences", group + 1, flush=True)

    if len(seen) != 1_873 or selected_rows != EXPECTED_SELECTED:
        raise ValueError("validation sequence count or scoring mask changed")
    result = {"metric": "global_wp", "sequences": len(seen), "selected_rows": selected_rows,
              "seconds": time.monotonic() - started,
              "models": {name: accumulator.result() for name, accumulator in accumulators.items()}}
    output = CACHE / f"oos_global_{mode}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: round(value["weighted_pearson"], 6)
                      for name, value in result["models"].items()}, indent=2), flush=True)
    print("saved", output, flush=True)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"baseline", "candidates"}:
        raise SystemExit("usage: oos_global.py {baseline|candidates}")
    torch.set_num_threads(1)
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    if parquet.num_row_groups != 1_873 or (ROOT / "datasets/valid.parquet").stat().st_size != VALID_BYTES:
        raise ValueError("unexpected validation file")
    if sequence_id_digest(parquet) != VALID_ID_DIGEST:
        raise ValueError("validation group order changed")
    score(parquet, sys.argv[1])


if __name__ == "__main__":
    main()
