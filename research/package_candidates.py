"""Export and locally check three NumPy-only candidate submission ZIPs."""
from __future__ import annotations

import argparse
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

from candidate_solution import PredictionModel
from pilot import CACHE, ROOT, gru_predict, make_gru, make_mlp
from scale import FIT, lag_features

from utils import DataPoint, FEATURE_COLUMNS

SELECTED = CACHE / "selected"
PACKAGES = CACHE / "packages"
NAMES = ("ridge_lag", "mlp_norm", "gru96", "gru96_ensemble", "gru96_distill")
NAMES += ("gru96_distill_alpha75",)


def export(name):
    folder = SELECTED / name
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "model.npz"
    if name == "ridge_lag":
        np.savez(output, kind=name, weights=np.load(folder / "scaled_ridge.npy"))
    elif name == "mlp_norm":
        state = torch.load(folder / "scaled_mlp.pt", map_location="cpu", weights_only=True)
        with np.load(folder / "mlp_standardizer.npz") as source:
            np.savez(output, kind=name, w0=state["0.weight"].numpy(), b0=state["0.bias"].numpy(),
                     w1=state["2.weight"].numpy(), b1=state["2.bias"].numpy(),
                     mean=source["mean"], std=source["std"])
    elif name == "gru96":
        state = torch.load(folder / "scaled_gru.pt", map_location="cpu", weights_only=True)
        np.savez(output, kind=name, **{key: value.numpy() for key, value in state.items()})
    elif name == "gru96_ensemble":
        source = CACHE / "improve/full_fit"
        states = [torch.load(source / model / "scaled_gru.pt", map_location="cpu",
                             weights_only=True)
                  for model in ("gru96_norm5ep", "gru96_seed17", "gru96_seed37")]
        with np.load(source / "gru96_norm5ep/gru_standardizer.npz") as standardizer:
            values = {f"m{index}_{key}": value.numpy()
                      for index, state in enumerate(states) for key, value in state.items()}
            np.savez(output, kind=name, mean=standardizer["mean"], std=standardizer["std"],
                     **values)
    else:
        source = CACHE / "advance/full_fit/distill" / (
            "model_alpha75.pt" if name.endswith("alpha75") else "model.pt")
        state = torch.load(source, map_location="cpu", weights_only=True)
        with np.load(CACHE / "improve/full_fit/gru96_norm5ep/gru_standardizer.npz") as standardizer:
            np.savez(output, kind="gru96_norm", mean=standardizer["mean"],
                     std=standardizer["std"],
                     **{key: value.numpy() for key, value in state.items()})
    return output


def check(name, model_path, x):
    callback = PredictionModel(model_path)
    started = time.perf_counter()
    predictions = []
    for step, row in enumerate(x):
        value = callback.predict(DataPoint(123, step, step >= 99, row))
        if step >= 99:
            predictions.append(value)
    elapsed = time.perf_counter() - started
    predictions = np.asarray(predictions, np.float32)
    if predictions.shape != (19_901, 2) or not np.isfinite(predictions).all():
        raise ValueError(f"invalid callback predictions: {name}")
    folder = SELECTED / name
    with torch.no_grad(), np.errstate(all="ignore"):
        if name == "ridge_lag":
            weights = np.load(folder / "scaled_ridge.npy")
            expected = lag_features(x) @ weights[:-1] + weights[-1]
        elif name == "mlp_norm":
            model = make_mlp().eval()
            model.load_state_dict(torch.load(folder / "scaled_mlp.pt", map_location="cpu", weights_only=True))
            with np.load(folder / "mlp_standardizer.npz") as source:
                normalized = np.clip((x - source["mean"]) / source["std"], -10, 10)
            expected = model(torch.from_numpy(np.ascontiguousarray(normalized))).numpy()
        elif name == "gru96":
            model = make_gru(96).eval()
            model.load_state_dict(torch.load(folder / "scaled_gru.pt", map_location="cpu", weights_only=True))
            expected = gru_predict(model, x)
        elif name == "gru96_ensemble":
            source = CACHE / "improve/full_fit"
            models = []
            for candidate in ("gru96_norm5ep", "gru96_seed17", "gru96_seed37"):
                model = make_gru(96).eval()
                model.load_state_dict(torch.load(source / candidate / "scaled_gru.pt",
                                                  map_location="cpu", weights_only=True))
                models.append(model)
            with np.load(source / "gru96_norm5ep/gru_standardizer.npz") as standardizer:
                normalized = np.clip((x - standardizer["mean"]) / standardizer["std"], -10, 10)
            expected = (gru_predict(models[0], normalized) + gru_predict(models[1], x)
                        + gru_predict(models[2], x)) / 3
        else:
            source = CACHE / "advance/full_fit/distill" / (
                "model_alpha75.pt" if name.endswith("alpha75") else "model.pt")
            model = make_gru(96).eval()
            model.load_state_dict(torch.load(source, map_location="cpu", weights_only=True))
            with np.load(CACHE / "improve/full_fit/gru96_norm5ep/gru_standardizer.npz") as standardizer:
                normalized = np.clip((x - standardizer["mean"]) / standardizer["std"], -10, 10)
            expected = gru_predict(model, normalized)
    difference = float(np.max(np.abs(predictions - expected[99:])))
    # Recurrent float32 round-off accumulates over 20,000 sequential steps.
    tolerance = 5e-5 if name.startswith("gru96") else 1e-5
    if difference > tolerance:
        raise ValueError(f"row-wise and batched inference differ: {name}, {difference}")
    return {"seconds_per_20k_train_rows_on_this_host": elapsed,
            "projected_minutes_per_1970_sequences_on_this_host": elapsed * 1970 / 60,
            "max_batch_difference": difference}


def package(name, model_path):
    PACKAGES.mkdir(exist_ok=True)
    output = PACKAGES / f"{name}.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(Path(__file__).with_name("candidate_solution.py"), "solution.py")
        archive.write(Path(__file__).with_name("numpy_gru.py"), "numpy_gru.py")
        archive.write(model_path, "model.npz")
    with zipfile.ZipFile(output) as archive:
        if set(archive.namelist()) != {"solution.py", "numpy_gru.py", "model.npz"}:
            raise ValueError("unexpected ZIP contents")
        with tempfile.TemporaryDirectory() as directory:
            archive.extractall(directory)
            subprocess.run(
                [sys.executable, "-c", "from types import SimpleNamespace; import numpy as np; "
                 "from solution import PredictionModel; m=PredictionModel(); "
                 "p=lambda seq,step,needed: SimpleNamespace(seq_ix=seq,step_in_seq=step,"
                 "need_prediction=needed,state=np.zeros(112,dtype=np.float32)); "
                 "assert m.predict(p(0,0,False)) is None; "
                 "assert m.predict(p(0,1,True)).shape==(2,); "
                 "assert m.predict(p(1,0,True)).shape==(2,)"],
                cwd=directory, check=True, timeout=30, capture_output=True)
    if output.stat().st_size >= 20_000_000:
        raise ValueError(f"candidate ZIP exceeds 20 MB: {output}")
    return output.stat().st_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=(*NAMES, "all"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    parquet = pq.ParquetFile(ROOT / "datasets/train.parquet")
    table = parquet.read_row_group(int(FIT[0]), columns=FEATURE_COLUMNS, use_threads=False)
    x = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    report = PACKAGES / "local_checks.json"
    results = json.loads(report.read_text()) if report.exists() else {}
    for name in NAMES if args.model == "all" else (args.model,):
        model_path = export(name)
        results[name] = {**check(name, model_path, x), "zip_bytes": package(name, model_path)}
        print(name, results[name], flush=True)
    PACKAGES.mkdir(exist_ok=True)
    report.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
