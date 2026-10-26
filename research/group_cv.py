"""Fixed, non-chronological sequence-group screen inside the locked fitting pool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from global_wp import GlobalWP
from pilot import CACHE, ROOT, N_ROWS, WARMUP, check_train, gru_predict, make_gru, make_mlp
from scale import DEV, FIT, lag_features, read_group, train_gru, train_mlp, train_ridge

POOL_SIZE = 1_536
N_FOLDS = 3
SEED = 20260917
OUTPUT = CACHE / "group_cv"
MODELS = ("ridge", "ridge_lag", "mlp", "mlp_sample8", "mlp_norm",
          "gru_epoch1", "gru_epoch2", "gru", "gru96")


def folds():
    pool = FIT[np.linspace(0, len(FIT) - 1, POOL_SIZE, dtype=int)]
    shuffled = np.random.default_rng(SEED).permutation(pool)
    parts = np.split(shuffled, N_FOLDS)
    for number, validation in enumerate(parts, 1):
        train = np.concatenate([part for index, part in enumerate(parts, 1) if index != number])
        yield number, train, validation


def selfcheck():
    plans = list(folds())
    assert len(plans) == 3
    assert len({int(g) for _, _, val in plans for g in val}) == POOL_SIZE
    for _, train, validation in plans:
        assert len(train) == 1_024 and len(validation) == 512
        assert set(train).isdisjoint(validation)
        assert set(train).isdisjoint(DEV) and set(validation).isdisjoint(DEV)
    x = np.arange(3 * 112, dtype=np.float32).reshape(3, 112)
    features = lag_features(x)
    assert features.shape == (3, 224) and np.all(features[0, 112:] == 0)
    assert np.array_equal(features[1, 112:], x[1] - x[0])
    print("group-CV split and causal lag checks passed")


def fit_models(parquet, train, folder):
    experiments = (("ridge", train_ridge, {}),
                   ("ridge_lag", train_ridge, {"lag_differences": True}),
                   ("mlp", train_mlp, {}),
                   ("mlp_sample8", train_mlp, {"sample_step": 8}),
                   ("mlp_norm", train_mlp, {"standardize": True}),
                   ("gru", train_gru, {}),
                   ("gru96", train_gru, {"hidden_size": 96}))
    for name, fit, options in experiments:
        output = folder / name
        output.mkdir(exist_ok=True)
        artifact = output / ("scaled_ridge.npy" if name.startswith("ridge") else
                             "scaled_mlp.pt" if name.startswith("mlp") else "scaled_gru.pt")
        if artifact.exists():
            print("using completed", name, flush=True)
            continue
        print("fitting", name, "groups", len(train), flush=True)
        fit(parquet, train, output, **options)


def load_predictors(folder):
    predictors = {}
    for name in ("ridge", "ridge_lag"):
        weights = np.load(folder / name / "scaled_ridge.npy")
        predictors[name] = lambda x, w=weights, lag=name == "ridge_lag": (
            (lag_features(x) if lag else x) @ w[:-1] + w[-1])
    for name in ("mlp", "mlp_sample8", "mlp_norm"):
        model = make_mlp()
        model.load_state_dict(torch.load(folder / name / "scaled_mlp.pt", map_location="cpu", weights_only=True))
        model.eval()
        standardizer = None
        if name == "mlp_norm":
            with np.load(folder / name / "mlp_standardizer.npz") as source:
                standardizer = (source["mean"], source["std"])

        def predict_mlp(x, model=model, standardizer=standardizer):
            if standardizer is not None:
                x = np.clip((x - standardizer[0]) / standardizer[1], -10, 10)
            return model(torch.from_numpy(np.ascontiguousarray(x))).numpy()

        predictors[name] = predict_mlp
    for name, width, checkpoint in (("gru_epoch1", 64, "scaled_gru_epoch1.pt"),
                                    ("gru_epoch2", 64, "scaled_gru_epoch2.pt"),
                                    ("gru", 64, "scaled_gru.pt"),
                                    ("gru96", 96, "scaled_gru.pt")):
        source = "gru96" if width == 96 else "gru"
        model = make_gru(width)
        model.load_state_dict(torch.load(folder / source / checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        predictors[name] = lambda x, model=model: gru_predict(model, x)
    return predictors


def score_models(parquet, validation, folder):
    from utils import BlockAccumulator

    predictors = load_predictors(folder)
    global_scores = {name: {mask: GlobalWP() for mask in ("all", "random20")} for name in MODELS}
    block_scores = {name: BlockAccumulator() for name in MODELS}
    required = np.arange(N_ROWS) >= WARMUP
    with torch.no_grad():
        for n, group in enumerate(validation, 1):
            x, y = read_group(parquet, group)
            random20 = required & (np.random.default_rng(SEED + int(group)).random(N_ROWS) < 0.2)
            for name, predict in predictors.items():
                values = predict(x)
                global_scores[name]["all"].add(y, values, required)
                global_scores[name]["random20"].add(y, values, random20)
                block_scores[name].add(y, values, required)
            if n % 100 == 0:
                print("scored fold groups", n, flush=True)
    results = {}
    for name in MODELS:
        per_sequence = np.asarray(block_scores[name].scores).mean(axis=1)
        results[name] = {"global": {mask: acc.result() for mask, acc in global_scores[name].items()},
                         "sequence_wp_quantiles": np.quantile(per_sequence, [0.1, 0.5, 0.9]).tolist()}
    (folder / "scores.json").write_text(json.dumps(results, indent=2) + "\n")
    print({name: round(result["global"]["all"]["weighted_pearson"], 6)
           for name, result in results.items()}, flush=True)


def summarize():
    results = [json.loads((OUTPUT / f"fold_{i}" / "scores.json").read_text()) for i in range(1, N_FOLDS + 1)]
    summary = {}
    for name in MODELS:
        values = np.asarray([fold[name]["global"]["all"]["weighted_pearson"] for fold in results])
        random20 = np.asarray([fold[name]["global"]["random20"]["weighted_pearson"] for fold in results])
        summary[name] = {"folds": values.tolist(), "mean": float(values.mean()),
                         "std": float(values.std()), "random20_mean": float(random20.mean())}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("selfcheck", "run", "summary"))
    parser.add_argument("--fold", type=int, choices=(1, 2, 3), help="one fold; omit to run all")
    args = parser.parse_args()
    if args.command == "selfcheck":
        selfcheck()
        return
    if args.command == "summary":
        summarize()
        return
    torch.set_num_threads(1)
    parquet = pq.ParquetFile(ROOT / "datasets/train.parquet")
    check_train(parquet)
    OUTPUT.mkdir(exist_ok=True)
    for number, train, validation in folds():
        if args.fold is not None and args.fold != number:
            continue
        folder = OUTPUT / f"fold_{number}"
        folder.mkdir(exist_ok=True)
        plan = {"seed": SEED, "train_groups": train.tolist(), "validation_groups": validation.tolist()}
        manifest = folder / "plan.json"
        if manifest.exists() and json.loads(manifest.read_text()) != plan:
            raise ValueError("existing fold plan differs")
        manifest.write_text(json.dumps(plan) + "\n")
        fit_models(parquet, train, folder)
        score_models(parquet, validation, folder)


if __name__ == "__main__":
    main()
