"""Small, reproducible train-only comparison; run prepare/score with the kit Python.

prepare: Python with pyarrow (the competition environment)
train: Python with torch
score: Python with pyarrow
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).resolve().parent / "cache"
FIT_GROUPS = 128
DEV_GROUPS = 32
N_ROWS = 20_000
N_FEATURES = 112
WARMUP = 99
SEED = 7
TRAIN_ID_DIGEST = "11ba03e69dfdb79a1a78db84d2895fa98711b8cbb2e2d0bb94d66d93ce19701b"
TRAIN_BYTES = 29_240_219_011
VALID_ID_DIGEST = "d85f36905d7ebdf72ddd1169862a1c9d2d87bf832bf7a81f0aabe953f1c331fd"
VALID_BYTES = 5_615_509_128


def groups(count: int, n_fit: int = FIT_GROUPS, n_dev: int = DEV_GROUPS):
    """Fixed group-index split; pilot groups span the entire train file."""
    fit = np.arange(count)[np.arange(count) % 10 != 0]
    dev = np.arange(count)[np.arange(count) % 10 == 0]
    if len(fit) < n_fit or len(dev) < n_dev:
        raise ValueError("not enough row groups")
    return fit[np.linspace(0, len(fit) - 1, n_fit, dtype=int)], dev[np.linspace(0, len(dev) - 1, n_dev, dtype=int)]


def check_train(parquet):
    if parquet.num_row_groups != 10_607 or (ROOT / "datasets/train.parquet").stat().st_size != TRAIN_BYTES:
        raise ValueError("unexpected train file; review the locked split")
    if sequence_id_digest(parquet) != TRAIN_ID_DIGEST:
        raise ValueError("train group order changed; review the locked split")


def sequence_id_digest(parquet):
    digest = hashlib.sha256()
    for group in range(parquet.num_row_groups):
        seq = parquet.read_row_group(group, columns=["seq_ix"], use_threads=False)["seq_ix"][0].as_py()
        digest.update(int(seq).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def make_mlp(input_size=N_FEATURES):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(input_size, 32), torch.nn.Tanh(),
                               torch.nn.Linear(32, 2))


def make_gru(hidden_size=64, input_size=N_FEATURES, num_layers=1, dropout=0.0):
    import torch

    class GRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                                   dropout=dropout if num_layers > 1 else 0.0)
            self.head = torch.nn.Linear(hidden_size, 2)

        def forward(self, values, hidden=None):
            values, hidden = self.rnn(values, hidden)
            return self.head(values), hidden

    return GRU()


def gru_predict(model, x):
    import torch
    outputs = []
    hidden = None
    for start in range(0, len(x), 512):
        out, hidden = model(torch.from_numpy(np.array(x[start:start + 512])).unsqueeze(0), hidden)
        outputs.append(out.squeeze(0).numpy())
    return np.concatenate(outputs)


def prepare():
    import pyarrow.parquet as pq

    sys.path.insert(0, str(ROOT))
    from utils import FEATURE_COLUMNS, TARGET_COLUMNS, validate_sequence

    parquet = pq.ParquetFile(ROOT / "datasets/train.parquet")
    check_train(parquet)
    fit, dev = groups(parquet.num_row_groups)
    CACHE.mkdir(exist_ok=True)
    metadata = {"train_bytes": (ROOT / "datasets/train.parquet").stat().st_size,
                "train_groups": parquet.num_row_groups, "fit": [], "dev": []}
    for name, indices in (("fit", fit), ("dev", dev)):
        x = np.lib.format.open_memmap(CACHE / f"{name}_x.npy", "w+", np.float32,
                                      (len(indices), N_ROWS, N_FEATURES))
        y = np.lib.format.open_memmap(CACHE / f"{name}_y.npy", "w+", np.float32,
                                      (len(indices), N_ROWS, 2))
        for j, group in enumerate(indices):
            table = parquet.read_row_group(int(group), columns=["seq_ix", "step_in_seq", "need_prediction",
                                                                 *FEATURE_COLUMNS, *TARGET_COLUMNS], use_threads=False)
            seq, _ = validate_sequence(table)
            x[j] = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS])
            y[j] = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS])
            if not np.isfinite(x[j]).all() or not np.isfinite(y[j]).all():
                raise ValueError(f"nonfinite train data in group {group}")
            metadata[name].append({"group": int(group), "seq_ix": seq})
        x.flush(); y.flush()
    if {v["seq_ix"] for v in metadata["fit"]} & {v["seq_ix"] for v in metadata["dev"]}:
        raise ValueError("sequence leaked between fit and development")
    (CACHE / "pilot_groups.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print({"fit_groups": len(fit), "dev_groups": len(dev), "cache": str(CACHE)})


def train():
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    x = np.load(CACHE / "fit_x.npy", mmap_mode="r")
    y = np.load(CACHE / "fit_y.npy", mmap_mode="r")
    dx = np.load(CACHE / "dev_x.npy", mmap_mode="r")
    if x.shape != (FIT_GROUPS, N_ROWS, N_FEATURES) or dx.shape != (DEV_GROUPS, N_ROWS, N_FEATURES):
        raise ValueError("unexpected pilot cache")

    # Same sampled rows and weighted squared error for the two memoryless models.
    xs = np.array(x[:, WARMUP::32].reshape(-1, N_FEATURES))
    ys = np.clip(np.array(y[:, WARMUP::32].reshape(-1, 2)), -2, 2)
    # Highly correlated book columns make the normal equations unstable in float32.
    design = np.column_stack((xs.astype(np.float64), np.ones(len(xs))))
    ridge = np.empty((N_FEATURES + 1, 2), np.float32)
    for k in range(2):
        weight = np.sqrt(np.abs(ys[:, k].astype(np.float64)) + 1e-4)
        weighted_design = design * weight[:, None]
        with np.errstate(all="ignore"):
            gram = weighted_design.T @ weighted_design
        gram.flat[::len(gram) + 1] += 100.0
        with np.errstate(all="ignore"):
            rhs = weighted_design.T @ (ys[:, k] * weight)
        if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
            raise ValueError("nonfinite ridge sufficient statistics")
        ridge[:, k] = np.linalg.solve(gram, rhs)
    if not np.isfinite(ridge).all():
        raise ValueError("nonfinite ridge coefficients")
    np.save(CACHE / "ridge_weights.npy", ridge)
    print("ridge fitted", flush=True)

    def weighted_loss(pred, target):
        return ((pred - target).square() * target.abs()).mean()

    mlp = make_mlp()
    opt = torch.optim.Adam(mlp.parameters(), lr=0.001)
    xt, yt = torch.from_numpy(xs), torch.from_numpy(ys)
    for epoch in range(8):
        order = torch.randperm(len(xt))
        for batch in order.split(1024):
            opt.zero_grad()
            loss = weighted_loss(mlp(xt[batch]), yt[batch])
            loss.backward(); opt.step()
        print("mlp epoch", epoch + 1, "loss", loss.item(), flush=True)
    torch.save(mlp.state_dict(), CACHE / "mlp.pt")

    gru = make_gru()
    opt = torch.optim.Adam(gru.parameters(), lr=0.001)
    window = 512
    for epoch in range(3):
        losses = []
        order = rng.permutation(len(x))
        for offset in range(0, len(x), 8):
            indices = order[offset:offset + 8]
            hidden = None
            for start in range(0, N_ROWS, window):
                xb = torch.from_numpy(np.stack([x[i, start:start + window] for i in indices]))
                yb = torch.from_numpy(np.clip(np.stack([y[i, start:start + window] for i in indices]), -2, 2))
                opt.zero_grad()
                pred, hidden = gru(xb, hidden)
                loss_start = WARMUP if start == 0 else 0
                loss = weighted_loss(pred[:, loss_start:], yb[:, loss_start:])
                loss.backward()
                opt.step()
                hidden = hidden.detach()
                losses.append(loss.item())
        print("gru epoch", epoch + 1, "mean loss", float(np.mean(losses)), flush=True)
    torch.save(gru.state_dict(), CACHE / "gru.pt")

    for name in ("ridge", "mlp", "gru"):
        predictions = np.lib.format.open_memmap(CACHE / f"dev_{name}.npy", "w+", np.float32,
                                                 (DEV_GROUPS, N_ROWS, 2))
        predictions[:] = np.nan
        with torch.no_grad():
            for i in range(len(dx)):
                if name == "ridge":
                    with np.errstate(all="ignore"):
                        pred = dx[i] @ ridge[:-1] + ridge[-1]
                elif name == "mlp":
                    pred = mlp(torch.from_numpy(np.array(dx[i]))).numpy()
                else:
                    pred = gru_predict(gru, dx[i])
                predictions[i, WARMUP:] = pred[WARMUP:]
        predictions.flush()
        print(name, "predictions saved", flush=True)


def score():
    sys.path.insert(0, str(ROOT))
    from utils import BlockAccumulator

    dy = np.load(CACHE / "dev_y.npy", mmap_mode="r")
    manifest = json.loads((CACHE / "pilot_groups.json").read_text())
    if len(manifest["dev"]) != DEV_GROUPS:
        raise ValueError("unexpected pilot development groups")
    result = {}
    for name in ("ridge", "mlp", "gru"):
        acc = BlockAccumulator()
        predictions = np.load(CACHE / f"dev_{name}.npy", mmap_mode="r")
        for i in range(len(dy)):
            pred = predictions[i]
            mask = np.arange(N_ROWS) >= WARMUP
            acc.add(dy[i], pred, mask)
        result[name] = acc.result()
        print(name, result[name], flush=True)
    (CACHE / "pilot_scores.json").write_text(json.dumps(result, indent=2) + "\n")


def score_all():
    """Score frozen pilot models on every held-out train sequence."""
    import pyarrow.parquet as pq
    import torch

    sys.path.insert(0, str(ROOT))
    from utils import BlockAccumulator, FEATURE_COLUMNS, TARGET_COLUMNS, validate_sequence

    torch.set_num_threads(1)
    parquet = pq.ParquetFile(ROOT / "datasets/train.parquet")
    check_train(parquet)
    models = load_frozen_models()
    accumulators = {name: BlockAccumulator() for name in ("ridge", "mlp", "gru")}
    mask = np.arange(N_ROWS) >= WARMUP
    with torch.no_grad():
        for n, group in enumerate(range(0, parquet.num_row_groups, 10), 1):
            table = parquet.read_row_group(group, columns=["seq_ix", "step_in_seq", "need_prediction",
                                                             *FEATURE_COLUMNS, *TARGET_COLUMNS], use_threads=False)
            validate_sequence(table)
            x = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
            y = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
            for name, values in predict_frozen(models, x).items():
                pred = np.full((N_ROWS, 2), np.nan, np.float32)
                pred[WARMUP:] = values[WARMUP:]
                accumulators[name].add(y, pred, mask)
            if n % 100 == 0:
                print("scored", n, "development sequences", flush=True)
    result = {name: acc.result() for name, acc in accumulators.items()}
    print(json.dumps(result, indent=2), flush=True)
    (CACHE / "all_dev_scores.json").write_text(json.dumps(result, indent=2) + "\n")


def load_frozen_models():
    import torch
    ridge = np.load(CACHE / "ridge_weights.npy")
    mlp = make_mlp()
    mlp.load_state_dict(torch.load(CACHE / "mlp.pt", map_location="cpu", weights_only=True))
    gru = make_gru()
    gru.load_state_dict(torch.load(CACHE / "gru.pt", map_location="cpu", weights_only=True))
    return ridge, mlp.eval(), gru.eval()


def predict_frozen(models, x):
    import torch
    ridge, mlp, gru = models
    with np.errstate(all="ignore"), torch.no_grad():
        return {"ridge": x @ ridge[:-1] + ridge[-1],
                "mlp": mlp(torch.from_numpy(x)).numpy(),
                "gru": gru_predict(gru, x)}


def score_oos():
    """Evaluate all three frozen pilot models on the official masked OOS set."""
    import pyarrow.parquet as pq
    import torch

    sys.path.insert(0, str(ROOT))
    from utils import BlockAccumulator, FEATURE_COLUMNS, TARGET_COLUMNS, validate_sequence

    torch.set_num_threads(1)
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    if parquet.num_row_groups != 1_873 or (ROOT / "datasets/valid.parquet").stat().st_size != VALID_BYTES:
        raise ValueError("unexpected valid file; review the locked OOS set")
    if sequence_id_digest(parquet) != VALID_ID_DIGEST:
        raise ValueError("valid group order changed; review the locked OOS set")
    models = load_frozen_models()
    accumulators = {name: BlockAccumulator() for name in ("ridge", "mlp", "gru")}
    with torch.no_grad():
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group, columns=["seq_ix", "step_in_seq", "need_prediction",
                                                            "is_scored", *FEATURE_COLUMNS, *TARGET_COLUMNS],
                                           use_threads=False)
            _, need = validate_sequence(table)
            x = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
            y = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
            mask = table["is_scored"].to_numpy().astype(bool) & need
            for name, values in predict_frozen(models, x).items():
                predictions = np.full((N_ROWS, 2), np.nan, np.float32)
                predictions[WARMUP:] = values[WARMUP:]
                accumulators[name].add(y, predictions, mask)
            if (group + 1) % 200 == 0:
                print("scored", group + 1, "OOS sequences", flush=True)
    result = {name: acc.result() for name, acc in accumulators.items()}
    print(json.dumps(result, indent=2), flush=True)
    (CACHE / "oos_scores.json").write_text(json.dumps(result, indent=2) + "\n")


def selfcheck():
    import torch
    fit, dev = groups(10_607)
    assert len(fit) == FIT_GROUPS and len(dev) == DEV_GROUPS
    assert set(fit).isdisjoint(dev)
    assert all(i % 10 for i in fit) and all(i % 10 == 0 for i in dev)
    torch.set_num_threads(1)
    model = make_gru().eval()
    x = np.random.default_rng(SEED).normal(size=(1_024, N_FEATURES)).astype(np.float32)
    with torch.no_grad():
        full = model(torch.from_numpy(x).unsqueeze(0))[0].squeeze(0).numpy()
        chunked = gru_predict(model, x)
    assert np.allclose(full, chunked, atol=1e-5)
    print("split and GRU state continuity checks passed")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"prepare", "train", "score", "score_all", "score_oos", "selfcheck"}:
        raise SystemExit("usage: pilot.py {prepare|train|score|score_all|score_oos|selfcheck}")
    globals()[sys.argv[1]]()
