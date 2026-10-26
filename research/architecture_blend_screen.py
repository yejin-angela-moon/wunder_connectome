"""Score a structurally different recurrent checkpoint inside the strong proxy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from ensemble_screen import MEMBERS, load as load_cached, wp
from ensdistill import make_student
from maskft import load as load_base
from maskmix import normalize, read_valid
from pilot import CACHE, ROOT

OUT = CACHE / "architecture_blend"
BLENDS = (0.05, 0.1, 0.2, 0.3)


def predict(model, parquet, groups, scaler, device, batch=16):
    predictions, targets = [], []
    model = model.to(device).eval()
    with torch.no_grad():
        for offset in range(0, len(groups), batch):
            x, y, mask = read_valid(parquet, groups[offset:offset + batch])
            output = model(torch.from_numpy(normalize(x, scaler)).to(device))[0].cpu().numpy()
            predictions.extend([p[m] for p, m in zip(output, mask)])
            targets.extend([t[m] for t, m in zip(y, mask)])
            if offset and offset % 160 == 0:
                print("scored groups", offset, flush=True)
    return np.concatenate(predictions), np.concatenate(targets)


def selfcheck():
    torch.manual_seed(1)
    model = make_student("mingru", 8, 2).eval()
    x = torch.randn(2, 70, 112)
    full = model(x)[0]
    first, state = model(x[:, :31])
    second = model(x[:, 31:], state)[0]
    assert torch.allclose(full, torch.cat((first, second), 1), atol=1e-4)
    print("candidate recurrent full/chunk continuity check passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cell", choices=("gru", "lstm", "mingru"), default="mingru")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--half", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = make_student(args.cell, args.width, args.layers)
    model.load_state_dict(state)
    _, scaler = load_base("v1")
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    groups = np.arange(parquet.num_row_groups)
    groups = groups[groups % 2 != args.half]
    candidate, y = predict(model, parquet, groups, scaler, args.device)
    cached, expected, _ = load_cached(args.half)
    if not np.array_equal(y, expected):
        raise ValueError("candidate rows do not align with cached strong proxy")
    names = list(MEMBERS)
    members = [names.index(name) for name in ("c71", "c72", "split")]
    strong = np.einsum("rmt,m->rt", cached[:, members], (0.4, 0.4, 0.2))
    scores = {"candidate": wp(y, candidate), "strong": wp(y, strong)}
    for weight in BLENDS:
        scores[f"candidate_{weight:g}"] = wp(y, (1 - weight) * strong + weight * candidate)
    best_name = max((name for name in scores if name.startswith("candidate_")),
                    key=scores.__getitem__)
    result = {"checkpoint": str(args.checkpoint), "half": args.half,
              "scores": scores, "best": best_name,
              "gain": scores[best_name] - scores["strong"],
              "passes_gate": scores[best_name] >= scores["strong"] + 0.002}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.checkpoint.stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
