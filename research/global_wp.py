"""Streaming Global Weighted Pearson from the current competition web specification."""
from __future__ import annotations

import numpy as np


class GlobalWP:
    def __init__(self):
        # Per target: sum(w), sum(w*y), sum(w*p), sum(w*y*y), sum(w*p*p), sum(w*y*p).
        self.sums = np.zeros((2, 6), np.float64)
        self.rows = 0

    def add(self, targets, predictions, mask):
        y = np.asarray(targets, np.float32)
        p = np.asarray(predictions, np.float32)
        mask = np.asarray(mask, bool)
        if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 2 or mask.shape != (len(y),):
            raise ValueError("expected equal (rows, 2) targets/predictions and a row mask")
        y, p = y[mask], p[mask]
        if not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError("nonfinite selected target or prediction")
        y = np.clip(y.astype(np.float64), -2, 2)
        p = np.clip(p.astype(np.float64), -2, 2)
        w = np.abs(y)
        self.sums += np.stack((w.sum(axis=0), (w*y).sum(axis=0), (w*p).sum(axis=0),
                              (w*y*y).sum(axis=0), (w*p*p).sum(axis=0),
                              (w*y*p).sum(axis=0)), axis=1)
        self.rows += len(y)

    def result(self):
        if not self.rows:
            raise ValueError("no selected rows")
        sw, sy, sp, syy, spp, syp = self.sums.T
        safe_sw = np.where(sw > 0, sw, 1)
        vy = syy - sy*sy/safe_sw
        vp = spp - sp*sp/safe_sw
        cov = syp - sy*sp/safe_sw
        rho = np.divide(cov, np.sqrt(vy*vp), out=np.zeros(2), where=(sw > 0) & (vy > 0) & (vp > 0))
        rho = np.clip(rho, -1, 1)
        return {"t0": float(rho[0]), "t1": float(rho[1]),
                "weighted_pearson": float(rho.mean()), "selected_rows": self.rows,
                "aggregation": "global clipped Weighted Pearson over all selected rows"}


def selfcheck():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils import weighted_pearson

    rng = np.random.default_rng(7)
    y = rng.normal(size=(211, 2)).astype(np.float32)
    p = (y * 0.3 + rng.normal(size=y.shape)).astype(np.float32)
    mask = rng.random(len(y)) > 0.2
    acc = GlobalWP()
    acc.add(y[:100], p[:100], mask[:100])
    acc.add(y[100:], p[100:], mask[100:])
    result = acc.result()
    for j, key in enumerate(("t0", "t1")):
        assert np.isclose(result[key], weighted_pearson(y[mask, j], p[mask, j]), atol=1e-12)
    assert result["selected_rows"] == mask.sum()
    print("streaming Global WP matches direct pooled WP")


if __name__ == "__main__":
    selfcheck()
