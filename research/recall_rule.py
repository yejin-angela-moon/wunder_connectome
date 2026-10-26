"""Rule-based recall of the same book state's earlier prediction (no training).

At each row, the i0 price state (22 book price columns) and the last row of its previous visit are known
causally. When that visit ended at least GAP rows ago, the output blends in A of the prediction made at
the visit's last row: (1 - A) * current + A * recalled. Rows whose state has no earlier visit, or a recent
one, keep the current prediction.
"""
from __future__ import annotations

import numpy as np

GAP, A = 500, 0.1


def previous_visit(raw):
    """Last row of the previous visit to each row's i0 price state (-1 if none); causal."""
    key = np.unique(raw[:, :22], axis=0, return_inverse=True)[1].ravel()
    start = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1]
    end = np.r_[start[1:] - 1, len(key) - 1]
    last, pointer = {}, np.full(len(key), -1)
    for s, e in zip(start, end):
        pointer[s:e + 1] = last.get(key[s], -1)
        last[key[s]] = e
    return pointer


def apply(prediction, pointer, gap=GAP, a=A):
    rows = np.arange(len(pointer))
    use = (pointer >= 0) & (rows - pointer >= gap)
    out = prediction.copy()
    out[use] = (1 - a) * prediction[use] + a * prediction[pointer[use]]
    return out


def selfcheck():
    raw = np.zeros((3000, 112), np.float32)
    raw[1000:2000, 0] = 1                                     # state A rows 0-999 and 2000-2999, state B between
    pointer = previous_visit(raw)
    assert pointer[0] == -1 and pointer[1500] == -1 and pointer[2000] == 999 and pointer[2999] == 999
    prediction = np.arange(3000, dtype=np.float64)[:, None] * np.ones((1, 2))
    out = apply(prediction, pointer)
    assert np.allclose(out[2000], 0.9 * 2000 + 0.1 * 999) and np.allclose(out[:2000], prediction[:2000])
    changed = raw.copy(); changed[2500:, 0] = 5                  # the future cannot move the past pointers
    assert np.array_equal(previous_visit(changed)[:2500], pointer[:2500])
    print("recall rule: pointers causal, blend only after long gaps")


if __name__ == "__main__":
    selfcheck()
