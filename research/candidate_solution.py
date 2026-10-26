"""NumPy-only competition callback; package this as solution.py with model.npz."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from numpy_gru import NumpyGRU


class PredictionModel:
    def __init__(self, model_path=None):
        source = Path(model_path) if model_path is not None else Path(__file__).resolve().parent / "model.npz"
        weights = np.load(source, allow_pickle=False)
        self.kind = str(weights["kind"])
        if self.kind == "ridge_lag":
            self.ridge = weights["weights"]
        elif self.kind == "mlp_norm":
            self.w0, self.b0 = weights["w0"], weights["b0"]
            self.w1, self.b1 = weights["w1"], weights["b1"]
            self.mean, self.std = weights["mean"], weights["std"]
        elif self.kind in ("gru96", "gru96_norm"):
            self.gru = NumpyGRU(weights)
            if self.kind == "gru96_norm":
                self.mean, self.std = weights["mean"], weights["std"]
        elif self.kind == "gru96_ensemble":
            self.grus = [NumpyGRU({key[3:]: weights[key] for key in weights.files
                                   if key.startswith(f"m{index}_")})
                         for index in range(3)]
            self.mean, self.std = weights["mean"], weights["std"]
        else:
            raise ValueError(f"unknown model kind: {self.kind}")
        weights.close()
        self.seq_ix = None
        self.previous_step = None
        self.previous_state = None

    def predict(self, data_point):
        seq_ix = int(data_point.seq_ix)
        step = int(data_point.step_in_seq)
        if seq_ix != self.seq_ix:
            if step != 0:
                raise ValueError("new sequence must start at step zero")
            self.seq_ix = seq_ix
            self.previous_step = None
            self.previous_state = None
            if self.kind in ("gru96", "gru96_norm"):
                self.gru.reset()
            elif self.kind == "gru96_ensemble":
                for gru in self.grus:
                    gru.reset()
        if self.previous_step is not None and step != self.previous_step + 1:
            raise ValueError("rows must arrive in order")
        self.previous_step = step
        state = np.asarray(data_point.state, np.float32)
        if state.shape != (112,) or not np.isfinite(state).all():
            raise ValueError("expected 112 finite features")
        with np.errstate(all="ignore"):
            if self.kind == "ridge_lag":
                change = np.zeros_like(state) if self.previous_state is None else state - self.previous_state
                value = np.concatenate((state, change)) @ self.ridge[:-1] + self.ridge[-1]
                self.previous_state = state.copy()
            elif self.kind == "mlp_norm":
                normalized = np.clip((state - self.mean) / self.std, -10, 10)
                value = self.w1 @ np.tanh(self.w0 @ normalized + self.b0) + self.b1
            elif self.kind in ("gru96", "gru96_norm"):
                value = self.gru.step(np.clip((state - self.mean) / self.std, -10, 10)
                                      if self.kind == "gru96_norm" else state)
            else:
                normalized = np.clip((state - self.mean) / self.std, -10, 10)
                value = (self.grus[0].step(normalized) + self.grus[1].step(state)
                         + self.grus[2].step(state)) / 3
        if not np.isfinite(value).all():
            raise ValueError("nonfinite model prediction")
        return None if not data_point.need_prediction else np.asarray(value, np.float32)
