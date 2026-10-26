"""Dependency-light one-step GRU inference with a PyTorch parity check."""
from __future__ import annotations

import numpy as np


class NumpyGRU:
    def __init__(self, weights):
        self.w_ih = weights["rnn.weight_ih_l0"]
        self.w_hh = weights["rnn.weight_hh_l0"]
        self.b_ih = weights["rnn.bias_ih_l0"]
        self.b_hh = weights["rnn.bias_hh_l0"]
        self.w_out = weights["head.weight"]
        self.b_out = weights["head.bias"]
        self.width = self.w_hh.shape[1]
        self.hidden = np.zeros(self.width, np.float32)

    def reset(self):
        self.hidden.fill(0)

    def step(self, x):
        x = np.asarray(x, np.float32)
        if x.shape != (self.w_ih.shape[1],):
            raise ValueError("wrong feature count")
        with np.errstate(all="ignore"):
            input_gates = self.w_ih @ x + self.b_ih
            hidden_gates = self.w_hh @ self.hidden + self.b_hh
            n = self.width
            reset = 1 / (1 + np.exp(-np.clip(input_gates[:n] + hidden_gates[:n], -50, 50)))
            update = 1 / (1 + np.exp(-np.clip(input_gates[n:2*n] + hidden_gates[n:2*n], -50, 50)))
            candidate = np.tanh(input_gates[2*n:] + reset * hidden_gates[2*n:])
            self.hidden = (1 - update) * candidate + update * self.hidden
            output = self.w_out @ self.hidden + self.b_out
        if not np.isfinite(output).all():
            raise ValueError("nonfinite GRU prediction")
        return output


def selfcheck(checkpoint):
    import torch
    from pilot import make_gru
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    width = state["rnn.weight_hh_l0"].shape[1]
    model = make_gru(width).eval()
    model.load_state_dict(state)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(1_024, 112)).astype(np.float32)
    numpy_model = NumpyGRU({name: value.numpy() for name, value in state.items()})
    actual = np.asarray([numpy_model.step(row) for row in x])
    with torch.no_grad():
        expected = model(torch.from_numpy(x).unsqueeze(0))[0].squeeze(0).numpy()
    difference = float(np.max(np.abs(actual - expected)))
    assert difference < 1e-5, difference
    print("NumPy GRU matches PyTorch over 1,024 ordered steps; max error", difference)


if __name__ == "__main__":
    import argparse
    from pilot import CACHE
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CACHE / "scaled_gru.pt")
    args = parser.parse_args()
    selfcheck(args.checkpoint)
