"""Stack of minimal gated linear recurrences (minGRU) for the online GRU pipeline.

Each layer computes, from its input only, an update gate z_t and a candidate c_t,
then h_t = (1 - z_t) * h_{t-1} + z_t * tanh(c_t). Because the gates do not read
h_{t-1}, a whole chunk is evaluated with a blocked log-space scan (linear_scan;
about 1e-4 relative rounding against the step-by-step recurrence used at
inference). Layers are pre-norm residual blocks, so depth is cheap at inference:
per step and layer one (2D x D) and one (D x D) product.

The module follows make_gru's interface: forward(values, hidden) returns
(predictions, hidden) with hidden of shape (layers, batch, width).

    python research/mingru.py --selfcheck
"""
from __future__ import annotations

import argparse

import torch


def linear_scan(a, b, h0, block=128):
    """h_t = a_t * h_{t-1} + b_t over dim 1, in blocks of `block` steps.

    Inside a block the recurrence is evaluated in log space: with L the cumulative
    log of a, h_t = exp(L_t) * (h_0 + sum_{s<=t} exp(-L_s) b_s); the signed b is
    split into positive and negative parts, each summed with logcumsumexp. The
    block carry restarts L at zero, which bounds float32 rounding."""
    batch, steps, width = a.shape
    h = torch.zeros(batch, width, dtype=a.dtype, device=a.device) if h0 is None else h0
    log_a = torch.log(a.clamp_min(1e-6))
    outputs = []
    for start in range(0, steps, block):
        L = torch.cumsum(log_a[:, start:start + block], 1)
        bb = b[:, start:start + block]
        pos = torch.exp(L + torch.logcumsumexp(torch.log(torch.relu(bb).clamp_min(1e-30)) - L, 1))
        neg = torch.exp(L + torch.logcumsumexp(torch.log(torch.relu(-bb).clamp_min(1e-30)) - L, 1))
        hb = pos - neg + torch.exp(L) * h[:, None, :]
        outputs.append(hb)
        h = hb[:, -1]
    return torch.cat(outputs, 1)


class MinGRULayer(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.gate = torch.nn.Linear(width, 2 * width)
        self.out = torch.nn.Linear(width, width)

    def forward(self, u, h0):
        z_logit, candidate = self.gate(self.norm(u)).chunk(2, -1)
        z = torch.sigmoid(z_logit)
        h = linear_scan(1 - z, z * torch.tanh(candidate), h0)
        return self.out(torch.nn.functional.gelu(h)), h[:, -1]


class MinGRUStack(torch.nn.Module):
    def __init__(self, width=96, layers=3, input_size=112):
        super().__init__()
        self.inp = torch.nn.Linear(input_size, width)
        self.blocks = torch.nn.ModuleList(MinGRULayer(width) for _ in range(layers))
        self.head = torch.nn.Linear(width, 2)

    def forward(self, values, hidden=None):
        u = self.inp(values)
        last = []
        for i, block in enumerate(self.blocks):
            out, h = block(u, None if hidden is None else hidden[i])
            u = u + out
            last.append(h)
        return self.head(u), torch.stack(last)


def selfcheck():
    torch.manual_seed(0)
    a, b = torch.rand(2, 77, 5) * 0.98 + 0.01, torch.randn(2, 77, 5)
    h0 = torch.randn(2, 5)
    ref, h = [], h0
    for t in range(77):
        h = a[:, t] * h + b[:, t]
        ref.append(h)
    assert torch.allclose(linear_scan(a, b, h0, block=16), torch.stack(ref, 1), atol=1e-4)
    model = MinGRUStack(8, 3).eval()
    x = torch.randn(2, 70, 112)
    full = model(x)[0]
    first, state = model(x[:, :33])
    second, _ = model(x[:, 33:], state)
    assert torch.allclose(full, torch.cat((first, second), 1), atol=1e-4)
    print("blocked scan matches the sequential recurrence; chunked stateful inference matches full")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", action="store_true")
    if parser.parse_args().selfcheck:
        selfcheck()
