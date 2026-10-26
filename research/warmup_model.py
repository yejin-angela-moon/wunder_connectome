"""Reusable GRU whose recurrent state is conditioned on rows 0--98."""
from __future__ import annotations

import torch

from pilot import N_FEATURES, WARMUP


class WarmupGRU(torch.nn.Module):
    def __init__(self, width=128, context_size=16):
        super().__init__()
        self.context_net = torch.nn.Sequential(
            torch.nn.Linear(3 * N_FEATURES, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, context_size),
            torch.nn.Tanh(),
        )
        self.rnn = torch.nn.GRU(N_FEATURES + context_size, width, batch_first=True)
        self.head = torch.nn.Linear(width, 2)

    def context(self, values):
        warmup = values[:, :WARMUP]
        summary = torch.cat((warmup.mean(1), warmup.std(1, unbiased=False),
                             (warmup[:, 1:] - warmup[:, :-1]).abs().mean(1)), -1)
        return self.context_net(summary)

    def forward_context(self, values, context, hidden=None):
        repeated = context[:, None].expand(-1, values.shape[1], -1)
        states, hidden = self.rnn(torch.cat((values, repeated), -1), hidden)
        return self.head(states), hidden

    def forward(self, values, state=None):
        if state is None:
            if values.shape[1] < WARMUP:
                raise ValueError("first recurrent chunk must contain the complete warm-up")
            context, hidden = self.context(values), None
        else:
            context, hidden = state
        prediction, hidden = self.forward_context(values, context, hidden)
        return prediction, (context, hidden)
