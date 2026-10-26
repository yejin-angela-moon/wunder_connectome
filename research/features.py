"""Causal event-time features of the i0 price block.

Targets are constant between changes of the i0 price-like block (columns
0-21), so the last price transitions carry most of the new information. For
each row this returns, all from the current and earlier rows:

- the change of the price block at the most recent transition (22),
- the change at the transition before that (22),
- log1p(rows since the last transition) and log1p(transition count) / 10.

`EventState` computes the same values one row at a time for the callback.
"""
from __future__ import annotations

import numpy as np

N_EVENT = 46
# RMS of the unscaled 44 difference features over 600 random train sequences
# (every 10th row) was 0.188; dividing by it keeps them near unit scale.
DIFF_SCALE = np.float32(0.188)


def event_features(raw):
    """(T, 112) raw rows of one sequence -> (T, 46) float32 event features."""
    state = raw[:, :22]
    change = np.r_[True, np.any(state[1:] != state[:-1], axis=1)]
    starts = np.flatnonzero(change)
    segment = np.cumsum(change) - 1
    seg_state = state[starts]
    prev = np.r_[seg_state[:1], seg_state[:-1]]
    prev2 = np.r_[seg_state[:1], prev[:-1]]
    out = np.empty((len(raw), N_EVENT), np.float32)
    out[:, :22] = (seg_state - prev)[segment] / DIFF_SCALE
    out[:, 22:44] = (prev - prev2)[segment] / DIFF_SCALE
    out[:, 44] = np.log1p(np.arange(len(raw)) - starts[segment])
    out[:, 45] = np.log1p(segment) / 10
    return out


class EventState:
    """Row-by-row equivalent of event_features for one sequence."""

    def __init__(self):
        self.state = None

    def step(self, raw):
        price = raw[:22]
        if self.state is None:
            self.state = price.copy()
            self.prev = self.prev2 = self.state
            self.since, self.count = 0, 0
        elif np.any(price != self.state):
            self.prev2, self.prev, self.state = self.prev, self.state, price.copy()
            self.since, self.count = 0, self.count + 1
        else:
            self.since += 1
        out = np.empty(N_EVENT, np.float32)
        out[:22] = (self.state - self.prev) / DIFF_SCALE
        out[22:44] = (self.prev - self.prev2) / DIFF_SCALE
        out[44] = np.log1p(self.since)
        out[45] = np.log1p(self.count) / 10
        return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    levels = rng.normal(size=(6, 22)).astype(np.float32)
    raw = np.zeros((300, 112), np.float32)
    raw[:, :22] = levels[np.minimum(np.arange(300) // 37, 5)]
    raw[:, 22:] = rng.normal(size=(300, 90))
    batch = event_features(raw)
    state = EventState()
    online = np.stack([state.step(row) for row in raw])
    assert np.allclose(batch, online), np.abs(batch - online).max()
    assert np.isclose(batch[36, 44], np.log1p(36)) and batch[37, 44] == 0 and np.isclose(batch[37, 45], np.log1p(1) / 10)
    print("event features: batch and row-wise versions agree")
