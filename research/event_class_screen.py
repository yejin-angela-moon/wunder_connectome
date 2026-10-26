"""Locked pilot for a causal next-price-event classification auxiliary head."""
from __future__ import annotations

from collections import Counter
import argparse
import copy
import json

import numpy as np
import torch

from event_pretrain import BATCH, CHUNK, normalize, weighted_loss
from global_wp import GlobalWP
from maskft import detach_state
from pilot import CACHE, DEV_GROUPS, FIT_GROUPS, N_FEATURES, N_ROWS, SEED, WARMUP, make_gru

OUT = CACHE / "event_class_screen"
N_CLASSES = 8


def patterns(values):
    price = values[:, :22]
    starts = np.flatnonzero(np.r_[True, np.any(price[1:] != price[:-1], axis=1)])
    return [np.sign(row).astype(np.int8).tobytes() for row in np.diff(price[starts], axis=0)]


def mapping_from_fit(x):
    counts = Counter(pattern for values in x for pattern in patterns(values))
    mapping = {pattern: index for index, (pattern, _) in enumerate(counts.most_common(N_CLASSES))}
    return mapping, sum(count for _, count in counts.most_common(N_CLASSES)) / sum(counts.values())


def event_labels(values, mapping):
    labels = np.full(values.shape[:2], len(mapping), np.int64)
    valid = np.zeros(values.shape[:2], bool)
    rows = np.arange(values.shape[1])
    for i, sequence in enumerate(values):
        price = sequence[:, :22]
        starts = np.flatnonzero(np.r_[True, np.any(price[1:] != price[:-1], axis=1)])
        segment = np.searchsorted(starts, rows, side="right") - 1
        keep = segment + 1 < len(starts)
        delta = price[starts[segment[keep] + 1]] - price[keep]
        labels[i, keep] = np.fromiter(
            (mapping.get(np.sign(row).astype(np.int8).tobytes(), len(mapping)) for row in delta),
            np.int64, count=len(delta))
        valid[i, keep] = True
    return labels, valid


class EventClassGRU(torch.nn.Module):
    def __init__(self, base, classes=N_CLASSES + 1):
        super().__init__()
        self.rnn, self.head = base.rnn, base.head
        self.event_head = torch.nn.Linear(self.rnn.hidden_size, classes)
        self.event_target = torch.nn.Linear(classes, 2, bias=False)
        torch.nn.init.zeros_(self.event_target.weight)

    def forward(self, values, hidden=None):
        states, hidden = self.rnn(values, hidden)
        logits = self.event_head(states)
        prediction = self.head(states) + self.event_target(logits.softmax(-1))
        return prediction, hidden, logits


def train(name, model, x, y, dx, dy, mean, std, mapping, device, epochs, auxiliary):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(SEED)
    history = [evaluate(model, dx, dy, mean, std, mapping, device)]
    for epoch in range(epochs):
        if epoch == 3:
            for group in optimizer.param_groups:
                group["lr"] = 3e-4
        losses = []
        order = rng.permutation(len(x))
        for offset in range(0, len(x), BATCH):
            indices = order[offset:offset + BATCH]
            values = normalize(np.array(x[indices]), mean, std)
            targets = np.clip(np.array(y[indices]), -2, 2).astype(np.float32)
            labels, valid = event_labels(values, mapping) if auxiliary else (None, None)
            if auxiliary:
                valid[:, :WARMUP] = False
            hidden = None
            for start in range(0, N_ROWS, CHUNK):
                stop = min(start + CHUNK, N_ROWS)
                xb = torch.from_numpy(np.ascontiguousarray(values[:, start:stop])).to(device)
                yb = torch.from_numpy(np.ascontiguousarray(targets[:, start:stop])).to(device)
                optimizer.zero_grad()
                output = model(xb, hidden)
                prediction, hidden = output[:2]
                begin = WARMUP if start == 0 else 0
                loss = weighted_loss(prediction[:, begin:], yb[:, begin:])
                if auxiliary:
                    lb = torch.from_numpy(np.ascontiguousarray(labels[:, start:stop])).to(device)
                    vb = torch.from_numpy(np.ascontiguousarray(valid[:, start:stop])).to(device)
                    loss = loss + auxiliary * torch.nn.functional.cross_entropy(output[2][vb], lb[vb])
                loss.backward(); optimizer.step(); hidden = detach_state(hidden)
                losses.append(loss.item())
        history.append(evaluate(model, dx, dy, mean, std, mapping, device))
        print(name, "epoch", epoch + 1, "loss", round(float(np.mean(losses)), 6),
              history[-1], flush=True)
    return history


def evaluate(model, x, y, mean, std, mapping, device):
    metric = GlobalWP()
    correct = total = other = 0
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(x), BATCH):
            values = normalize(np.array(x[offset:offset + BATCH]), mean, std)
            labels, valid = event_labels(values, mapping)
            valid[:, :WARMUP] = False
            hidden, predictions = None, []
            for start in range(0, N_ROWS, CHUNK):
                stop = min(start + CHUNK, N_ROWS)
                xb = torch.from_numpy(np.ascontiguousarray(values[:, start:stop])).to(device)
                output = model(xb, hidden)
                prediction, hidden = output[:2]
                predictions.append(prediction.cpu().numpy())
                if len(output) == 3:
                    vb = valid[:, start:stop]
                    guessed = output[2].argmax(-1).cpu().numpy()
                    correct += int((guessed[vb] == labels[:, start:stop][vb]).sum())
                    total += int(vb.sum())
                    other += int((labels[:, start:stop][vb] == len(mapping)).sum())
            predictions = np.concatenate(predictions, axis=1)
            for i in range(len(values)):
                metric.add(y[offset + i], predictions[i], np.arange(N_ROWS) >= WARMUP)
    model.train()
    result = metric.result()
    result.update({"event_accuracy": correct / total if total else None,
                   "event_other_rate": other / total if total else None})
    return result


def selfcheck():
    values = np.zeros((2, 80, N_FEATURES), np.float32)
    values[:, :, :22] = (np.arange(80) // 10)[None, :, None]
    pattern = np.ones(22, np.int8).tobytes()
    labels, valid = event_labels(values, {pattern: 0})
    assert valid[:, :70].all() and not valid[:, 70:].any() and np.all(labels[:, :70] == 0)
    torch.manual_seed(1)
    base = make_gru(8)
    model = EventClassGRU(copy.deepcopy(base), 2).eval()
    x = torch.randn(2, 19, N_FEATURES)
    full = model(x)[0]
    first, state = model(x[:, :7])[:2]
    second = model(x[:, 7:], state)[0]
    assert torch.allclose(full, torch.cat((first, second), 1), atol=1e-6)
    assert torch.allclose(full, base(x)[0], atol=1e-7)
    print("event labels and zero-initialized causal correction checks passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--auxiliary", type=float, default=0.1)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck(); return
    torch.set_num_threads(1)
    x = np.load(CACHE / "fit_x.npy", mmap_mode="r")
    y = np.load(CACHE / "fit_y.npy", mmap_mode="r")
    dx = np.load(CACHE / "dev_x.npy", mmap_mode="r")
    dy = np.load(CACHE / "dev_y.npy", mmap_mode="r")
    if x.shape != (FIT_GROUPS, N_ROWS, N_FEATURES) or dx.shape != (DEV_GROUPS, N_ROWS, N_FEATURES):
        raise ValueError("unexpected locked pilot cache")
    sampled = np.asarray(x[:, WARMUP::32], np.float64)
    mean = sampled.mean((0, 1)).astype(np.float32)
    std = sampled.std((0, 1)).astype(np.float32)
    std[std < 1e-6] = 1
    mapping, coverage = mapping_from_fit(x)
    torch.manual_seed(SEED)
    initial = make_gru(96)
    control = copy.deepcopy(initial).to(args.device)
    candidate = EventClassGRU(copy.deepcopy(initial)).to(args.device)
    control_history = train("control", control, x, y, dx, dy, mean, std, mapping,
                            args.device, args.epochs, 0)
    candidate_history = train("event_class", candidate, x, y, dx, dy, mean, std, mapping,
                              args.device, args.epochs, args.auxiliary)
    control_best = max(row["weighted_pearson"] for row in control_history[1:])
    candidate_best = max(row["weighted_pearson"] for row in candidate_history[1:])
    result = {"classes": N_CLASSES, "fit_coverage": coverage, "auxiliary": args.auxiliary,
              "control": control_history, "candidate": candidate_history,
              "control_best": control_best, "candidate_best": candidate_best,
              "gain": candidate_best - control_best,
              "passes_gate": candidate_best >= control_best + 0.005}
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"aux{args.auxiliary:g}"
    (OUT / f"pilot_{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(candidate.cpu().state_dict(), OUT / f"pilot_{tag}.pt")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
