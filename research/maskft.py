"""Fine-tune a frozen full-FIT GRU on validation sequences with scored rows emphasized.

Validation row groups are split by parity. `run --half 0` fits on even groups
and scores the odd groups' official scored rows after each epoch, and vice
versa. `--unscored 1` is the uniform-weight control; smaller values
down-weight rows the official mask does not select. `final` fits on every
validation group with a fixed epoch count chosen from the two half runs.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pyarrow.parquet as pq
import torch

from features import event_features
from global_wp import GlobalWP
from oos_global import read_group as read_valid_group
from pilot import CACHE, N_ROWS, ROOT, WARMUP, make_gru
from stack import FULL, MODELS

OUT = CACHE / "maskft"
CHUNK = 512
BATCH = 32


class CausalConvGRU(torch.nn.Module):
    """Small causal convolutional residual in front of an existing GRU."""

    def __init__(self, base, width, kernel=5):
        super().__init__()
        inputs = base.rnn.input_size
        self.rnn, self.head = base.rnn, base.head
        self.input = torch.nn.Linear(inputs, width)
        self.convolution = torch.nn.Conv1d(width, width, kernel, groups=width)
        self.output = torch.nn.Linear(width, inputs)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)
        self.context = kernel - 1

    def forward(self, values, state=None):
        conv_state, hidden = (None, None) if state is None else state
        encoded = torch.tanh(self.input(values)).transpose(1, 2)
        if conv_state is None:
            conv_state = torch.zeros(encoded.shape[0], encoded.shape[1], self.context,
                                     dtype=encoded.dtype, device=encoded.device)
        joined = torch.cat((conv_state, encoded), dim=2)
        mixed = torch.tanh(self.convolution(joined)).transpose(1, 2)
        outputs, hidden = self.rnn(values + self.output(mixed), hidden)
        return self.head(outputs), (joined[:, :, -self.context:], hidden)


class IntervalResidual(torch.nn.Module):
    """Nonlinear correction from causal statistics that reset at i0 price events."""

    def __init__(self, base, width):
        super().__init__()
        self.base = base
        self.interval = torch.nn.Sequential(torch.nn.Linear(157, width), torch.nn.Tanh(),
                                            torch.nn.Linear(width, 2))
        torch.nn.init.zeros_(self.interval[-1].weight)
        torch.nn.init.zeros_(self.interval[-1].bias)

    def forward(self, values, interval, state=None):
        prediction, state = self.base(values, state)
        return prediction + self.interval(interval), state


def detach_state(state):
    if isinstance(state, torch.Tensor):
        return state.detach()
    return tuple(detach_state(value) for value in state)


def load(name):
    """A named full-FIT GRU, or a standardized-input student under cache/distill_mps.

    Returns (model, scaler); scaler is None for raw-input models, else
    (mean, std, uses_event_features)."""
    with np.load(FULL / "gru96_norm5ep/gru_standardizer.npz") as source:
        mean, std = source["mean"], source["std"]
    path, standardized = MODELS.get(name, (CACHE / "distill_mps" / f"{name}.pt", True))
    state = torch.load(path, map_location="cpu", weights_only=True)
    from distill_mps import build
    model = build(state)
    model.load_state_dict(state)
    inputs = (state["rnn.weight_ih_l0"] if "rnn.weight_ih_l0" in state else state["inp.weight"]).shape[1]
    uses_event_features = (inputs > 112 and "instrument.0.weight" not in state and
                           "context_net.0.weight" not in state)
    return model, (mean, std, uses_event_features) if standardized else None


def batch_arrays(parquet, groups, scaler):
    data = [read_valid_group(parquet, int(g)) for g in groups]
    x = np.stack([d[2] for d in data])
    if scaler is not None:
        normalized = np.clip((x - scaler[0]) / scaler[1], -10, 10)
        if scaler[2]:
            normalized = np.concatenate([normalized, np.stack([event_features(r) for r in x])], axis=2)
        x = normalized
    return x.astype(np.float32), np.stack([d[3] for d in data]), np.stack([d[4] for d in data])


def score(model, parquet, groups, scaler):
    acc = GlobalWP()
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(groups), BATCH):
            x, y, mask = batch_arrays(parquet, groups[offset:offset + BATCH], scaler)
            xb = torch.from_numpy(x)
            prediction = (model(xb, torch.from_numpy(interval_features(x)))[0]
                          if isinstance(model, IntervalResidual) else model(xb)[0]).numpy()
            for i in range(len(x)):
                acc.add(y[i], prediction[i], mask[i])
    model.train()
    return acc.result()["weighted_pearson"]


def weighted_loss(prediction, target, row_weight, target_balance=False):
    weight = row_weight[..., None] * target.abs()
    error = (prediction - target).square() * weight
    if target_balance:
        return (error.sum((0, 1)) / weight.sum((0, 1)).clamp_min(1e-6)).mean()
    return error.sum() / weight.sum().clamp_min(1e-6)


def pearson_loss(prediction, target, row_weight):
    """1 - mean over targets of the |y|-weighted Pearson correlation on this chunk."""
    weight = (row_weight[..., None] * target.abs()).reshape(-1, 2)
    weight = weight / weight.sum(0).clamp_min(1e-6)
    p = prediction.reshape(-1, 2)
    t = target.reshape(-1, 2)
    p = p - (weight * p).sum(0)
    t = t - (weight * t).sum(0)
    covariance = (weight * p * t).sum(0)
    variance = (weight * p.square()).sum(0) * (weight * t.square()).sum(0)
    return 1 - (covariance / variance.clamp_min(1e-12).sqrt()).mean()


def block_lengths(x):
    """Length of the i0 price block (rows with an unchanged i0 price vector) containing each row; (batch, rows).

    Targets are constant inside a block, so 1 / length is each row's label uniqueness (AFML ch. 4)."""
    out = np.empty(x.shape[:2], np.float32)
    for i, seq in enumerate(x):
        changed = np.r_[True, np.any(seq[1:, :22] != seq[:-1, :22], axis=1)]
        block = np.cumsum(changed) - 1
        out[i] = np.bincount(block)[block]
    return out


def pooled_wp_gradient(prediction, target, mask):
    """Exact derivative of the pooled |y|-weighted Pearson (official Global WP, per target) w.r.t. each prediction.

    prediction, target: (batch, rows, 2); mask: (batch, rows) selected rows. All selected rows of all sequences
    in the batch are pooled, as the official metric pools all sequences. Targets and predictions are clipped to
    [-2, 2] as in scoring; rows outside the prediction clip get zero gradient. Returns (gradient, correlations)."""
    y = np.clip(target.astype(np.float64), -2, 2)
    p = np.clip(prediction.astype(np.float64), -2, 2)
    grad = np.zeros(prediction.shape, np.float32)
    rho = []
    for k in range(2):
        w = np.abs(y[..., k]) * mask
        total = w.sum()
        if total <= 0:
            rho.append(0.0)
            continue
        y_bar, p_bar = (w * y[..., k]).sum() / total, (w * p[..., k]).sum() / total
        dy, dp = y[..., k] - y_bar, p[..., k] - p_bar
        cov, var_y, var_p = (w * dy * dp).sum() / total, (w * dy * dy).sum() / total, (w * dp * dp).sum() / total
        if var_y <= 1e-12 or var_p <= 1e-12:
            rho.append(0.0)
            continue
        r = cov / np.sqrt(var_y * var_p)
        g = w / (total * np.sqrt(var_y * var_p)) * (dy - cov / var_p * dp)
        g[np.abs(prediction[..., k]) > 2] = 0
        grad[..., k] = g
        rho.append(float(r))
    return grad, rho


def next_event_deltas(x):
    """Next i0 price-event delta for auxiliary training; never an inference input."""
    target = np.zeros((*x.shape[:2], 22), np.float32)
    valid = np.zeros(x.shape[:2], bool)
    for i, values in enumerate(x):
        price = values[:, :22]
        changed = np.r_[True, np.any(price[1:] != price[:-1], axis=1)]
        starts = np.flatnonzero(changed)
        segment = np.cumsum(changed) - 1
        keep = segment + 1 < len(starts)
        target[i, keep] = price[starts[segment[keep] + 1]] - price[keep]
        valid[i, keep] = True
    return target, valid


def interval_features(x):
    """One-step, since-event and running interval-mean changes for i0."""
    out = np.empty((*x.shape[:2], 157), np.float32)
    for i, values in enumerate(x):
        values = values[:, :52]
        one = np.zeros_like(values)
        one[1:] = values[1:] - values[:-1]
        changed = np.r_[True, np.any(values[1:, :22] != values[:-1, :22], axis=1)]
        starts = np.flatnonzero(changed)
        segment = np.cumsum(changed) - 1
        begin = starts[segment]
        prefix = np.vstack((np.zeros((1, 52), np.float32), np.cumsum(values, axis=0)))
        count = np.arange(len(values)) - begin + 1
        average = (prefix[np.arange(len(values)) + 1] - prefix[begin]) / count[:, None]
        out[i] = np.column_stack((one, values - values[begin], average - values[begin],
                                  np.log1p(count - 1)))
    return out


def fit(name, groups, unscored, epochs, lr, seed, evaluate_groups=None, tag="", save=False,
        freeze_recurrent=False, l2sp=0.0, decay_after=0, decay_lr=None, chunk=CHUNK,
        target_balance=False, corr=0.0, event_aux=0.0, causal_conv=0, interval_mlp=0, gwp=0.0, gwp_mse=1.0,
        dropconnect=0.0, sam=0.0, beta2=0.999, input_noise=0.0, uniqueness=0.0):
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    model, scaler = load(name)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if causal_conv:
        if hasattr(model, "instrument"):
            raise ValueError("causal convolution screen currently supports direct-input GRUs only")
        model = CausalConvGRU(model, causal_conv)
    if interval_mlp:
        if causal_conv or event_aux:
            raise ValueError("screen interval residual separately")
        model = IntervalResidual(model, interval_mlp)
    if event_aux:
        if causal_conv:
            raise ValueError("screen causal convolution and event auxiliary separately")
        if hasattr(model, "instrument"):
            raise ValueError("event auxiliary screen currently supports direct-input GRUs only")
        model.event_head = torch.nn.Linear(model.rnn.hidden_size, 22)
    if freeze_recurrent:
        recurrent = model.base.rnn if isinstance(model, IntervalResidual) else model.rnn
        recurrent.weight_hh_l0.requires_grad_(False)
    anchor = {k: v.detach().clone() for k, v in model.named_parameters()}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, betas=(0.9, beta2))
    if (dropconnect or sam or input_noise) and (event_aux or interval_mlp or causal_conv):
        raise ValueError("fine-tune regularizers are implemented for plain models only")
    trainable = [p for p in model.parameters() if p.requires_grad]
    history = []
    if evaluate_groups is not None:
        history.append(score(model, parquet, evaluate_groups, scaler))
        print(tag, "epoch 0 held-out scored WP", round(history[-1], 6), flush=True)
    for epoch in range(epochs):
        if decay_after and epoch == decay_after:
            for group in optimizer.param_groups:
                group["lr"] = decay_lr
        losses = []
        order = rng.permutation(groups)
        for offset in range(0, len(order), BATCH):
            x, y, mask = batch_arrays(parquet, order[offset:offset + BATCH], scaler)
            event_y, event_valid = next_event_deltas(x) if event_aux else (None, None)
            interval = interval_features(x) if interval_mlp else None
            y = np.clip(y, -2, 2)
            row_weight = np.where(mask, 1.0, unscored).astype(np.float32)
            row_weight[:, :WARMUP] = 0
            if uniqueness:                                             # down-weight rows of long constant-label blocks
                row_weight *= block_lengths(x) ** (-uniqueness)
            if gwp:
                # pooled Global-WP term: statistics from a no-grad pass over the whole sequences, then each chunk
                # back-propagates the exact per-row derivative (scaled by the chunk count so a chunk step carries
                # the same weight as its MSE term)
                if event_aux or interval_mlp or causal_conv:
                    raise ValueError("pooled Global-WP term is implemented for plain models only")
                with torch.no_grad():
                    full, state = [], None
                    for start in range(0, N_ROWS, chunk):
                        out, state = model(torch.from_numpy(np.ascontiguousarray(x[:, start:start + chunk])), state)
                        full.append(out.numpy())
                wp_grad, _ = pooled_wp_gradient(np.concatenate(full, 1), y, mask)
                n_chunks = -(-N_ROWS // chunk)
            hidden = None
            for start in range(0, N_ROWS, chunk):
                sl = slice(start, start + chunk)
                xb = torch.from_numpy(np.ascontiguousarray(x[:, sl]))
                yb = torch.from_numpy(np.ascontiguousarray(y[:, sl]))
                wb = torch.from_numpy(np.ascontiguousarray(row_weight[:, sl]))
                optimizer.zero_grad()
                if input_noise:                                        # Gaussian noise on standardized inputs
                    xb = xb + input_noise * torch.randn_like(xb)
                if dropconnect:                                        # weight-dropped recurrent matrix, one mask per chunk
                    recurrent = model.rnn.weight_hh_l0
                    keep = (torch.rand_like(recurrent) >= dropconnect).float() / (1 - dropconnect)
                    raw = recurrent.data.clone()
                    recurrent.data.mul_(keep)
                hidden_in = hidden
                if event_aux:
                    states, hidden = model.rnn(xb, hidden)
                    prediction = model.head(states)
                elif interval_mlp:
                    fb = torch.from_numpy(np.ascontiguousarray(interval[:, sl]))
                    prediction, hidden = model(xb, fb, hidden)
                else:
                    prediction, hidden = model(xb, hidden)
                loss = weighted_loss(prediction, yb, wb, target_balance)
                if gwp:
                    gb = torch.from_numpy(np.ascontiguousarray(wp_grad[:, sl]))
                    loss = gwp_mse * loss - gwp * n_chunks * (gb * prediction).sum() / 2
                if event_aux:
                    eb = torch.from_numpy(np.ascontiguousarray(event_y[:, sl]))
                    vb = torch.from_numpy(np.ascontiguousarray(event_valid[:, sl]))
                    event_weight = wb[..., None] * yb.abs().mean(-1, keepdim=True) * vb[..., None]
                    event_error = (model.event_head(states) - eb).square() * event_weight
                    loss = loss + event_aux * event_error.sum() / (
                        event_weight.sum() * event_error.shape[-1]).clamp_min(1e-6)
                if corr:
                    loss = loss + corr * pearson_loss(prediction, yb, wb)
                if l2sp:
                    loss = loss + l2sp * sum((p - anchor[k]).square().sum()
                                             for k, p in model.named_parameters() if p.requires_grad)
                loss.backward()
                if sam:                                                # sharpness-aware step: gradient at w + rho g/|g|
                    norm = torch.sqrt(sum((p.grad ** 2).sum() for p in trainable if p.grad is not None)).clamp_min(1e-12)
                    shifts = []
                    with torch.no_grad():
                        for p in trainable:
                            e = sam * p.grad / norm if p.grad is not None else torch.zeros_like(p)
                            p.add_(e); shifts.append(e)
                    optimizer.zero_grad()
                    sharp, _ = model(xb, hidden_in)
                    sharp_loss = weighted_loss(sharp, yb, wb, target_balance)
                    sharp_loss.backward()
                    with torch.no_grad():
                        for p, e in zip(trainable, shifts):
                            p.sub_(e)
                if dropconnect:
                    recurrent.data.copy_(raw)
                    if recurrent.grad is not None:
                        recurrent.grad.mul_(keep)
                optimizer.step()
                hidden = detach_state(hidden)
                losses.append(loss.item())
        if save:
            torch.save(model.state_dict(), OUT / f"{tag}_e{epoch + 1}.pt")
        if evaluate_groups is not None:
            history.append(score(model, parquet, evaluate_groups, scaler))
        print(tag, "epoch", epoch + 1, "loss", round(float(np.mean(losses)), 6),
              "held-out scored WP", round(history[-1], 6) if history else None, flush=True)
    return model, history


def member_predictions(member, half, epochs, parquet, groups):
    """Held-out scored-row predictions of one (weight-averaged) member, cached on disk."""
    path = OUT / f"pred_{member}_e{'-'.join(map(str, epochs))}_h{half}.npz"
    if path.exists():
        with np.load(path) as cached:
            return cached["p"], cached["y"]
    model, scaler = load(member.split("_u")[0])
    states = [torch.load(OUT / f"{member}_h{half}_e{e}.pt", map_location="cpu", weights_only=True)
              for e in epochs]
    model.load_state_dict({k: sum(s[k] for s in states) / len(states) for k in states[0]})
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for offset in range(0, len(groups), BATCH):
            x, y, mask = batch_arrays(parquet, groups[offset:offset + BATCH], scaler)
            prediction = model(torch.from_numpy(x))[0].numpy()
            ps.append(prediction[mask])
            ys.append(y[mask])
    p, y = np.concatenate(ps), np.concatenate(ys)
    np.savez(path, p=p, y=y)
    return p, y


def combo(members, epochs):
    """Held-out scored WP of members, their average, and each leave-one-out average.

    Several epochs average each member's weights over those epochs (SWA)."""
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    result = {}
    for half in (0, 1):
        groups = np.arange(1_873)
        groups = groups[groups % 2 != half]
        predictions = {}
        for member in members:
            predictions[member], y = member_predictions(member, half, epochs, parquet, groups)

        def wp(p):
            acc = GlobalWP()
            acc.add(y, p, np.ones(len(y), bool))
            return acc.result()["weighted_pearson"]
        result[half] = {m: wp(p) for m, p in predictions.items()}
        result[half]["mean"] = wp(np.mean(list(predictions.values()), 0))
        for m in members if len(members) > 2 else ():
            result[half][f"without {m}"] = wp(np.mean([p for k, p in predictions.items() if k != m], 0))
        print("half", half, "epochs", epochs, {k: round(v, 6) for k, v in result[half].items()}, flush=True)
    for name in result[0]:
        print(f"{name:45s} mean {np.mean([result[h][name] for h in (0, 1)]):.6f}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "final", "combo", "selfcheck"))
    parser.add_argument("--members", nargs="+")
    parser.add_argument("--swa", type=int, nargs="+")
    parser.add_argument("--freeze-recurrent", action="store_true")
    parser.add_argument("--l2sp", type=float, default=0.0)
    parser.add_argument("--decay-after", type=int, default=0, help="epochs before switching to --decay-lr")
    parser.add_argument("--decay-lr", type=float, default=5e-5)
    parser.add_argument("--model", default="v1")
    parser.add_argument("--half", type=int, choices=(0, 1))
    parser.add_argument("--unscored", type=float, default=0.06)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--target-balance", action="store_true")
    parser.add_argument("--corr", type=float, default=0.0, help="weight of a weighted-Pearson loss term")
    parser.add_argument("--gwp", type=float, default=0.0,
                        help="weight of the pooled full-sequence Global-WP term (exact gradient)")
    parser.add_argument("--gwp-mse", type=float, default=1.0, help="MSE weight when --gwp is used")
    parser.add_argument("--dropconnect", type=float, default=0.0, help="recurrent weight-drop probability")
    parser.add_argument("--sam", type=float, default=0.0, help="SAM neighbourhood radius rho")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--input-noise", type=float, default=0.0, help="Gaussian noise std on standardized inputs")
    parser.add_argument("--uniqueness", type=float, default=0.0, help="row weight x (price-block length)^-P")
    parser.add_argument("--event-aux", type=float, default=0.0,
                        help="weight of next-price-event delta auxiliary loss")
    parser.add_argument("--causal-conv", type=int, default=0,
                        help="width of a kernel-5 causal convolutional residual")
    parser.add_argument("--interval-mlp", type=int, default=0,
                        help="width of an event-reset interval residual")
    args = parser.parse_args()
    if args.command == "selfcheck":
        x = np.zeros((1, 12, 112), np.float32)
        x[:, 3:, :22] = 1
        x[:, 7:, :22] = 3
        target, valid = next_event_deltas(x)
        assert np.all(target[0, :3] == 1) and np.all(target[0, 3:7] == 2)
        assert valid[0, :7].all() and not valid[0, 7:].any()
        torch.manual_seed(1)
        base = make_gru(4)
        model = CausalConvGRU(base, 3)
        values = torch.randn(2, 11, 112)
        expected = base(values)[0]
        actual = model(values)[0]
        first, state = model(values[:, :6])
        second, _ = model(values[:, 6:], state)
        assert torch.equal(expected, actual)
        assert torch.allclose(actual, torch.cat((first, second), 1), atol=1e-7)
        interval = interval_features(x)
        changed = x.copy(); changed[:, 8:] = 99
        assert np.array_equal(interval[:, :8], interval_features(changed)[:, :8])
        wrapped = IntervalResidual(base, 3)
        assert torch.equal(expected, wrapped(values, torch.zeros(2, 11, 157))[0])
        print("next-event auxiliary labels are causal inputs with future-only training targets")
        return
    torch.set_num_threads(3)
    OUT.mkdir(parents=True, exist_ok=True)
    groups = np.arange(1_873)
    tag = (f"{args.model}_u{args.unscored:g}_lr{args.lr:g}_s{args.seed}"
           f"{'_frz' if args.freeze_recurrent else ''}{f'_l2sp{args.l2sp:g}' if args.l2sp else ''}"
           f"{f'_d{args.decay_after}x{args.decay_lr:g}' if args.decay_after else ''}"
           f"{'_tb' if args.target_balance else ''}{f'_c{args.chunk}' if args.chunk != CHUNK else ''}"
           f"{f'_corr{args.corr:g}' if args.corr else ''}"
           f"{f'_evaux{args.event_aux:g}' if args.event_aux else ''}"
           f"{f'_conv{args.causal_conv}' if args.causal_conv else ''}"
           f"{f'_interval{args.interval_mlp}' if args.interval_mlp else ''}"
           f"{f'_gwp{args.gwp:g}m{args.gwp_mse:g}' if args.gwp else ''}"
           f"{f'_dc{args.dropconnect:g}' if args.dropconnect else ''}{f'_sam{args.sam:g}' if args.sam else ''}"
           f"{f'_b2{args.beta2:g}' if args.beta2 != 0.999 else ''}{f'_in{args.input_noise:g}' if args.input_noise else ''}"
           f"{f'_uq{args.uniqueness:g}' if args.uniqueness else ''}")
    if args.command == "run":
        fit_groups, eval_groups = groups[groups % 2 == args.half], groups[groups % 2 != args.half]
        _, history = fit(args.model, fit_groups, args.unscored, args.epochs, args.lr, args.seed,
                         eval_groups, f"{tag}_h{args.half}", save=True,
                         freeze_recurrent=args.freeze_recurrent, l2sp=args.l2sp,
                         decay_after=args.decay_after, decay_lr=args.decay_lr,
                         chunk=args.chunk, target_balance=args.target_balance, corr=args.corr, gwp=args.gwp, gwp_mse=args.gwp_mse, dropconnect=args.dropconnect,
                         sam=args.sam, beta2=args.beta2, input_noise=args.input_noise, uniqueness=args.uniqueness,
                         event_aux=args.event_aux, causal_conv=args.causal_conv,
                         interval_mlp=args.interval_mlp)
        (OUT / f"{tag}_h{args.half}.json").write_text(json.dumps(
            {"args": vars(args), "held_out_scored_wp_by_epoch": history}, indent=2) + "\n")
    elif args.command == "combo":
        combo(args.members, args.swa or [args.epochs])
    else:
        fit(args.model, groups, args.unscored, args.epochs, args.lr, args.seed, tag=f"{tag}_all",
            save=True, freeze_recurrent=args.freeze_recurrent, l2sp=args.l2sp,
            decay_after=args.decay_after, decay_lr=args.decay_lr,
            chunk=args.chunk, target_balance=args.target_balance, corr=args.corr, gwp=args.gwp, gwp_mse=args.gwp_mse, dropconnect=args.dropconnect,
                         sam=args.sam, beta2=args.beta2, input_noise=args.input_noise, uniqueness=args.uniqueness,
            event_aux=args.event_aux, causal_conv=args.causal_conv,
            interval_mlp=args.interval_mlp)


if __name__ == "__main__":
    main()
