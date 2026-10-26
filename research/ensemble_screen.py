"""Cross-fit cached members: global/tick weights and greedy pruning."""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np

from pilot import CACHE

OUT = CACHE / "ensemble_screen.json"
PRED = CACHE / "maskft"
MEMBERS = {
    "v1": "pred_v1_u0.06_lr0.0002_s7_e2-3_h{half}.npz",
    "g128a50": "pred_gru128_a0.5_s57_u0.06_lr0.0002_s7_e2-3_h{half}.npz",
    "g192a50": "pred_gru192_a0.5_s57_u0.06_lr0.0002_s7_e2-3_h{half}.npz",
    "g128a75": "pred_gru128_a0.75_s58_u0.06_lr0.0002_s7_e2-3_h{half}.npz",
    "g192ev": "pred_gru192ev_a0.5_s59_u0.06_lr0.0002_s7_e2-3_h{half}.npz",
    "g192a75": "pred_gru192_a0.75_s60_u0.06_lr0.0002_s7_e2-3_h{half}.npz",
    "g128s61": "pred_gru128_a0.75_s61_u0.06_lr0.0002_s7_e2-3_h{half}.npz",
    "split": "pred_split128_a0.75_s81_u0.06_lr0.0002_s7_tb_e2-3_h{half}.npz",
    "sd71": "pred_scratch128_uSD_e3-4-5_h{half}.npz",
    "sd72": "pred_scratch128s72_uSD_e3-4-5_h{half}.npz",
    "g71": "pred_scratch128g2_uSD_e3-4-5_h{half}.npz",
    "g72": "pred_scratch128g2s72_uSD_e3-4-5_h{half}.npz",
    "c71": "pred_scratch128g2c71_uC_e1-2-3_h{half}.npz",
    "c72": "pred_scratch128g2c72_uC_e1-2-3_h{half}.npz",
    "v1u02": "pred_v1_u0.02_lr0.0002_s7_e2-3_h{half}.npz",
}
TEACHER = tuple(range(8))
COST = (60, 92, 175, 92, 202, 175, 92, 106, 92, 92, 92, 92, 92, 92, 60)
BUDGET = 480


def wp(y, p):
    y = np.clip(y.astype(np.float64), -2, 2)
    p = np.clip(p.astype(np.float64), -2, 2)
    scores = []
    for target in range(2):
        w = np.abs(y[:, target])
        yc = y[:, target] - np.average(y[:, target], weights=w)
        pc = p[:, target] - np.average(p[:, target], weights=w)
        scores.append(np.sum(w * yc * pc) /
                      np.sqrt(np.sum(w * yc * yc) * np.sum(w * pc * pc)))
    return float(np.mean(scores))


def load(half):
    predictions, y = [], None
    for filename in MEMBERS.values():
        with np.load(PRED / filename.format(half=half)) as cached:
            if y is None:
                y = cached["y"].copy()
            elif not np.array_equal(y, cached["y"]):
                raise ValueError("cached target rows do not align")
            predictions.append(cached["p"].copy())
    with np.load(CACHE / "sgru" / f"shat_sgru128_s101_h{half}.npz") as cached:
        if not np.array_equal(y, cached["y"]):
            raise ValueError("tick rows do not align")
        ticks = cached["ticks"].copy()
    return np.stack(predictions, axis=1), y, ticks


def fit_weights(p, y, ticks=None, shrink=0.01):
    """Per-target nonnegative ridge weights, shrunk toward an equal blend."""
    groups = [None] if ticks is None else np.unique(ticks).tolist()
    fitted = {}
    for group in groups:
        rows = np.ones(len(y), bool) if group is None else ticks == group
        coefficients = np.empty((p.shape[1], 2))
        for target in range(2):
            x = p[rows, :, target].astype(np.float64)
            value = np.clip(y[rows, target].astype(np.float64), -2, 2)
            weight = np.abs(value)
            with np.errstate(all="ignore"):
                gram = x.T @ (weight[:, None] * x)
                rhs = x.T @ (weight * value)
            if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
                raise ValueError("nonfinite ensemble sufficient statistics")
            penalty = shrink * np.trace(gram) / len(gram)
            prior = np.full(len(gram), 1 / len(gram))
            beta = np.linalg.solve(gram + penalty * np.eye(len(gram)), rhs + penalty * prior)
            beta = np.maximum(beta, 0)
            coefficients[:, target] = beta / beta.sum()
        fitted[group] = coefficients
    return fitted


def weighted_prediction(p, fitted, ticks=None):
    out = np.empty((len(p), 2), np.float32)
    for group, weights in fitted.items():
        rows = np.ones(len(p), bool) if group is None else ticks == group
        out[rows] = np.einsum("rmt,mt->rt", p[rows], weights)
    return out


def greedy(p, y, limit=5):
    selected, remaining = [], list(range(p.shape[1]))
    for _ in range(limit):
        candidate = max(remaining, key=lambda j: wp(y, p[:, selected + [j]].mean(1)))
        selected.append(candidate)
        remaining.remove(candidate)
    return selected


def moments(p, y):
    result = []
    for target in range(2):
        value = np.clip(y[:, target].astype(np.float64), -2, 2)
        weight = np.abs(value)
        centered_y = value - np.average(value, weights=weight)
        values = p[:, :, target].astype(np.float64)
        values -= np.average(values, axis=0, weights=weight)
        with np.errstate(all="ignore"):
            covariance = values.T @ (weight * centered_y)
            variance = values.T @ (weight[:, None] * values)
        if not np.isfinite(covariance).all() or not np.isfinite(variance).all():
            raise ValueError("nonfinite subset sufficient statistics")
        result.append((covariance, variance, np.sum(weight * centered_y**2)))
    return result


def moment_score(stats, members):
    a = np.zeros(len(COST))
    a[list(members)] = 1 / len(members)
    return moment_weight_score(stats, a)


def moment_weight_score(stats, weights):
    return float(np.mean([weights @ cov / np.sqrt(var_y * (weights @ variance @ weights))
                          for cov, variance, var_y in stats]))


def budget_subset(p, y):
    stats = moments(p, y)
    candidates = (members for size in range(1, 6) for members in combinations(range(len(COST)), size)
                  if sum(COST[i] for i in members) <= BUDGET)
    shortlist = sorted(candidates, key=lambda members: moment_score(stats, members), reverse=True)[:30]
    return max(shortlist, key=lambda members: wp(y, p[:, members].mean(1)))


def continued_weights(train_p, train_y):
    names = list(MEMBERS)
    pair = (names.index("c71"), names.index("c72"))
    gru, split = names.index("g192a50"), names.index("split")
    candidates = []
    for pair_units in range(21):
        for gru_units in range(21 - pair_units):
            split_units = 20 - pair_units - gru_units
            weights = np.zeros(len(names))
            weights[list(pair)] = pair_units / 40
            weights[gru] = gru_units / 20
            weights[split] = split_units / 20
            candidates.append(weights)
    stats = moments(train_p, train_y)
    shortlist = sorted(candidates, key=lambda weights: moment_weight_score(stats, weights),
                       reverse=True)[:20]
    return max(shortlist, key=lambda weights: wp(train_y, np.einsum("rmt,m->rt", train_p, weights)))


def selfcheck():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(1000, 2)).astype(np.float32)
    p = np.stack((y + rng.normal(scale=0.1, size=y.shape),
                  rng.normal(size=y.shape)), axis=1).astype(np.float32)
    ticks = np.repeat([0.1, 0.2], 500)
    fitted = fit_weights(p, y, ticks)
    assert set(fitted) == {0.1, 0.2}
    assert all(np.allclose(weights.sum(0), 1) and (weights >= 0).all()
               for weights in fitted.values())
    assert wp(y, weighted_prediction(p, fitted, ticks)) > wp(y, p.mean(1))
    print("cross-fit ensemble weighting self-check passed")


def main():
    halves = [load(0), load(1)]
    names = list(MEMBERS)
    report = []
    for test_half in (0, 1):
        train_p, train_y, train_ticks = halves[1 - test_half]
        test_p, test_y, test_ticks = halves[test_half]
        global_fit = fit_weights(train_p, train_y)
        tick_fit = fit_weights(train_p, train_y, train_ticks)
        chosen = greedy(train_p, train_y)
        budget = budget_subset(train_p, train_y)
        continued = continued_weights(train_p, train_y)
        v21 = [names.index(name) for name in ("split", "sd71", "sd72", "g71", "g72")]
        v21_global = fit_weights(train_p[:, v21], train_y)
        v21_tick = fit_weights(train_p[:, v21], train_y, train_ticks)
        result = {
            "test_half": test_half,
            "teacher8_equal": wp(test_y, test_p[:, TEACHER].mean(1)),
            "all_equal": wp(test_y, test_p.mean(1)),
            "global_weights": wp(test_y, weighted_prediction(test_p, global_fit)),
            "tick_weights": wp(test_y, weighted_prediction(test_p, tick_fit, test_ticks)),
            "greedy_members": [names[i] for i in chosen],
            "greedy5": wp(test_y, test_p[:, chosen].mean(1)),
            "budget_members": [names[i] for i in budget],
            "budget_cost": sum(COST[i] for i in budget),
            "budget_score": wp(test_y, test_p[:, budget].mean(1)),
            "v21_equal": wp(test_y, test_p[:, v21].mean(1)),
            "v21_global_weights": wp(test_y, weighted_prediction(test_p[:, v21], v21_global)),
            "v21_tick_weights": wp(test_y, weighted_prediction(
                test_p[:, v21], v21_tick, test_ticks)),
            "continued_weights": {names[i]: float(value) for i, value in enumerate(continued) if value},
            "continued_weight_score": wp(test_y, np.einsum("rmt,m->rt", test_p, continued)),
            "global_weight_values": global_fit[None].tolist(),
        }
        report.append(result)
        print(json.dumps(result), flush=True)
    summary = {key: float(np.mean([r[key] for r in report]))
               for key in ("teacher8_equal", "all_equal", "global_weights", "tick_weights", "greedy5",
                           "budget_score", "v21_equal", "v21_global_weights", "v21_tick_weights")}
    summary["continued_weight_score"] = float(np.mean(
        [row["continued_weight_score"] for row in report]))
    pair_scores = {}
    for left, right in combinations(("sd71", "sd72", "g71", "g72"), 2):
        members = [names.index(name) for name in ("g192a50", "split", left, right)]
        scores = [wp(y, p[:, members].mean(1)) for p, y, _ in halves]
        pair_scores[f"{left}+{right}"] = {"halves": scores, "mean": float(np.mean(scores))}
    fixed_sets = {}
    for label, selected in {
        "c71+c72+split": ("c71", "c72", "split"),
        "c71+c72+g192+split": ("c71", "c72", "g192a50", "split"),
        "sd71+c71+g192+split": ("sd71", "c71", "g192a50", "split"),
        "c71+c72+split+v1u02": ("c71", "c72", "split", "v1u02"),
        "c71+c72+g192+split+v1u02": ("c71", "c72", "g192a50", "split", "v1u02"),
    }.items():
        members = [names.index(name) for name in selected]
        scores = [wp(y, p[:, members].mean(1)) for p, y, _ in halves]
        fixed_sets[label] = {"halves": scores, "mean": float(np.mean(scores)),
                             "cost": sum(COST[i] for i in members)}
    output = {"members": names, "halves": report, "mean": summary,
              "g192_split_student_pairs": pair_scores, "continued_sets": fixed_sets}
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(json.dumps(pair_scores, indent=2), flush=True)
    print(json.dumps(fixed_sets, indent=2), flush=True)


if __name__ == "__main__":
    import sys
    selfcheck() if sys.argv[-1:] == ["selfcheck"] else main()
