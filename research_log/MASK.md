# Mask-aware fine-tuning (2026-09-24)

## Why

The official score is Global Weighted Pearson over rows selected by the hidden
`is_scored` mask. On public validation that mask is not a random sample:

- about 8–9% of required rows are scored overall, but the rate ranges from about 1% to 18% per sequence and scored rows arrive in bursts;
- scored rows are about twice as likely to coincide with an `i1` ask-price change; `a3` alone separates them with AUC 0.69;
- v1 scores **0.688 on scored rows and 0.464 on unscored rows** (80-sequence probe).

Every earlier selection metric (FIT-CV, DEV, early/late/random masks) scores
all rows, so it did not target the rows that count. A linear model on v1's
causal hidden state predicts the mask with AUC 0.835, so the selection is
learnable from past inputs.

Leaderboard context (public API, 2026-09-24): v1 = 0.6487 (rank 230 of 430
scored). The supplied baseline scores 0.6171 on local validation and 0.5935 on
the leaderboard (62 entries share that score). v1 drops by a similar amount,
0.6756 → 0.6487, so the local-to-leaderboard gap is mostly common to all
entries and is not specific to v1.

## Protocol

Validation row groups are split by parity. A model is fine-tuned on one half
and scored on the other half's official scored rows; the halves are then swapped.
The reported number is the mean of the two held-out halves. The final models
are refitted on all 1,873 validation sequences using the chosen recipe. The
rules allow training on `valid.parquet`.

Fine-tuning (`maskft.py`) starts from a frozen full-FIT GRU. It uses Adam with
lr 2e-4, 32 sequences per batch, and 512-row truncated backpropagation with
the hidden state carried across chunks. The loss weight is
`|y| * (1 if is_scored else u)`. `u = 1` is the uniform-weight control.

## Results (held-out scored-row Global WP, mean of the two halves)

| Model | Score | vs v1 |
|---|---:|---:|
| v1 (frozen) | 0.675590 | — |
| v1 head-only refit, all rows (`stack.py`) | 0.675260 | −0.00033 |
| v1 head-only refit, scored ×16 | 0.677470 | +0.00188 |
| 5 frozen GRUs, scored ×16 linear readout | 0.677610 | +0.00202 |
| v1 + v2 frozen average | 0.676870 | +0.00128 |
| v1 fine-tuned, u = 1 (control), 3 epochs | 0.677191 | +0.00160 |
| v1 fine-tuned, u = 0.06, 3 epochs | 0.680127 | +0.00454 |
| v2 fine-tuned, u = 0.06, 3 epochs | 0.680306 | +0.00472 |
| **v1 + v2 fine-tuned average (v3)** | **0.681516** | **+0.00593** |
| v1 + v2 + t_norm fine-tuned average (v3b) | 0.681871 | +0.00628 |

The per-half control gains were +0.0027 and +0.0005. The mask-weighted gains
were +0.0057 and +0.0034. Mask weighting therefore accounts for most of the
improvement. The v3 average improved both halves (+0.0067 and +0.0051). v3b
improved on v3 in both halves (+0.0002 and +0.0005).

Epoch curve for v1 at u = 0.06 (mean gain over the two halves): e1 +0.0035,
e2 +0.0042, e3 +0.0045, e4 +0.0035. At e6, the second half had fallen below
its e1 score. Three epochs is the chosen recipe for the final refit.

## Submission packages

`multi_solution.py` steps K standardized-input GRUs together: one input
matmul, one batched recurrent matmul, and a pre-averaged output head. It
reproduces v1 exactly (7–8×10⁻⁷ vs PyTorch). Best of three idle-host runs per
20,000 rows: v1 ZIP 0.411 s, v3 0.405 s, v3b 0.466 s. v1 took 22m16s on the
official runner, so v3 and v3b project to roughly 22 and 25 minutes. That is a
host-ratio projection, not a Linux measurement.

| ZIP | Members | Bytes | Max diff vs PyTorch |
|---|---|---:|---:|
| `cache/packages/submission_v3.zip` | v1, v2 fine-tuned on all validation | 453,871 | 7.2×10⁻⁷ |
| `cache/packages/submission_v3b.zip` | v1, v2, t_norm fine-tuned on all validation | 679,341 | 8.3×10⁻⁷ |

Caveats: validation now enters training, so no local holdout remains for the
final refit. The half-split estimates are the only local evidence. If the
hidden test mask is drawn differently from the validation mask, the gain from
mask weighting will shrink. It did not turn negative on held-out rows in
either half. On about 312 held-out sequences per half, fine-tuned v1
(u = 0.06, e3) beat frozen v1 on all required rows (0.50237 vs 0.50057;
0.51365 vs 0.51112) and on unscored rows alone (0.46859 vs 0.46764;
0.48290 vs 0.48073).

Reproduce:

```bash
python research/stack.py extract && python research/stack.py evaluate
python research/maskft.py run --model v1 --half 0 --unscored 0.06 --epochs 3   # also --half 1, --model v2/t_norm
python research/maskft.py combo --members v1_u0.06_lr0.0002_s7 v2_u0.06_lr0.0002_s7 --epochs 3
python research/maskft.py final --model v1 --unscored 0.06 --epochs 3           # also v2, t_norm
python research/package_ft.py submission_v3 research/cache/maskft/v1_u0.06_lr0.0002_s7_all_e3.pt \
    research/cache/maskft/v2_u0.06_lr0.0002_s7_all_e3.pt
```

## Round 2 (2026-09-24)

These use the same held-out protocol: scored-row Global WP, mean of the two
validation halves. All use u = 0.06.

| Change | Score | vs v3b (0.681871) | Decision |
|---|---:|---:|---|
| v1 FT, weights averaged over epochs 2 and 3 (SWA) vs e3 alone | 0.680842 vs 0.680127 | — | adopt (+0.0007; +0.0001 / +0.0013 per half) |
| v1 FT, lr 1e-4, e3 / e4 | 0.680274 / 0.680260 | — | no gain over lr 2e-4 |
| v1 FT with train sequences weighted by learned mask probability (`maskmix.py`), best epoch | 0.679914 (h0 e3) / 0.678054 (h1 e2) | — | reject: below validation-only FT on both halves |
| **v1 + v2 + t_norm, each SWA(e2, e3) (v4)** | **0.682493** | **+0.00062** | adopt; halves +0.0000 / +0.0012 |
| v4 + v1 FT at lr 1e-4 as fourth member | 0.682407 | +0.00054 | reject: same-start members add nothing |

The mask net in `maskmix.py` is a causal GRU64 trained on one validation half.
On the other half it predicts `is_scored` with AUC 0.908 / 0.905, and its
mean probability (0.104) is close to the scored rate (0.095). Even with that
accuracy, mixing train sequences into fine-tuning diluted the gain. Much of
the effect appears to come from fitting the validation period itself, as the
uniform-weight control also suggests.

`submission_v4.zip` (677,402 bytes; 6.0×10⁻⁷ max difference vs PyTorch) uses
the all-validation e2/e3 checkpoints of v1, v2 and t_norm, weight-averaged per
member. Its callback costs the same as v3b's.

### Wider independent student

`distill_mps.py` distils the cached full-FIT three-GRU teacher into a
GRU128 student with v1's recipe (alpha 0.5, 5 epochs, lr 1e-3 then 3e-4). It
runs on the Apple GPU at about 12 minutes per epoch when the GPU is otherwise
idle. Two concurrent MPS jobs ran several times slower each.

| Model | Held-out scored WP |
|---|---:|
| GRU128 student, frozen | 0.675843 (v1 0.675590) |
| GRU128 FT, e3 | 0.679411 (h1 peaks at e1 and declines) |
| GRU128 FT, SWA(e2, e3) | 0.680500 |
| **v4 members + GRU128, each SWA(e2, e3) (v5)** | **0.682965** (+0.00047 vs v4; +0.0005 / +0.0004 per half; +0.0074 vs v1) |

`package_ft.py` now zero-pads narrower members to the widest member's width.
A padded GRU unit stays exactly zero and its head weight is zero, so outputs
are unchanged; a PyTorch check shows zero difference. `submission_v5.zip` is
1,026,919 bytes with a 6.0×10⁻⁷ maximum difference vs PyTorch. Its callback
took 0.58 s per 20,000 rows while GPU training shared the host.

### GRU192 student and member selection

| Model | Held-out scored WP |
|---|---:|
| GRU192 student, frozen | 0.676363 (best frozen base so far) |
| GRU192 FT, e1 / e3 | 0.680120 / 0.680014 (h1 peaks at e1) |
| GRU192 FT, SWA(e2, e3) | 0.681119 |
| v1, v2, t_norm, GRU128, GRU192 (all SWA e2/e3) | 0.683521 (+0.00056 vs v5 in each half) |
| **same without t_norm (v6)** | **0.683549** (+0.0080 vs v1) |

Leave-one-out means: without v1 0.683306, without v2 0.683452, without GRU128
0.683326, without GRU192 0.682965, without t_norm 0.683549. Only t_norm adds
nothing, so v6 drops it. `maskft.py combo` now caches each member's
held-out scored-row predictions (`cache/maskft/pred_*.npz`), so later subset
checks cost seconds.

`submission_v6.zip`: v1, v2, GRU128, GRU192, each SWA(e2, e3) fine-tuned on
all validation, zero-padded to width 192. It is 1,460,325 bytes, with a
6.0×10⁻⁷ maximum difference vs PyTorch, and took 0.66 s per 20,000 rows
while GPU training shared the host.

### Second GRU128 student (alpha 0.75, seed 58) and v7

| Model | Held-out scored WP |
|---|---:|
| GRU128 α0.75 student, frozen | 0.675363 |
| GRU128 α0.75 FT, e3 | 0.681344 (best single FT so far) |
| GRU128 α0.75 FT, SWA(e2, e3) | 0.682044 |
| v6 + GRU128 α0.75 | 0.683765 (+0.00022 vs v6; +0.00018 / +0.00025 per half) |
| **same without v2 (v7)** | **0.683816** (+0.0082 vs v1) |

Dropping v2 changed the halves by +0.00016 and −0.00006, which is a tie. v7
keeps four members to save runtime. `submission_v7.zip` holds v1, GRU128
α0.5, GRU192 and GRU128 α0.75, each SWA(e2, e3) fine-tuned on all validation.
It is 1,580,439 bytes, with a 6.0×10⁻⁷ maximum difference vs PyTorch.
Best-of-three idle-host callback time per 20,000 rows: v1 0.359 s, v6 0.616 s,
v7 0.617 s. Scaling v1's official 22m16s by 1.7× projects v7 to about 38
minutes. That is a host-ratio projection, not a Linux measurement.

Selection caveat: v4 through v7 each add +0.0002 to +0.0006, and every such
choice was made on the same two validation halves. Part of those later
increments may be selection noise. The main step, mask-weighted fine-tuning
(+0.0045 per model, +0.006 for v3), is much larger than that noise.

| Package | Held-out scored WP | vs v1 | Members |
|---|---:|---:|---|
| v1 | 0.675590 | — | distilled GRU96 |
| v3 | 0.681516 | +0.0059 | v1, v2 FT |
| v4 | 0.682493 | +0.0069 | v1, v2, t_norm FT, SWA |
| v6 | 0.683549 | +0.0080 | v1, v2, GRU128, GRU192 FT, SWA |
| **v7** | **0.683816** | **+0.0082** | v1, GRU128 α0.5, GRU192, GRU128 α0.75 FT, SWA |

## Round 3: data structure findings (2026-09-25)

**Targets are tick-quantized and event-timed.** Within a sequence, t0 and t1
are integer multiples of one tick unit. (t0, t1) is exactly constant between
changes of the i0 price block (columns 0–21) in the three train sequences
checked, and changes only at those events. (t0 + t1) / tick is mostly −1 or
−2, which reads like a spread term S = −(t0 + t1); D = t0 − t1 is the
direction term. For v1 on 80 validation sequences' scored rows,
corr(pred S, S) = 0.51 and corr(pred D, D) = 0.59. Replacing only S with the
truth would raise the score from 0.688 to 0.726. S changes only at i0 price
events but is not a function of the current price state alone: in one
sequence, 10 of 34 states carry more than one S value. An MLP on current plus
event-time features reached corr(S) ≈ 0.54, so S is not a quick win.

**Four tick-size groups.** The tick unit takes exactly four values: 0.125,
0.241, 0.469 and 0.657. By sequence count, train is 12% / 12% / 33% / 43% and
validation is 21% / 19% / 23% / 37%. A linear classifier on warm-up-row
statistics identifies the group with 97% accuracy (the 0.657 group at 100%).
Held-out scored-row WP for v1 by group is 0.787 / 0.759 / 0.720 / **0.420**.
The 0.657 group holds 49% of validation scored rows. Per-group calibration
slopes are about 1, and per-group affine recalibration gains only 0.0004. A
0.657-only specialist fine-tune lowered that group's held-out score (0.432 →
0.431 and 0.419 → 0.415). Every member scores 0.424–0.426 on the 0.657 group;
the v7 ensemble scores 0.429.

**Validation is a different period.** A linear classifier on sequence-level
feature statistics separates train from validation with held-out AUC 0.976, and
0.965–0.979 within each tick group. This fits validation fine-tuning helping
and train mixing hurting. Mixing the 1,500 most validation-like train
sequences into fine-tuning peaked at e1 (0.679523). That is above
random-train mixing but below validation-only fine-tuning (e3 0.680127, SWA
0.680842), so it is rejected.

`features.py` adds 46 causal event-time inputs: the price-block change at the
last two transitions, rows since the last transition, and the transition
count. The batch and row-wise versions agree in a self-check. The packager and
callback accept members with these inputs (zero input columns for the other
members). Re-packaging v7 with the new code gives a byte-identical
`model.npz`.

### Round 3 model results

All scores are held-out scored-row WP, mean of the two halves, with SWA(e2, e3)
unless stated otherwise.

| Change | Result | Decision |
|---|---|---|
| GRU192 with `features.py` event inputs (GRU192ev), frozen | 0.676365 (plain GRU192 0.676363) | inputs alone add nothing |
| GRU192ev FT | e3 0.678413; SWA 0.679974 (overfits faster on h1) | weaker single |
| v7 + GRU192ev | **0.684028** (+0.00021; +0.0003 / +0.0001 per half; +0.0084 vs v1) | adopt as v8 candidate |
| GRU192 FT with frozen recurrent weights | e3 0.680645 vs 0.680014; SWA 0.681299 vs 0.681119 | better single, but adds nothing in the ensemble next to plain GRU192 |
| GRU192 FT with L2-SP (λ = 0.01) | e3 0.679579 | reject |
| Within-interval EMA of predictions (80-sequence probe) | 0.68776 raw → 0.68595 / 0.68301 / 0.67144 for α 0.5 / 0.3 / 0.1 | reject: the latest row carries the most information |
| Callback with separate width groups instead of one padded batch | 0.92 s vs 0.68 s per 20,000 rows under the same load | reject: NumPy call overhead dominates, so padding is cheaper |

### GRU192 α0.75 student, subset search and v10 (2026-09-25)

The GRU192 α0.75 (seed 60) distillation was interrupted after epoch 4 when
the session ended. `distill_mps.py --resume 4` continued it, replaying the
skipped epoch permutations so the epoch order matches an uninterrupted run;
only the Adam moments restart. The student scores 0.675343 frozen, 0.680838
after FT e1 and 0.682055 with SWA(e2, e3).

Every subset of the six fine-tuned members was scored from cached held-out
predictions. Cost is padded MACs per row, 3·K·H·(inputs + H).

| Members (all SWA e2/e3) | Score | h0 / h1 | kMAC per row |
|---|---:|---|---:|
| all six | 0.684189 | 0.684992 / 0.683387 | 1210 |
| v9: v1, GRU192, GRU128 α0.75, GRU192ev, GRU192 α0.75 | 0.684186 | 0.684987 / 0.683386 | 1008 |
| v8: v7 + GRU192ev | 0.684028 | 0.684887 / 0.683168 | 1008 |
| **v10: v1, GRU192, GRU128 α0.75, GRU192 α0.75** | **0.684038** | 0.684744 / 0.683331 | 700 |
| v7: v1, GRU128 α0.5, GRU192, GRU128 α0.75 | 0.683816 | 0.684588 / 0.683044 | 700 |

Best-of-three idle-host callback time per 20,000 rows: v1 0.36–0.38 s,
v7 0.615 s, v10 0.623 s, v8 0.804 s, v9 0.806 s. Scaling v1's official
22m16s gives about 37 minutes for v10 and about 50 minutes for v8/v9. v8 and
v9 are not recommended: they gain at most 0.00015 over v10 for a large share
of the remaining time budget. The event member forces 158 inputs on every
padded member.

**Recommended: `submission_v10.zip`** (1,888,930 bytes; 6.0×10⁻⁷ maximum
difference vs PyTorch). Held-out scored-row score 0.684038, +0.0084 over v1
(0.675590).

## Round 4 (2026-09-26)

**Target formula, refined (0.657 group).** Per interval, (t0, t1) in ticks
is mostly (−1, −1), (0, −1) or (−1, 0). At a price event, usually exactly one
of t0 and t1 moves by ±1 tick, or neither moves (in six sequences, 34–45% of
events leave both unchanged). This fits t0 = F − ask(now) and
t1 = bid(now) − F, with F a slowly changing future price on the tick grid.
Then S = −(t0 + t1) ≈ the current spread, and D = t0 − t1 = 2(F − mid). S is
largely a function of the current price state within a sequence
(majority-state purity 0.98 and 0.85 in two sequences). About 7% of rows have
S ≤ 0 or S ≥ 3, which a spread cannot explain. Across sequences the state→S
map does not transfer: an MLP on the 22 i0 price features predicts S with
held-out-sequence corr 0.39 (accuracy 0.777 vs 0.659 for the mode). Causal
within-sequence relative features (distance from running max/min, running
z-score) give 0.38. S is therefore not a cheap win either.

| Change | Held-out scored WP | Decision |
|---|---|---|
| Ensemble weights cross-fitted between halves (v10 members / all six) | 0.684018 / 0.684178 vs equal 0.684038 / 0.684189 | keep equal weights |
| v1 FT with lr 2e-4 for 2 epochs, then 5e-5 to epoch 6; SWA windows e2–3, e3–6, e2–6, {2, 4, 6} | 0.680697 / 0.680681 / 0.680773 / 0.680715 vs 0.680842 for the standard recipe | keep standard recipe |

CPU-heavy evaluation jobs slow MPS distillation severely (2,499 s instead of
about 790 s for one GRU128 epoch), because batch loading and standardization
run on the CPU.

**Seventh member.** A GRU128 α0.75 student with seed 61 scores 0.674984
frozen, 0.680801 after FT e3 and 0.681787 with SWA(e2, e3). It improves no
four-member set. The best four-member set, which fits v10's runtime, is still
v10 (0.684038); next are {v1, GRU192, GRU192 α0.75, GRU128 s61} at 0.683990
and {GRU192, GRU128 α0.75, GRU192 α0.75, GRU128 s61} at 0.683978. The best
set overall is v9 + seed 61 at 0.684221, but it costs 1,210 kMAC per row, about
1.7× v10's cost, which would exceed a safe runtime.

**Where this leaves the approach.** Inside the runtime budget, the
fine-tuned-GRU ensemble has plateaued at about 0.6840 held-out scored WP,
+0.0084 over v1. Recent additions each changed the score by 0.0002 or less.
The official leaderboard score of v10 (or v7) would change the plan most: it
shows whether validation fine-tuning transfers to the hidden test period. A GRU192 α0.75 seed-62 distillation was started, then stopped
before completion, because its expected gain was below the selection noise.

## Official result for v10 and the runtime model (2026-09-26)

v10 scored **0.6582** on the public leaderboard (rank 53), against 0.6487 for
v1: **+0.0095 official vs +0.0084 held-out**. Validation fine-tuning
transfers to the hidden test period, so held-out gains can be trusted.

v10 ran for **56m07s** against the 60-minute limit. The host-ratio projection
(about 37 minutes) was badly wrong: the official CPU is far more sensitive to
per-row compute than this Mac. A linear fit through the two official runs
(v1: 60 kMAC per row, 22m16s; v10: 700 kMAC per row with padding, 56m07s)
gives

    official minutes ≈ 19.1 + 52.9 × (kMAC per row / 1000)

Consequences:
- v10 has no safety margin. For a final re-run on a possibly different
  dataset, prefer packages estimated at 45 minutes or less.
- Padding narrow members up to width 192 is real cost on the official
  machine. Separate width groups are not the fix, because each group adds
  NumPy call overhead (measured locally). The fix is ensembles whose members
  share one width.
- After fine-tuning, GRU128 matches GRU192 (SWA 0.6820 vs 0.6821) at about
  half the compute. Width-128 sets are the efficient choice:
  {v1, GRU128 α0.5, GRU128 α0.75 s58, GRU128 α0.75 s61} scores 0.683562
  (369 kMAC, about 39 minutes). {GRU192, GRU128 α0.75, GRU192 α0.75} scores
  0.683853 (525 kMAC, about 47 minutes).

### Leaner callback (2026-09-26)

`multi_solution.py` now folds the work below into the weights at load time:
- standardization goes into the input weights; clipping raw values to
  mean ± 10·std equals clipping standardized values to ±10, which fires on
  about 0.9% of validation rows, so the clip stays;
- all biases enter through a constant-1 input and a constant-1 hidden unit;
- `scipy.special.expit` computes the sigmoid, with a NumPy fallback;
- the per-row NaN guard is dropped: v1 raised on nonfinite input and was
  scored without error.

Predictions match PyTorch to ≤ 6.3×10⁻⁷, including a member with event
inputs. Under the same background load, the callback took 0.532 s instead of
0.633 s per 20,000 rows on the v10 members, and 0.443 s instead of 0.575 s on
the v11 members. The saving is call overhead, which the official runner
charges too; how much it saves there is not yet measured.

### Ensemble distillation (`ensdistill.py`)

The teacher is the mean of the seven fine-tuned members (SWA e2/e3) for one
validation half. The student starts from the GRU128 α0.75 s58 base and trains
on that half's validation sequences plus a fixed pool of 2,000 train
sequences, with loss 0.5 · label + 0.5 · imitation. Validation rows use
true-mask weights and train rows use mask-net weights. Teacher predictions are
cached per split, which cut an epoch from about 21 to 8 minutes.

| Student (GRU128 α0.75 s58 base) | h0 | h1 | Mean |
|---|---:|---:|---:|
| Regular FT, SWA(e2, e3) | 0.682614 | 0.681474 | 0.682044 |
| Distilled, epochs 1 / 2 / 3 / 4 (h0 first run, without cache) | 0.681865 / 0.682607 / 0.682809 / 0.683176 | 0.681355 / 0.681129 / 0.680849 / 0.680555 | — |
| Distilled, SWA(e1–e4) | 0.683214 | 0.681534 | **0.682374** (+0.00033) |
| Teacher (mean of seven) | ≈ 0.6849 | ≈ 0.6835 | 0.684221 |

The student recovers only a small part of the teacher's margin. Its
predictions correlate 0.9975 with the regular FT member's; two different
GRU128 α0.75 seeds correlate 0.9912. Swapping it in for the regular member
still helps both halves: v11 set 0.683562 → 0.683705, and v10 set
0.684038 → **0.684183**.

### Packages after the official v10 result

The final distilled student was trained on all validation with teachers
fine-tuned on all validation, for 4 epochs; SWA covers e1–e4. Both packages
below use the leaner callback. Best-of-three idle-host times per 20,000 rows,
all measured in one run: v1 0.362 s, v10 0.628 s, v12 0.519 s. v13 was
measured in a later run next to v1 (0.365 s) at 0.439 s.

| Package | Members | Held-out scored WP | kMAC per row | Official runtime |
|---|---|---:|---:|---|
| v10 (submitted) | v1, GRU192, GRU128 α0.75, GRU192 α0.75 | 0.684038 | 700 | 56m07s measured |
| **v12** | v10 with GRU128 α0.75 replaced by its distilled version | **0.684183** | 700 | same compute as v10 with fewer calls, so at most v10's time; how much faster is unmeasured |
| v13 | v1, GRU128 α0.5, distilled GRU128 α0.75, GRU128 α0.75 s61 | 0.683705 | 369 | about 39 minutes by the linear model before the callback change; likely less |

Both match PyTorch to ≤ 7.2×10⁻⁷ (v12 1,889,754 bytes; v13 1,267,869 bytes).
v12 is the next submission to try: its official runtime calibrates how much
the leaner callback saves. v13 is the safe choice for a final re-run on a
dataset of unknown size.

### v12 official result: call overhead is not the bottleneck (2026-09-27)

v12 scored **0.6583** (+0.0001 over v10; the held-out difference was
+0.00015) and ran **56m01s**, against 56m07s for v10. The leaner callback cut
NumPy calls by about 40% and made no measurable official difference. On the
official runner, time follows matmul work (weights streamed every row);
per-call overhead matters only on this Mac, where large caches make the
matmuls cheap.

So padding narrow members to width 192 wastes real official time, and width
groups are the right trade there, even though they were slower locally.
`multi_solution.py` now stacks every member's input rows into one shared
input matmul (no padding), then runs the recurrent matmul and gate math once
per width group. `package_ft.py` pads only input columns (for event members).

| Package | Members | Held-out | kMAC per row | Estimated official time |
|---|---|---:|---:|---|
| v12 (submitted, padded) | v1, GRU192, distilled GRU128 α0.75, GRU192 α0.75 | 0.684183 | 700 | 56m01s measured |
| **v14** (same members, grouped) | same | 0.684183 | 502 | about 46 minutes if the linear model holds |

v14 matches PyTorch to 6.6×10⁻⁷ (1,885,836 bytes). Locally it is slower
(0.69 s vs 0.52 s per 20,000 rows) because of the extra per-group calls. Its
official run will test the grouping hypothesis; the score should equal v12's.

With unpadded costs, ensembles plateau near 0.6842: the best set estimated
under 51 minutes adds GRU128 s61 to v14 for 0.684221 (about 50.5 minutes).
More members do not raise the ceiling; better members would.

## From-scratch ensemble distillation (2026-09-27)

The earlier distillation started from an existing base, used a 2,000-sequence
train pool and alpha 0.5, and recovered only about 15% of the teacher's
margin. `ensdistill.py --scratch 128` instead trains a fresh GRU128. The
teacher is again the seven fine-tuned members for one validation half, with
alpha 0.75 (mostly imitation). Training data is that half plus 6,000 train
sequences with cached teacher predictions (the original 2,000-sequence pool
plus 4,000 more); train rows use mask-net weights. The schedule is
distill_mps's: 5 epochs at lr 1e-3, then 3e-4 from epoch 4.

Half 1 (teachers fine-tuned on odd groups, scored on even groups):

| Model | Scored WP on even groups |
|---|---:|
| Scratch student, epochs 1–5 | 0.678945 / 0.680664 / 0.679973 / 0.683053 / 0.682976 |
| Scratch student, SWA(e3–e5) | **0.683506** |
| Best regular FT single (GRU128 α0.75 s58, SWA) | 0.681474 |
| v12 members (4 models, 700 kMAC padded / 502 unpadded) | 0.683419 |
| Seven-member teacher | 0.683544 |
| 0.5 · student + 0.5 · mean(v12 members) | 0.684199 |
| 0.5 · student + 0.5 · mean(GRU128 s61, distilled GRU128 α0.75, GRU192) | 0.684252 (451 kMAC, about 43 minutes) |

A single GRU128 student (92 kMAC) matches the four-model v12 ensemble and
nearly the seven-member teacher; it recovers most of the teacher's margin.
Weighted about 0.5 against other members, it adds about +0.0008 on this half.
Half 0 is being checked on the CPU, and the final student (all-validation
teachers, seed 71) is training on the GPU. `package_ft.py` accepts
per-member weights (`spec@w`); outputs and output biases are weighted without
duplicating any member's compute.

### v15: final student packaged

The final student (GRU128, seed 71) trained on all validation plus 6,000
train sequences, with the seven all-validation fine-tuned members as teacher
(5 epochs; SWA e3–e5). `submission_v15.zip` weights it 0.5 and gives 1/6 each
to GRU128 α0.75 s61, the distilled GRU128 α0.75 and GRU192. That is the
best set under about 43 minutes on half 1 (0.684252, vs 0.683419 for the v12
members on the same half). It has two width groups (128 × 3, 192 × 1),
452 kMAC per row, about 43 minutes by the linear model; it matches PyTorch to
7.0×10⁻⁷ and is 1,696,967 bytes. Only half 1 validated the student recipe;
the leaderboard is the check for half 0. A second final student (seed 72)
is training for a two-student package.

### v16: two students

A second final student (seed 72, same recipe) finished with the same training
loss as seed 71 (0.3309 vs 0.3306). On 24 train sequences outside the
distillation pool, its t0 predictions correlate 0.957 with seed 71's, about
the same as seed 71 vs GRU128 s61 (0.958). The second student therefore
brings member-level diversity. `submission_v16.zip` weights the two students
0.3 each and gives 0.2 each to the distilled GRU128 α0.75 and GRU192:
452 kMAC per row, the same cost as v15. No held-out score exists for a
two-student set; the weights are a judgment extrapolated from v15's
half-1 optimum.

### Official results for v15 and v16 (2026-09-27)

| Package | Official score | vs v12 | Official runtime |
|---|---:|---:|---|
| v12 (padded, 700 kMAC) | 0.6583 | — | 56m01s |
| v15 (one student, grouped, 452 kMAC) | 0.6579 | −0.0004 | 46m11s |
| **v16** (two students, grouped, 452 kMAC) | **0.6586** | **+0.0003** | **45m32s** |

- Width grouping works on the official runner: at 452 kMAC it ran about
  46 minutes against 56 for the padded v12. The linear model predicted about
  43 minutes, so it runs a few minutes optimistic for grouped packages.
- v15 fell short of its half-1 estimate (+0.0008 held-out, −0.0004
  official). The student recipe and its 0.5 weight were chosen on one half
  only, so that estimate was optimistic. Adding the seed-72 student (v16) gained
  +0.0007 over v15, consistent with large seed variance for single students
  that averaging removes.
- Next: two more students (seeds 73 and 74). With five width-128 members (four
  students plus the distilled GRU128 α0.75), the package is one group at
  461 kMAC.

### Next: a teacher that learns different information (2026-09-27)

Leaderboard now: rank 52 at 0.6586. Rank 8 is 0.6643 (gap 0.0057), rank 20
is 0.6616 and rank 30 is 0.6603.

The v15 miss changes the protocol: any change is adopted only if it improves
both validation halves. Adding seeds or tuning weights is low priority; the
seed-73/74 student runs were stopped. The first new model is the split i0/i1
encoder from `advance.py` (on FIT-CV: +0.0023 standalone, and useful in
blends). One shared encoder (52→32, tanh) is applied to i0 and to i1, then
[i0, i1, i0 − i1, i0 · i1, encoded a0–a7] feeds a GRU. `distill_mps.py
--split` distils it at width 128 (α 0.75, seed 81, cached three-GRU teacher).
`maskft.py` builds the right architecture from the state dict. Plan, cheapest
first:
1. fine-tune on each half with mask weights; score it alone against the
   GRU128 members on both halves;
2. add it to the seven-member teacher using cached predictions; only if the
   teacher improves on both halves, re-distill students from the new teacher.

### Split encoder result (2026-09-28)

| Model (SWA e2/e3) | h0 | h1 | Mean |
|---|---:|---:|---:|
| Split encoder GRU128, frozen | 0.675469 | 0.675077 | 0.675273 |
| Split encoder, FT e3 | 0.682829 | 0.679045 | 0.680937 |
| **Split encoder, FT SWA** | 0.682895 | 0.681752 | **0.682323** (best single member) |
| GRU128 α0.75 s58, FT SWA | 0.682614 | 0.681474 | 0.682044 |
| Seven-member teacher | 0.684912 | 0.683520 | 0.684216 |
| Seven-member teacher + split encoder | 0.685008 | 0.683696 | 0.684352 |

The split encoder is the strongest single member and improves the teacher on
both halves (+0.000096 / +0.000176), so it passes the two-half rule. The gain
is small: a different input structure still learned mostly the same signal.
Re-distilling students from the eight-member teacher would need new per-half
and all-validation teacher caches plus two-half student runs (well over 12
hours of GPU time) for an expected gain of roughly +0.0001 to +0.0002.

### Event-time GRU (option C): negative (2026-09-29)

`eventgru.py` steps a GRU128 only at i0 price events (median 3,567 events per
sequence, 5.6× compression). Its input at each event is the event row, the
row just before it, the previous interval's length and the event count. Each
interval gets one prediction, and the loss is |y| × interval row count. The
base model trained on FIT (5 epochs, about 5 minutes each on the GPU); it was
then fine-tuned with mask weights on each validation half.

| Test | h0 | h1 | Mean |
|---|---:|---:|---:|
| Event GRU, frozen | 0.580588 | 0.583505 | 0.582047 |
| Event GRU, FT SWA(e2, e3) | 0.589032 | 0.594540 | 0.591786 |
| Eight-member teacher T | 0.685008 | 0.683696 | 0.684352 |
| 0.95 T + 0.05 event | 0.684602 | 0.683356 | 0.683979 (−0.00037) |
| 0.90 T + 0.10 event | 0.683857 | 0.682677 | 0.683267 (−0.00109) |
| Linear readout on v1 hidden (scored ×16) | 0.677083 | 0.677855 | 0.677469 |
| Linear readout on v1 hidden + frozen event hidden | 0.675184 | 0.676474 | 0.675829 |

The event model's predictions correlate 0.83 with the teacher, yet every
blend weight lowers both halves. Its hidden state also lowers a linear readout
that already has v1's hidden state. The longer price history carries no
usable signal beyond what the row-level GRUs already hold, so the hybrid
(event state fed into a row-level GRU) is not pursued.

### Dedicated spread-component model (`sgru.py`)

Replacing only S with the truth, and keeping v1's D, would lift validation
from 0.6756 to 0.7238. By tick group the pooled gain would be +0.020 (0.125),
+0.010 (0.241), +0.003 (0.469) and +0.020 (0.657). The joint models barely
predict S in the 0.657 group (corr 0.08). A GRU128 trained on FIT for S alone
(standardized plus event inputs, weight |t0| + |t1|, 4 epochs) predicts it
better on both halves:

| corr(pred S, S), scored rows | Eight-member ensemble | Dedicated model |
|---|---:|---:|
| all, h0 / h1 | 0.618 / 0.618 | 0.642 / 0.639 |
| tick 0.469, h0 / h1 | 0.250 / 0.246 | 0.414 / 0.413 |
| tick 0.657, h0 / h1 | 0.095 / 0.081 | 0.221 / 0.200 |

Swapping its S into the ensemble, (t0, t1) = ((D − S′)/2, (−D − S′)/2) with
S′ = (1 − a)·S_ensemble + a·S_dedicated, still lowers the score: a = 0.25
gives 0.684126 vs 0.684352, and larger a is worse. The dedicated model has no
validation fine-tuning or mask weighting, so its level and scale by group do
not match the fine-tuned ensemble. A cross-fitted linear stack (ensemble t0/t1,
S_dedicated, tick-group intercepts; fit on one half, scored on the other) did
not help either:

| Cross-fitted stack | h0 | h1 | Mean |
|---|---:|---:|---:|
| Ensemble t0/t1 + group intercepts | 0.684461 | 0.683429 | 0.683945 |
| + S_dedicated | 0.684434 | 0.683411 | 0.683923 |
| + S_dedicated × tick group | 0.684331 | 0.683314 | 0.683823 |
| Plain ensemble (reference) | 0.685008 | 0.683696 | 0.684352 |

Recalibration alone loses 0.0004 across halves, and S_dedicated adds nothing
on top. Its better S correlation is information the ensemble already has
through t0 and t1. The S route is closed.

### Two-half validation for distilled students

v15 showed that one-half student evidence can mislead. Students are now
trained per half from the cached per-half teachers: h0 seed 71 first, then
seed 72 on both halves. Student ensembles and their weights will be chosen on
both halves.

### Two-half student results (2026-09-30)

The half-0 student (seed 71, from the h0 teachers) repeats the half-1 pattern:
epochs 1–5 score 0.678639 / 0.680891 / 0.680658 / 0.684167 / 0.684585 on
odd groups.

| Set (weights) | h0 | h1 | Mean | Official |
|---|---:|---:|---:|---:|
| Student SD71 alone, SWA(e3–e5) | 0.684212 | 0.683506 | 0.683859 | — |
| Best other single (split encoder) | 0.682895 | 0.681752 | 0.682323 | — |
| v12 set (v1, ED, GRU192, GRU192 α0.75) | 0.684948 | 0.683419 | 0.684183 | 0.6583 |
| v15 set (SD71 0.5; GRU128 s61, ED, GRU192 1/6 each) | 0.685072 | 0.684252 | 0.684662 | 0.6579 |
| SD71 0.4; ED, GRU192, split 0.2 each (464 kMAC) | 0.685337 | 0.684326 | 0.684831 | — |

The student recipe holds on both halves: one student beats every other
single model by about +0.0015, and student-weighted sets beat the
eight-member teacher (0.684352). Yet v15 missed by about 0.0009 (held-out
+0.0005 over v12, official −0.0004). The all-validation student is a
separate training run from the per-half students, so a single final student
can be an unlucky draw. Public-leaderboard noise at this scale is also
plausible. v16 (two students) recovered to +0.0003. The next package will
therefore use two students, chosen once the seed-72 per-half students
finish. The split encoder has been fine-tuned on all validation for
packaging.

Runtime model refit on grouped packages (v1 22m16s at 60 kMAC; v16 45m32s at
452 kMAC): about 18.8 + 59.2 × (kMAC / 1000) minutes. The cap for new
packages is about 480 kMAC (about 47 minutes).

### Seed-72 students and v17 (2026-10-01)

| Student (SWA e3–e5) | h0 | h1 | Mean |
|---|---:|---:|---:|
| SD71 | 0.684212 | 0.683506 | 0.683859 |
| SD72 | 0.684135 | 0.681501 | 0.682818 |

Same recipe, different seed: 0.002 apart on half 1. Student quality varies
a lot from run to run, which explains v15: a single final student can be an
unlucky draw. Two-student sets on both halves:

| Set (students' total weight) | h0 | h1 | Mean | kMAC |
|---|---:|---:|---:|---:|
| v16 composition: SD71, SD72 (0.6); ED, GRU192 | 0.685399 | 0.683909 | 0.684654 | 452 |
| SD71, SD72 (0.6); GRU192, split | 0.685445 | 0.684081 | 0.684763 | 464 |
| SD71 alone (0.4); ED, GRU192, split | 0.685337 | 0.684326 | 0.684831 | 464 |

The single-student set scores highest only because SD71 was a lucky run, and
the final students are separate runs. `submission_v17.zip` therefore uses
two students (the all-validation SD71 and SD72, 0.3 each) with GRU192 and the
split encoder (0.2 each). It has three groups (two GRU128 students, one
GRU192, one split encoder), 464 kMAC per row, about 46 minutes by the refit
runtime model. It matches PyTorch to 5.1×10⁻⁷ and is 1,737,771 bytes. Expected
change vs v16 is about +0.0001, within leaderboard noise.

### v18: four students (2026-10-01)

Final students with seeds 73 and 74 (same recipe as 71 and 72, all
validation, 5 epochs at about 20 minutes each on AC power) ended at training
loss 0.3306 and 0.3301, matching seeds 71 and 72. Because single students
vary from run to run, `submission_v18.zip` averages four students (seeds
71–74, SWA e3–e5) plus the split encoder, 0.2 each. It has two groups
(four GRU128 students batched, one split encoder), about 474 kMAC per row and
about 47 minutes by the refit runtime model. It matches PyTorch to
4.5×10⁻⁷ and is 1,773,661 bytes. No held-out score exists for a four-student
set, since per-half students exist only for seeds 71 and 72; the design aims
to lower variance rather than chase a validated gain.

### Official v17/v18 and loss experiments (2026-10-01)

| Package | Official score | Runtime |
|---|---:|---:|
| v16 | 0.6586 | 45m32s |
| v17 | 0.6586 | 54m26s |
| **v18** | **0.6587** | 51m01s |

Three fine-tuning changes were tested on the split-encoder member with the
same seed and validation halves. `target-balanced` computes each target's
weighted MSE separately and averages the two, matching the official metric's
equal treatment of t0 and t1. `chunk 1024` doubles truncated-BPTT length from
512 rows without changing inference.

| Split encoder, SWA(e2, e3) | h0 | h1 | Mean | vs reference |
|---|---:|---:|---:|---:|
| Reference, chunk 512 | 0.682895 | 0.681752 | 0.682323 | — |
| **Target-balanced, chunk 512** | **0.682971** | **0.682128** | **0.682550** | **+0.000227** |
| Original loss, chunk 1024 | 0.682495 | 0.682189 | 0.682342 | +0.000019 |
| Target-balanced, chunk 1024 | 0.682517 | 0.682383 | 0.682450 | +0.000127 |

Target balancing is the only change retained. It improves both halves and is
better without the longer chunk. Replacing the original split member in the
eight-member teacher also improves both halves: 0.685008 → 0.685028 and
0.683696 → 0.683775 (mean 0.684352 → 0.684401).

`submission_v19.zip` replaces v18's split member with the all-validation
target-balanced split model; the four students and equal weights are
unchanged. It is 1,773,657 bytes, matches PyTorch to 5.3×10⁻⁷, and has the
same inference structure and compute as v18. The expected gain is small
because the changed member has weight 0.2.

### Alternative output heads and v20 (2026-10-01)

Three heads were trained on the frozen split encoder on each validation half:
a target-balanced linear control, a 33-bin categorical distribution whose
prediction is its expected value, and two regression experts mixed by a
learned scoring-mask probability. SWA averages epochs 2 and 3.

| Frozen recurrent head | h0 | h1 | Mean |
|---|---:|---:|---:|
| Linear control | 0.678502 | 0.678972 | 0.678737 |
| 33-bin distribution | 0.675219 | 0.673960 | 0.674589 |
| Mask-gated experts | 0.678724 | 0.679131 | 0.678927 |

Mask gating improves the matched linear control by +0.00022 / +0.00016, but
is too weak to help the existing teacher when blended. The distribution head
is weaker alone but makes different errors: giving it 10% of the eight-member
teacher improves h0 0.685028 → 0.685189 and h1 0.683775 → 0.683976.

To avoid running a second split GRU, the distribution head was retrained on
the frozen target-balanced split member. Inside the v17 two-student proxy,
replacing 25% of the split member's output with the distribution expectation
improves h0 0.685469 → 0.685664 and h1 0.684205 → 0.684242 (mean +0.000116).
A hard argmax-bin output is faster but lowers h1 and is rejected.

`submission_v20.zip` applies the same 75/25 mixture inside v19's split member,
sharing its encoder and recurrent state. It is 1,805,948 bytes and matches
PyTorch to 5.7×10⁻⁷. Local callback time rises from 0.71 s to 0.89 s per
20,000 rows because 66 logits and a softmax are evaluated per row. Official
runtime is therefore the main risk; the score gain is expected to be small.

## Round 6: searching for a larger step (2026-10-01/02)

Official: v17 0.6586 (54m26s), v18 0.6587 (51m01s). v18 is the best public
score, but the split-encoder group cost more official time than the runtime
model predicted (v17 +8 minutes). Leaderboard rank 83 of 604; the field keeps
rising (rank 50 at 0.6606).

Negative results this round (all two-half):
- **Price-state lookup for S.** i0 price features are global, discrete book
  shape codes: 118 distinct states across 240 sequences of the 0.657 group,
  99.5% of validation scored rows have a state seen in train, and trade-price
  features also sit on a global grid. Within train, state → S lookup on
  held-out sequences gives corr 0.43, but on validation only 0.22, whether
  the table is built from train or from the other validation half. Adding the
  last trade price or the previous state does not raise purity (0.78). Mixing
  lookup S into the ensemble: −0.0005 at weight 0.25; +0.0002 at 0.1, but the
  halves disagree.
- **Label-only GRU256 (no distillation).** Frozen validation score 0.66578,
  against 0.67636 for the distilled GRU192. Its training loss fell lower
  (0.924), so it overfits. Distillation, not width, is the lever.
- **Smoothed training targets.** A linear readout on v1's hidden state fitted
  to forward means of the targets over 10/50/200 rows scores 0.6746 / 0.6619 /
  0.6127 against 0.6775 with raw targets. Distillation helps by averaging
  model hypotheses, not by smoothing labels in time.

## Round 7: public ideas and a second-generation teacher (2026-10-02)

Public leaderboard on 2026-10-02: v18 0.6587 is rank 87; rank 5 is 0.6659 and
rank 1 is 0.6677. The competition Discord needs a login, so it was not read.
Public write-ups of earlier Wunder challenges were checked for general ideas
only (no code used). One untested idea: a weighted-Pearson loss term.

**Pearson loss.** `maskft.py --corr W` adds W · (1 − |y|-weighted Pearson of
the chunk) to the fine-tuning loss, with the same row weights; the term
matches `GlobalWP` exactly on a synthetic check. Result on v1, corr 1, both
halves: per epoch h0 0.67985 / 0.68143 / 0.681925 and h1 0.678898 / 0.677839 /
0.678371; SWA(e2, e3) 0.682006 / 0.679669 (mean 0.680837) against 0.681944 /
0.679740 (0.680842) without it. No effect; rejected. The |y|-weighted MSE
already targets the metric.

**Second-generation teacher.** `ensdistill.py --gen 2` uses
0.5 · mean(seven gen-1 members, target-balanced split encoder, distilled
GRU128) + 0.5 · mean(scratch students: 71 and 72 per half; 71–74 for all
validation). It reuses the cached gen-1 average and runs only the four added
models. On both halves it scores 0.685443 / 0.684061 (mean 0.684752) against
about 0.6844 for the gen-1 teacher; the cached-mixture check matches to
5.9×10⁻⁷. Students use seed 71 so they compare directly with SD71.

**Gen-2 student results (seed 71, same recipe as SD71).** Per epoch on the
held-out half: h0 0.678744 / 0.681168 / 0.681218 / 0.684181 / 0.684673, h1
0.678729 / 0.681329 / 0.679707 / 0.683866 / 0.683096. SWA(e3–e5):

| Set | h0 | h1 | Mean |
|---|---:|---:|---:|
| SD71 (gen-1) | 0.684212 | 0.683506 | 0.683859 |
| **G71 (gen-2)** | **0.684436** | **0.683962** | **0.684199** |
| SD71 + SD72 | 0.685042 | 0.683501 | 0.684272 |
| SD71 + G71 (same seed) | 0.684574 | 0.683989 | 0.684282 |
| SD72 + G71 | 0.685205 | 0.683794 | 0.684500 |
| v18-like: SD71, SD72 0.4 each; split 0.2 | 0.685281 | 0.683988 | 0.684634 |
| SD71, SD72, G71 0.8; split 0.2 | 0.685404 | 0.684411 | 0.684908 |
| SD71, SD72, G71 0.6; GRU192, split 0.2 | 0.685587 | 0.684553 | 0.685070 |

The gen-2 student beats its gen-1 counterpart by +0.00034 (both halves up)
and still trails its teacher (0.684752). With the same seed it is highly
correlated with SD71 (SD71 + G71 adds almost nothing over SD71 + SD72), so
final gen-2 students use new seeds (75, 76) for diversity. The
all-validation gen-2 teacher (four all-validation students 71–74) is cached.

## Round 8: target horizon and new-family screens (2026-10-02)

`horizon.py` fits weighted linear oracle models on the locked pilot FIT
sequences and scores the disjoint pilot DEV sequences. Future values are used
only to diagnose label construction and are never inference inputs. Fixed row
horizons add little, but the next i0 price event is decisive:

| Oracle input | DEV Global WP |
|---|---:|
| Current i0 price/trade-price state | 0.256515 |
| Next-event change only | 0.694685 |
| Current state + next-event change | **0.752467** |
| 2 / 4 / 8 events ahead | 0.602206 / 0.506453 / 0.427832 |

The target is therefore strongly tied to the immediately following i0 price
event, not to a fixed row horizon. A causal model still has to forecast that
event; the oracle score is not an achievable prediction result.

Two cheap applications failed the predeclared +0.0005 gate on validation half
0 and were stopped before further tuning:

- A next-event 22-dimensional price-change auxiliary loss (`maskft.py
  --event-aux 0.1`) scored 0.679418 / 0.680765 / 0.681661 over epochs 1--3,
  below the matched v1 control 0.679694 / 0.681310 / 0.681858.
- A zero-initialized kernel-5 causal Conv1D residual before v1
  (`--causal-conv 32`) scored 0.679637 / 0.681095 over epochs 1--2, versus
  0.679694 / 0.681310 for the control. Epoch 3 and the other half were skipped.

`flow_screen.py` then cross-fitted a linear residual on the two validation
halves using fine-tuned v1 predictions plus causal i0 one-row changes,
within-price-interval changes and event age. The prediction-only control was
0.680911; flow features lowered it to 0.678990 on the mean and lowered both
halves (gain -0.001921). This route is closed.

**v21: two gen-1 and two gen-2 students.** Final gen-2 students (all
validation, seeds 75 and 76) ended at training loss 0.3301 and 0.3292.
`submission_v21.zip` = all-validation SD71, SD72 (gen-1), G75, G76 (gen-2)
and the target-balanced split encoder, 0.2 each: the same structure and
compute as v19 (474 kMAC, about 51 minutes officially for v18). It matches
PyTorch to 5.2×10⁻⁷ and is 1,774,002 bytes. The NumPy "invalid value
encountered in matmul" warning while building the split encoder also appears
for v19 and comes from macOS Accelerate; all weights are finite. Held-out
analogue: SD71, SD72, G71 0.8 + split 0.2 scored 0.684908 against 0.684634
for the v18-like set, so the expected gain is about +0.0003, within
leaderboard noise. Seeds 77 and 78 are training for an all-gen-2 set.
(`submission_v20.zip` was built separately with a distribution head in
`multi_solution.py`; v21's members do not use it.)

**v22: four gen-2 students.** Seeds 77 and 78 ended at training loss 0.3283
and 0.3288. `submission_v22.zip` = all-validation G75–G78 and the
target-balanced split encoder, 0.2 each (same structure as v19/v21), matches
PyTorch to 5.2×10⁻⁷, 1,774,031 bytes. Per-half gen-2 students with seed 72
are training to check whether the gen-2 gain survives a second seed (SD72 vs
G72) and whether an all-gen-2 set beats the mixed v21 set.

**Seed-72 check: the gen-2 gain is mostly seed noise.**

| Set (SWA e3–e5) | h0 | h1 | Mean |
|---|---:|---:|---:|
| SD72 (gen-1) | 0.684135 | 0.681501 | 0.682818 |
| G72 (gen-2) | 0.683939 | 0.681321 | 0.682630 |
| SD71 + SD72 | 0.685042 | 0.683501 | 0.684272 |
| G71 + G72 | 0.685115 | 0.683708 | 0.684412 |
| v19-like: SD71, SD72 0.8; split 0.2 | 0.685281 | 0.683988 | 0.684634 |
| v22-like: G71, G72 0.8; split 0.2 | 0.685396 | 0.684211 | 0.684804 |
| v21-like: SD71, SD72, G71, G72 0.8; split 0.2 | 0.685424 | 0.684188 | 0.684806 |

Per seed, gen-2 minus gen-1 is +0.00034 (seed 71) and −0.00019 (seed 72). Pairs
and package-like sets gain about +0.00015, both halves positive but well
inside seed and leaderboard noise. v21 and v22 are therefore expected to tie
v19 within about 0.0002. Distilling from a stronger ensemble has saturated:
students stay about 0.0005–0.002 below the teacher, and seed-to-seed spread
(0.0015) dominates.

**Weight EMA for students (`ensdistill.py --ema`).** Same seed-71 gen-2 run
(raw trajectory reproduced exactly), with EMAs of the weights at decays 0.9998
(about half an epoch) and 0.99995 (about 2.3 epochs):

| Seed-71 gen-2 student | h0 | h1 | Mean |
|---|---:|---:|---:|
| SWA(e3–e5) of epoch ends (G71) | 0.684436 | 0.683962 | 0.684199 |
| EMA 0.9998 at e5 | 0.685177 | 0.683976 | 0.684576 |
| EMA 0.9998, mean of e4–e5 | 0.685092 | 0.683923 | 0.684508 |
| EMA 0.99995 at e5 | 0.683147 | 0.684037 | 0.683592 |
| G71 + SD72 | 0.685205 | 0.683794 | 0.684500 |
| EMA e5 + SD72 | 0.685448 | 0.683627 | 0.684537 |
| G71, G72 0.8; split 0.2 | 0.685396 | 0.684211 | 0.684804 |
| EMA e5, G72 0.8; split 0.2 | 0.685587 | 0.684100 | 0.684844 |

EMA improves a single student by +0.0004 (h0 only), but inside a set the
gain is +0.00004: averaging over weights and averaging over members remove
the same noise. Final students are not retrained with EMA.

**Other negatives this round.**
- *MLP readout.* On the cached frozen hidden states (5 GRUs, 480 units,
  5.55M rows), a readout fitted on one half and scored on the other gives
  0.675842 linear vs 0.640623 with a 256-unit GELU MLP (same Adam training):
  a nonlinear head fitted on validation-period data overfits badly.
- *Targets vs the next price block (0.657 group, 557,834 train blocks).*
  Purity of the (t0, t1) tick pair is 0.702 given the current i0 price block,
  0.816 given current and next, 0.831 with the next two, and 0.801 given
  previous and current. Split by component, S = −(t0 + t1) is 0.907 pure given
  the current block alone, and the next block adds nothing; D = t0 − t1 behaves like
  the pair. The block is a book-shape code without absolute price, so the next
  block does not reveal the move. A "predict the next block, then map through
  a table" head is not viable.

Status after round 7: v21 (two gen-1 + two gen-2 students + split) and v22
(four gen-2 students + split) are packaged; both are expected within about
±0.0002 of v18/v19. Distillation generations, EMA, loss variants, and readout
changes have all saturated near 0.6848 held-out. Closing the 0.007 public gap
to rank 5 needs a new information source, not more averaging.

## Round 9: interval state, i1 flow and LSTM (2026-10-02)

A minimal dual-timescale proxy adds a zero-initialized nonlinear residual to
v1 using causal i0 statistics that reset at each i0 price event: one-row
changes, changes since the event, running interval-mean changes and event age.
On half 0 it scored 0.679504 / 0.681269 at epochs 1 / 2, versus the matched v1
control 0.679694 / 0.681310. It did not meet the +0.0005 gate, so epoch 3, the
other half and a richer resettable fast GRU were skipped.

The cross-fitted flow screen was repeated for i1, whose event history had not
been included in `features.py`. Adding i1 one-row changes, changes since its
own price event and event age to fine-tuned v1 lowered the two-half mean from
0.680911 to 0.680586 (gain -0.000325); both halves fell. Explicit i0 and i1
event-flow features therefore add no linear residual signal.

Finally, a native PyTorch LSTM128 was trained from scratch against the same
gen-2 teacher, validation half and 6,000-sequence train pool as the gen-2
GRU128. Stateful full/chunk inference agreed in a self-check. At epoch 1 the
LSTM scored 0.678136, below the matched GRU's 0.678744 by 0.000608, while
taking 1,508 seconds. The run was stopped before epoch 2 under the fixed gate;
the other half and seeds were not run.

No new-family screen passed. Of the ready packages, v21 is the conservative
next submission: it mixes two gen-1 and two gen-2 students plus the
target-balanced split member, while v22 replaces all four students by gen-2
models for no measurable held-out advantage.

## Round 10: next-event pretraining and hard bottleneck (2026-10-02)

`event_pretrain.py` used only the locked 128 FIT sequences to pretrain a GRU96
to predict the next i0 price-event delta, then trained t0/t1 on those FIT
sequences and scored the disjoint 32-sequence DEV set. The control and
pretrained models had identical recurrent and target-head initialization and
identical target-training order. Event loss fell from 0.9086 to 0.7557.

| Pilot model | Best DEV Global WP | vs control |
|---|---:|---:|
| Target-only GRU96 | **0.316064** | — |
| Event-pretrained GRU96, then target fine-tuning | 0.314421 | −0.001643 |
| Hard path: GRU → predicted event delta → t0/t1 | 0.312516 | −0.003548 |

Pretraining was ahead at target epoch 2 (0.296065 vs 0.279765), but the
control caught and passed it by epoch 3. This is an optimization warm start,
not a stronger converged representation. Forcing target prediction through
the event forecast also reduced the score, so neither variant passes the
pilot gate and neither is scaled to the full training set.

**Low-LR continuation of students (`ensdistill.py --init ... --epoch-lrs`).**
Each per-half gen-2 student continues from its epoch-5 checkpoint for three
more epochs at lr 1e-4, 1e-4, 3e-5 (fresh Adam, same teacher and data mix).
Epoch 0 reproduces the e5 checkpoint score exactly, so the comparison is
paired. "C" = SWA over the three continuation epochs.

| Set | h0 | h1 | Mean | vs original |
|---|---:|---:|---:|---:|
| G71 → C71 | 0.685150 | 0.684065 | 0.684607 | +0.00041 |
| G72 → C72 | 0.684934 | 0.681876 | 0.683405 | +0.00078 |
| C71 + C72 (G71 + G72: 0.684412) | 0.685884 | 0.683896 | 0.684890 | +0.00048 |
| C71, C72 0.8; split 0.2 (G: 0.684804) | 0.686021 | 0.684340 | **0.685181** | **+0.00038** |
| SD71, SD72, C71, C72 0.8; split 0.2 (G: 0.684806) | 0.685767 | 0.684288 | 0.685027 | +0.00022 |

Unlike weight EMA, the gain survives ensembling (both halves up in every
row): the original schedule stopped while the imitation loss was still
falling (0.3236 → 0.3191 on h0). Final gen-2 students G75–G78 get the same
continuation for the next package.

## Round 11: exhaustive cached ensemble and cheap cell screens (2026-10-02)

`ensemble_screen.py` cross-fits all cached conventional, gen-1 and gen-2
members. Fitted global or tick-specific weights add nothing material to the
v21 proxy (0.684815 / 0.684801 vs 0.684806), confirming that equal weights
remain the robust choice. Searching equal-weight subsets under a 480 kMAC
budget selects a GRU192, the target-balanced split encoder and two students.

| Deployable four-member structure | h0 | h1 | Mean | kMAC |
|---|---:|---:|---:|---:|
| GRU192 + split + SD71 + G71 | 0.685444 | 0.684711 | 0.685078 | 465 |
| GRU192 + split + C71 + C72 | **0.685946** | **0.684467** | **0.685207** | 465 |

The first structure is packaged as `submission_v23.zip` using the analogous
all-validation SD71 and G75 checkpoints. It is 1,738,119 bytes and matches
PyTorch to 5.8e-7. The continued-student structure is stronger on the mean
and improves over the v21 proxy on both halves; it becomes the next package
after final G75/G76 continuation finishes.

The locked 128-FIT / 32-DEV pilot also screened recurrent cells at roughly
the GRU96 compute scale:

| Cell | Best pilot DEV Global WP | vs GRU96 0.316064 |
|---|---:|---:|
| Vanilla RNN128 | 0.302053 | -0.014011 |
| Two-layer GRU64 | 0.317995 | +0.001931 |
| Target-specific dual GRU64 | 0.314269 | -0.001795 |

None clears the predeclared +0.005 scale-up gate. The official docs mention
Mamba-2, but the reference package is built around CUDA/Triton kernels and no
CPU-ready dependency is installed; a full run is deferred unless a
single-core callback prototype first fits the runtime budget.

**v24: continued gen-2 students.** Final students G75–G78 continued for
three epochs at 1e-4, 1e-4, 3e-5 (training loss 0.3301/0.3292/0.3283/0.3288
→ 0.3240/0.3237/0.3242/0.3246). `submission_v24.zip` = SWA over the three
continuation epochs for each of the four students, plus the target-balanced
split encoder, 0.2 each (same structure and compute as v19/v22). It matches
PyTorch to 5.0×10⁻⁷ and is 1,774,077 bytes. Held-out analogue: +0.00038 over
the v22-like set and +0.00055 over the v19-like set. (A duplicate seed-75
continuation with identical arguments ran concurrently from another session;
either run's checkpoints are the same recipe.)

**Second continuation: saturated.** Three more epochs from the continued
students at 5e-5, 3e-5, 1e-5 ("D", SWA over them): C71 0.684607 → D71 0.684643,
C72 0.683405 → D72 0.683341; pair + split 0.685181 → 0.685214 (+0.00003).
The first continuation captured the gain; v24 stays as packaged.

**Gen-3 teacher check.** Replacing the gen-1 students in the teacher with the
continued gen-2 students: 0.5 · FT9 + 0.5 · mean(C71, C72) scores 0.685154
(gen-2 teacher 0.684752), no better than the deployable C71, C72 + split set
(0.685181). Another distillation generation is not pursued.

**Validation-weighted continuation.** Same continuation from G71/G72 e5 but with
the fitting half's validation sequences repeated three times per epoch
(`--valid-repeat 3`), testing whether more weight on the validation period
helps; compared with C71/C72 from the same start.

The G71/half-0 gate reproduced epoch 0 at 0.684673, then scored 0.684635
after epoch 1 (loss 0.305040, 1,584 seconds). This is 0.000233 below the
ordinary C71 epoch-1 score of 0.684868 and slightly below its own start.
The run and the remaining three half/seed variants were stopped.

**Final Pareto packages.** `submission_v25.zip` uses C75 and C76 at weights
0.4 each plus the target-balanced split member at 0.2. It is 1,083,265 bytes,
matches PyTorch to 5.8e-7, and took 0.62 seconds per 20k rows locally. Its
held-out structural proxy is 0.685181 at about 290 kMAC. `submission_v26.zip`
adds GRU192 and uses four equal weights: 1,738,106 bytes, 5.4e-7 agreement,
0.83 seconds locally, and proxy 0.685207 at about 465 kMAC.

Reducing the baseline member's unscored-row weight from 0.06 to 0.02 improved
its SWA score on both halves (0.682059 vs 0.681944; 0.679913 vs 0.679740),
mean +0.000144. It did not improve the deployable ensembles: adding it to the
equal-weight continued safe set changed 0.685167 to 0.685032, and adding it
to the four-member score set changed 0.685207 to 0.685118. No all-validation
refit is warranted.

**Selective SSM pilot.** A dependency-free Mamba-like block used input-dependent
decay, write and read vectors with width 96 and state size 8. Full/chunk output
and gradients agreed, and a NumPy inference-kernel proxy took 0.205 seconds per
20k rows on this host. On the locked 128-FIT / 32-DEV pilot its score rose from
0.266774 at epoch 1 to 0.299261 at epoch 5, still 0.016803 below GRU96's
0.316064. Runtime feasibility alone is insufficient; this implementation is
not scaled up.

**Local causal attention pilot.** A stateful width-64, four-head block attended
to the previous 64 rows with learned lag biases. CPU and MPS self-checks showed
that full and chunked inference agree. On the same locked pilot it peaked at
0.303908 at epoch 3, 0.012156 below GRU96, then finished at 0.301441. It also
fails the scale-up gate.

**Residual signal test (LightGBM, cross-fit by half).** Target: residual of
the v24-like set (C71, C72 0.4 each; split 0.2) on held-out scored rows.
Inputs: the 112 raw features, their one-row differences, step, tick group and
the set's own predictions (228 columns); |y| weights; 300 trees, 63 leaves.

| Fit → test | Set | Corrected | Half-step correction |
|---|---:|---:|---:|
| h0 → h1 | 0.684340 | 0.678491 | 0.684413 |
| h1 → h0 | 0.686021 | 0.677187 | 0.684199 |

The top split features are the step index and a few volume columns, which
are sequence-specific and do not transfer. Current-row inputs hold no usable
signal beyond what the ensemble already extracts; better input
representations of the current row are not a lever.

**Validation-weighted continuation: stopped.** With validation repeated three
times per epoch, seed 71 scored h0 e1 0.684635 (plain continuation 0.684868)
and h1 e1/e2 0.683733/0.684007 (0.683968/0.683956): −0.0002 to +0.00005, no
sign of a large effect, so the runs were stopped to free compute.

## Round 8: screening for large effects (2026-10-03)

Goal: top-10 public (about 0.665, +0.006). Gains so far are +0.0003–0.0005,
below the single-seed noise of a from-scratch student (about 0.0015), so new
ideas are screened only for large effects (≥ +0.002 on both halves). Screen 1:
a GRU256 student, same recipe as G71 (gen-2 teacher, 5 epochs, then the
low-LR continuation), compared with G71/C71.

**Callback overhead.** v24's callback takes 35.4 µs per row on this Mac, of
which the matrix products alone take about 17.5 µs: about half is NumPy call
overhead (about 30 small calls per row). The scorer image ships numba 0.67.0,
torch (CPU), onnxruntime and jax, so a fused numba step could free runtime for
a larger model. Worth building only if a larger student clears the screen.
(Caveat from v12: cutting NumPy calls by about 40% did not shorten the official
run, so the official machine is bound by weight traffic, not call overhead. A
real speed-up needs fewer weight bytes, e.g. half-precision weights in a numba
kernel, not just fusion. Deferred until a larger model proves worth it.)

**Screen 1 (GRU256 student): rejected.** Same recipe as G71, seed 71. Epochs
1–5: h0 0.679585 / 0.680242 / 0.679772 / 0.684291 / 0.684456 and h1 0.677832 /
0.681401 / 0.678512 / 0.681035 / 0.681232, against G71's e5 0.684673 /
0.683096. Training loss is lower (0.3220/0.3321 vs 0.3235/0.3336), held-out is
lower: extra width overfits. Width is not the bottleneck. (GRU256 trains as
fast as GRU128 on MPS, 21 minutes per epoch.)

**Screen 2 (full train pool): started.** `teacher_cache` now extends the pool
beyond 6,000 with a third segment "y" (the remaining 4,607 train sequences);
pools of at most 6,000 draw exactly the same sequences as before (checked).
Gen-1 and gen-2 teacher predictions are being cached for the new segment.

**Residual test, regularized.** Without the step feature (fewer leaves,
min_child_samples 5000, L2 10), shrunk corrections give h1 +0.0001 at
shrink 0.1 and h0 −0.00007; at 0.25 and above both halves fall, and the
0.657 group falls on both halves. The selected features (a6, a3, i0 volume
columns) do not transfer. Current-row inputs are exhausted.

**Scoring-mask structure** (94 validation sequences). Scored fraction per
sequence: mean 0.091, range 0.010–0.450. Burst lengths: median 2, 90th
percentile 8, 44% are single rows; gaps: median 6, 90th percentile 88.
Burst starts barely line up with observable events: an i0 price change at the
start 0.30 vs a 0.28 base rate, i0 trade 0.28 vs 0.16, i1 trade 0.36 vs 0.30.
The mask looks like a clustered random process modulated by activity, not a
deterministic function of events; mask-weighted fine-tuning already
captures what is usable.

**Screen 2 (full train pool): rejected.** Same seed and compute as G71/C71 but
each epoch's 6,000 train sequences are drawn from all 10,607: e5 h0 0.684583 /
h1 0.683090 (G71 0.684673 / 0.683096); after the continuation, e3 h0 0.685216 /
h1 0.683576 (C71 0.685131 / 0.683927). No effect: more unique train
sequences do not help. (Gen-1/gen-2 teacher caches for the extra 4,607
sequences took about an hour per split.)

**Train → validation feature shift** (300 sequences each, in units of the
train standard deviation). Largest mean shifts: i1_v18 −0.53, i1_v13 −0.36,
i0_p3 +0.31; group means |shift| 0.03–0.15. Within-sequence standard
deviations are about 1.05–1.2× larger in validation (i1 most). The AUC 0.976
separability comes from many small shifts, not a large one in any feature;
per-sequence input normalization is not a promising large lever.

**Round 8 conclusion.** No large effect found: GRU256 (worse, overfits), full
train pool (no change), residual signal in current-row inputs (none, two
variants), mask structure (clustered random process, already used), feature
shift (small). Together with the closed directions in DIRECTIONS.md, the
GRU-distillation pipeline is at its ceiling near 0.685 held-out (v24-like
0.685181). v24 remains the best package (expected public ≈ 0.6592).

## Round 12: v25/v26 leaderboard result (2026-10-04)

v25 scored **0.6590** in 39m21s; v26 scored **0.6589** in 54m37s. The local
proxy ordered them v26 0.685207 > v25 0.685181, but the official score reversed
that 0.000026 difference. This is expected at the measured uncertainty scale,
not evidence that the GRU192 member is intrinsically harmful. Operationally,
however, v25 strictly dominates: +0.0001 official score, 15m16s less runtime,
and 38% fewer estimated MACs.

The continuation result transferred only weakly: v25 is +0.0003 over v18's
0.6587. The public experiment therefore closes GRU width, seed, continuation,
and ensemble-weight tuning. Future submissions require a preregistered large
effect from a causally valid, structurally different representation; gains at
the 0.0001–0.0005 scale are no longer submission-worthy.

## Round 13: causal multi-scale window pilot (2026-10-04)

`multiscale_screen.py` tests a 96-wide feed-forward mixer over the current row
and causal differences between rolling means at 4, 16 and 64 rows. It has no
future access; CPU and MPS full/chunk self-checks agree. The locked FIT/DEV
split, normalizer, loss, schedule and scoring path are the same as the GRU96
control. All three initializations peaked at epoch 3:

| Seed | t0 | t1 | Global WP | vs GRU96 0.316064 |
|---:|---:|---:|---:|---:|
| 7 | 0.378319 | 0.277761 | 0.328040 | +0.011976 |
| 8 | 0.370053 | 0.274061 | 0.322057 | +0.005993 |
| 9 | 0.383783 | 0.279736 | 0.331760 | +0.015696 |
| Mean | 0.377385 | 0.277186 | 0.327286 | +0.011222 |

This is the first new-family screen to clear the preregistered +0.005 pilot
gate for every seed. Most of the mean advantage comes from t1 (+0.0178 versus
the control's epoch-3 t1), while mean t0 improves by about +0.0047. Epochs 4–5
fell for every seed, making a fixed three-epoch schedule the next test.

DEV contains only 32 sequences, the result has now been inspected, and
train→validation shift remains severe. The model therefore advanced only to a
larger group-CV gate, not directly to a submission.

**Larger-fold rejection.** With the epoch-3 schedule frozen, the first
1,024-train/512-test fold scored 0.386436 (t0 0.382623, t1 0.390250), versus
0.405980 for the normalized GRU96 on that fold. The pilot gain did not
generalize. A final diversity check mixed the candidate into already-frozen
ensembles on the same 512 sequences:

| Blend | Reference | With mixer | Gain |
|---|---:|---:|---:|
| Three-GRU ensemble, 10% mixer | 0.413768 | 0.414596 | +0.000827 |
| Three-GRU ensemble, 25% mixer | 0.413768 | 0.414339 | +0.000571 |
| Ensemble + split + wide, 10% mixer | 0.418084 | 0.418564 | +0.000480 |

The small positive blend confirms different errors but misses the +0.002
promotion gate by a wide margin. Remaining folds, full validation training
and packaging are cancelled. This also demonstrates why a 32-sequence pilot
is only a rejection screen: even a three-seed +0.011 result can be a
nonrepresentative subset effect.

## Round 14: predictive contrastive pretraining (2026-10-04)

`contrastive_pretrain.py` pretrained the same GRU96 encoder to identify future
normalized market states at horizons 1, 8 and 32 from its causal state, using
in-batch InfoNCE negatives. Future rows are train-only self-supervision labels;
the deployed target predictor remains strictly causal. The FIT-only
normalizer, 128-FIT/32-DEV split, target schedule and initialization are
matched to the control. Full/chunk recurrent states agree in the self-check.

The contrastive loss fell 2.3585 → 1.7838 → 1.7109. Target epoch 1 was
0.288941 versus 0.282089 for the control (+0.006853), but the advantage did
not persist. The candidate peaked at epoch 3 with 0.313787, below the
control's 0.316064 by 0.002277. This repeats the next-event-pretraining result:
self-supervision accelerates early fitting but does not improve the converged
representation. It fails the +0.005 pilot gate and is not scaled up.

## Round 15: dedicated future-direction model (2026-10-04)

`difference_screen.py` trains a separate GRU96 for `D = t0 − t1`, weighted
by `|t0| + |t1|`. At inference it preserves a joint model's
`S = −(t0 + t1)` and replaces half of its D prediction. The frozen pilot
recipe is epoch 5 / D weight 0.5. Seed gains were +0.005525, +0.009135 and
+0.003060 (mean +0.005907); every seed was positive, although seed 9 missed
the individual +0.005 pilot gate.

Unlike the multi-scale pilot, the effect survived the larger group-CV test.
`difference_cv.py` used 1,024 fitting and 512 untouched sequences per fold:

| Fold | Normalized GRU96 | 50% dedicated D | Gain |
|---:|---:|---:|---:|
| 1 | 0.405980 | 0.410002 | +0.004022 |
| 2 | 0.427730 | 0.430244 | +0.002515 |
| 3 | 0.412565 | 0.415069 | +0.002504 |

Both targets improved in every fold. This establishes a real weakness in a
single joint GRU: its loss does not learn the shared future-direction factor
as well as a dedicated objective.

The deployment gate nevertheless failed. On fold 1, replacing 50% of D in
the weaker three-GRU ensemble changed 0.413768 to 0.414304 (+0.000536), while
doing so in the strongest fixed ensemble (three GRUs + split + wide) changed
0.418084 to 0.416483 (−0.001601). The strong ensemble already contains the
dedicated model's useful D variation. Since v25 is also a strong
distilled/split ensemble, full-FIT training, validation-mask fine-tuning and a
v27 package are cancelled rather than spending another leaderboard trial on
a sub-gate effect.

## Official v25/v26 (2026-10-04)

| Package | Members | Official | Runtime | Validation proxy |
|---|---|---:|---:|---:|
| v18 | SD71–74, split (0.2 each) | 0.65870 | 51m01s | 0.684634 (v19-like) |
| **v25** | C75, C76 0.4 each; split 0.2 | **0.65898** | **39m21s** | 0.685181 |
| v26 | C75, C76, GRU192, split (0.25 each) | 0.6589 | 54m37s | 0.685207 |

v25 is the new best (+0.00028 official vs +0.00056 proxy, 95% bootstrap
[+0.00007, +0.00104]). v26 adds nothing (proxy +0.00003, interval
[−0.00027, +0.00033]) and runs within six minutes of the cap. Runtime: two
groups at 290 kMAC took 39m21s; the third group (GRU192) in v26/v17 costs
about 15 minutes for 175 kMAC, more than the per-kMAC rate, so members should
stay in as few width groups as possible. v24 (four continued students +
split, two groups, 474 kMAC) has v18's shape (about 51 minutes) and the same
proxy as v25 with half the per-student seed variance.

Public leaderboard on 2026-10-04: rank 104 (0.65898); rank 100 0.65900,
rank 50 0.66177, rank 10 0.66554, rank 5 0.66731, rank 1 0.67109 (rank 1 was
0.66772 two days earlier).

## Round 9: target structure (loop toward prize level, 2026-10-04)

Prize line: top 8 on the private leaderboard (public rank 8 ≈ 0.666, +0.007
over v25). The model-side levers are exhausted, so this round studies the
target itself.

**Price events carry the quote move.** At i0 price-block changes, a linear
map of the block's change vector explains the change of the targets with
held-out R² 0.64 (t0) / 0.60 (t1) in the 0.657 group (0.36–0.47 in others).

**F is a persistent future price.** One 0.657 train sequence flickers through
five book states every 1–2 rows; the targets cycle with them exactly as
t0 = F − ask, t1 = bid − F with F fixed (states (A−2, A−1) → (0, −1),
(A−2, A) → (−1, −1), (A−1, A) → (−1, 0)). A transition table learned from
labels ((from-state, to-state) → modal (Δask, Δbid), purity 0.926) decodes
absolute quote tracks: F = ask + t0 then changes at only 1–15% of events
(e.g. 23 of 11,620). The target is mostly "fixed F minus the current quotes".

**The price block is a shape code, not a level.** In decoded held-out
sequences 120 of 145 states occur at several absolute ask levels (ranges of
tens of ticks), so absolute price movement is visible only by integrating
event changes. F matches a future quote at event lags 1–10 about 58–60% of
the time (≈ mid within half a tick 74–78%), decaying slowly beyond: no sharp
horizon, partly because decoding errors accumulate along the track.

Test under way: decoded absolute-track features (moves of the decoded mid
over 10–5000 rows, distance from trailing means, decoded spread), using a
train-only transition table, against the v24-like ensemble residual.

**Decoded i0 absolute-track features: negative.** Train-only transition table
(216,049 entries over all tick groups, purity 0.883; 1.6% of validation
transitions unseen). Features: decoded spread, mid moves over 10/50/200/1000/5000
rows, distance of the mid from 50–5000-row trailing means. Residual
correction of the v24-like set: h0 → h1 0.684340 → 0.683624 / 0.682075 / 0.676557
at shrink 0.25 / 0.5 / 1; h1 → h0 0.686021 → 0.685431 / 0.683720 / 0.677131.
The GRUs already capture (or cannot use) i0's integrated absolute movement.

**Decoded i1 absolute-track features (lead-lag): negative.** i1's book uses
the same representation, so i1 transitions were decoded with the i0-trained
table (82% of i1 transitions found). Adding i1 decoded spread, moves over
10–5000 rows and trailing-mean distances to the i0 features: h0 → h1
0.684340 → 0.683963 / 0.682661 / 0.677354 (shrink 0.25 / 0.5 / 1); h1 → h0
0.686021 → 0.685648 / 0.684099 / 0.677654.

**Round 9 conclusion.** The target structure is now clearer (t0 = F − ask,
t1 = bid − F with a persistent future price F; the price block is a shape
code), but every feature family derived from it (event change vectors,
integrated absolute i0 movement, integrated i1 movement) adds nothing to the
ensemble on held-out validation. Together with rounds 7–8 (width, data,
distillation generations, EMA, continuation saturation, residual tests,
mask structure, feature shift) and the parallel session's closed families,
no evidence-based path to prize level (about +0.007 public) remains.
Best official: v25 0.65898; v24 (same proxy, four students) is unsubmitted.

## Round 10: candidates from public evidence (2026-10-04)

A model that matched the first Wunder challenge's public #1 was an ensemble of
2-layer GRUs (256/384) trained with full-sequence BPTT and a scored-step loss.
Candidates: (1) 2-layer GRU128 student, (2) longer BPTT (2048-row chunks, batch
8 so rows per update and updates per epoch match the 512 × 32 baseline),
(3) a diagonal linear recurrent (LRU/SSM) model, (4) bid/ask mirror symmetry
for augmentation or test-time averaging, (5) K-fold validation ensembles and
large tuning (leaderboard-only). (1) and (2) run with seed 71, gen-2 teacher,
5 epochs, compared at e5 with G71 (h0 0.684673, h1 0.683096); only winners get
the low-LR continuation. `pilot.make_gru` and `ensdistill.make_student` take a
layer count; `ensdistill.py` takes `--layers`, `--chunk`, `--batch`.

**(4) Bid/ask mirror: rejected.** Marginals suggest a price-column pairing with
sign flip (p0↔p16, p1↔p11, p2↔p15, …; the same permutation for i0 and i1),
but after z-scoring it barely separates from other pairings (median quantile
distance 0.31 vs 0.52 std) and volume marginals are uninformative. Functional
test with the continued seed-71 student on 60 held-out sequences (original
0.6755): all 16 input mirrors (price sign, trade-price sign, a0–a4 sign,
volume swap) under all four output maps (swap or not, negate or not) score
between −0.56 and +0.57; none preserves the signal. No usable simple mirror
exists in the obfuscated encoding, so no augmentation or test-time averaging.

**(3) minGRU stack: running.** `mingru.py`: residual pre-norm stack of minimal
gated linear recurrences (gates read only the layer input, h_t = (1 − z_t)
h_{t−1} + z_t tanh(c_t)), trained with a blocked log-space scan (≈1e-4
relative rounding; 128-step blocks; self-check against the sequential
recurrence and chunked stateful inference). Width 128 × 3 layers, 164k
parameters, about 162 kMAC per row at inference. `ensdistill.py --cell mingru
--layers 3`; `distill_mps.build` loads it. Same seed/teacher/schedule as G71.

**Official v24 (2026-10-04): 0.65914, 53m02s** — new best (v25 0.65898, v18
0.65870), rank 98; rank 8 is 0.66582. Same runtime shape as v18 (51m01s), so
runtime varies by about two minutes between runs; about seven minutes of margin.

## Round 16: event-state and public-evidence follow-up (2026-10-04)

All screens below use causal inference and the locked 128-FIT/32-DEV pilot
unless a validation-half score is stated. The promotion gate remains +0.005
on the pilot or +0.002 on both validation halves.

- A GRU observation plus fitted event-delta state filter failed: the raw
  GRU96 scored 0.316064, while the best filtered setting scored 0.293547.
  Causal smoothing within price intervals also failed; the best setting was
  0.308313. Although targets are exactly unchanged on 89.82% of adjacent
  rows, within-interval book updates improve the raw GRU prediction.
- A 2,048-row BPTT candidate matched the control's rows and optimizer updates
  per epoch but scored 0.313811 (−0.002254). A two-layer GRU128 checkpoint
  scored 0.681017 on validation half 0 and reduced the strong proxy even at
  5% weight. A three-layer minGRU128 scored 0.683720; its best 20% blend moved
  the proxy only 0.686021 → 0.686098 (+0.000076). These architecture branches
  are closed without training the missing opposite-half runs.
- Oracle tick quantization was negative even when the true sequence tick was
  supplied. Direct rounding lowered the two halves; spread-grid variants
  disagreed between halves and supplied no robust gain.
- A shared-trunk `D = t0 - t1` auxiliary head peaked at 0.316627 (+0.000563)
  and fell to 0.313628 at the frozen epoch-5/50% mix. It does not retain the
  separate D model's gain.
- A cross-fitted future-event linear oracle used only as a training teacher
  accelerated epoch 1 to 0.288548 versus 0.282089, but constant 50%
  distillation converged to 0.315364. Switching to label-only training after
  epoch 1 ended at 0.316151 (+0.000087). No future value enters inference.
- The exact previous interval target is a 0.979703 causal-persistence oracle,
  but it cannot be recovered accurately enough. A linear reconstruction from
  the completed interval endpoint and newly observed event scored 0.511123
  on event rows and only 0.205363 when carried into the next interval. A
  nonlinear MLP raised event reconstruction to 0.536508, below its 0.56 gate.
  Feeding a sequence-disjoint, cross-fitted delayed estimate into a zero-
  initialized GRU input produced the only converged positive result:
  0.317544 versus 0.316064 (+0.001480). It is real enough to retain in the
  research map but too small and too repeatedly selected to justify scaling.

No Round-16 candidate meets the preregistered promotion gate. No v27 package
is built. The best verified official result remains v24 at 0.65914 (v25
0.65898); neither is evidence of prize-range performance.

**(3) minGRU stack: no large effect.** Seed 71, h0, epochs 1–5: 0.67741 /
0.678507 / 0.679007 / 0.68372 / 0.684597 (G71 e5 0.684673); SWA(e3–e5)
0.684287 vs G71 0.684436. Its predictions correlate 0.993 with G71, the same
as two GRU seeds (G71–G72 0.993), so the different cell adds no diversity:
G71 + minGRU 0.685268 vs G71 + G72 0.685115; adding it to the v24-like set
0.686021 → 0.686081 (+0.00006). Stopped before the h1 run finished; a 3-layer
linear-recurrent stack behaves like one more GRU seed.

## Status and direction (2026-10-04)

Official path: v1 0.6487 → v10 0.6582 → v18 0.6587 → v25 0.6590 → **v24 0.6591**
(rank 98). Prize line (private top 8; public rank 8 now 0.6658) is +0.0067 away.

What holds: every model family we can train converges to the same function
(students, minGRU stacks and GRU seeds correlate 0.993 with each other;
ensemble ≈ teacher ≈ 0.685 held-out), and no input-derived signal remains
(residual tests on current-row, integrated i0/i1 absolute tracks). Small
validation gains reach the public board at about 0.5–1×, with ±0.0002–0.0003
noise. Remaining screens: 2-layer GRU (running), long BPTT (queued).

Direction: keep v24 (best, lowest variance) and v25 (faster) as final
candidates; finish the two screens and package only a winner that holds in a
set on both halves; treat prize level as out of reach without a new insight.

**Test-time (per-sequence) calibration bound: rejected.** Oracle per-sequence
affine maps of the v24-like set fitted with TRUE labels: in-sample (all rows of
the sequence, i.e. using its future) 0.686 → 0.728 / 0.684 → 0.736, which only
shows how much pooled WP depends on each sequence's realized drift; online
(only scored rows at least 500 / 2000 / 5000 rows older, shrunk to identity)
0.642 / 0.638 / 0.641 (h0) and 0.645 / 0.637 / 0.639 (h1), far below the
0.686 / 0.684 base. Even perfect past labels cannot calibrate a sequence
online: its drift does not persist. Test-time adaptation is not a lever.

**F is not the next trade price.** 0.657 group, 80 train sequences, 46,314
trades: (1) just before a trade F sits at a current quote 55.1% of the time vs
59.5% over all rows; (2) at 191,182 trade rows without a price-block change
the targets never change (0.0000), whereas under the hypothesis they would
change whenever the following trade differs; (3) trade-row features give the
side of F before the trade with held-out AUC 0.805 only. F is tied to price
events, not trades.

## Round 11: pooled Global-WP objective and lineage diversity (2026-10-05)

The parallel session pointed out that the earlier Pearson test used per-chunk
(512-row) correlations during a 3-epoch fine-tune at one weight, while the
metric pools all selected rows of all sequences. `maskft.py --gwp L
[--gwp-mse W]` now adds the pooled term exactly: a no-grad pass over the
batch's whole sequences gives the pooled statistics, and each truncated-BPTT
chunk back-propagates the exact per-row derivative of the pooled, clipped,
|y|-weighted correlation (scaled by the chunk count). `pooled_wp_gradient`
matches `GlobalWP` and autograd (2e-9). Tests: v1 fine-tunes with (L, W) =
(1, 1) and (3, 0.2) against the MSE baseline (0.681944 / 0.679740).

Public evidence behind the parallel session's plan, checked at the source:
the first challenge's strong model (2-layer GRU, full 1,000-row sequences,
55–65 epochs) trained on 517 × 1,000 rows, about 30M row passes in total,
whereas one of our students already sees about 1.1B; its "full sequence" is
comparable to our 512–2,048-row chunks, and the report has no depth or epoch
ablations. The Predictorium top-1% repository credits direct Pearson training
without numbers, on a metric that appears to be per-sequence (where MSE wastes
capacity on per-sequence offsets that the pooled metric rewards).

**(1) 2-layer GRU128 student: small gain.** Seed 71, gen-2 teacher, 5 epochs:
e5 h0 0.68612 / h1 0.683326 vs G71 0.684673 / 0.683096 (+0.0014 / +0.0002).
SWA(e3–e5): 0.685245 / 0.683799 (mean 0.684522) vs G71 0.684436 / 0.683962
(0.684199), +0.0003. It correlates 0.993 with G71 (same teacher lineage). In
sets: replacing G71 by it in (·, C72 0.4; split 0.2) 0.685051 → 0.685156;
adding it as a third continued-student-sized member 0.685181 → 0.685526, mostly
the effect of one more member. Twice the inference cost (about 190 kMAC);
not packaged.

**Lineage diversity check.** The label-only GRU256 (no distillation; round 6)
fine-tuned on each half: e1–e3 h0 0.676791 / 0.676967 / 0.678295, h1
0.675801 / 0.673402 / 0.671695; SWA(e2, e3) 0.678371 / 0.674740. It correlates
0.978 with C71, against 0.993 between distilled students, so a different
lineage does give a less correlated model, but it is too weak to help:
adding it to the v24-like set at 5 / 10 / 20% gives 0.685226 / 0.685222 /
0.685063 (set 0.685181). A useful diverse member would need student-level
accuracy at that lower correlation.

**Pooled Global-WP objective: no effect.** v1 fine-tunes, SWA(e2, e3):
MSE 0.681944 / 0.679740 (0.680842); MSE + pooled WP (1, 1) 0.681906 / 0.679894
(0.680900, +0.00006); WP-centric (3, 0.2) 0.681323 / 0.679833 (0.680578,
−0.00026). The WP-centric model correlates 0.997–0.998 with the MSE model.
With the metric-exact pooled gradient the result matches the earlier chunk
test: |y|-weighted MSE already targets the pooled metric's optimum, so the
objective is not a lever for teachers either.

**Label-only student (alpha 0): diverse but weak.** Same recipe, seed, init and
data order as G71, teacher term off. Epochs 1–5: h0 0.667316 / 0.670444 /
0.668947 / 0.673153 / 0.677032, h1 0.666204 / 0.670936 / 0.669371 / 0.675555 /
0.674278; SWA(e3–e5) 0.677281 / 0.676605 (mean 0.676943) vs G71 0.684199. It
correlates 0.980–0.984 with G71 (students 0.993), yet adding it to the
v24-like set gives 0.685205 / 0.685183 / 0.685000 at 5 / 10 / 20% (set
0.685181). Without the teacher a model is about 0.007 weaker; with it, it
becomes the teacher's function. Lineage diversity is not a usable lever.

**Round 11 conclusion.** The pooled Global-WP objective (metric-exact),
depth, minGRU stacks and label-only lineages all leave the ensemble within
±0.0003. The parallel session's main path (large Global-WP teachers with
diverse objectives, distilled into students) has no supporting evidence left:
the objective does not change the learned function (0.998 correlation), and
diversity comes only with a 0.007 accuracy loss. Long BPTT (2,048 rows) is
still running on h0 (e1 0.678842 vs G71 0.678744).

**(2) Long BPTT (2,048-row chunks, batch 8): no effect.** Same rows per update
and updates per epoch as the 512 × 32 baseline; 65–80 minutes per epoch on MPS.
e1–e5 h0 0.678842 / 0.680424 / 0.680817 / 0.683979 / 0.685127, h1 0.67756 /
0.682376 / 0.682078 / 0.682408 / 0.68333 (G71 e5 0.684673 / 0.683096). SWA(e3–e5)
0.684384 / 0.683602 (mean 0.683993) vs G71 0.684199 (−0.0002). It correlates
0.994 with G71. In sets: replacing G71 0.685051 → 0.684966; as a third member
0.685181 → 0.685364 (the extra-member effect). A longer gradient horizon does
not change the learned function either. All five candidates are now closed.

## Complete warm-up context follow-up (2026-10-05)

The first scored row is 99, so all rows 0--98 can be summarized without
look-ahead. `warmup_context_screen.py` forms per-feature mean, standard
deviation and mean absolute movement over those 99 rows, maps the 336 values
to 16 dimensions, and appends the fixed context to each GRU input. Causality
and full/chunk recurrence checks pass.

On the locked 128/32 pilot it ended at 0.322988 versus 0.316064 (+0.006924).
In the three 1,024-train/512-test FIT folds, epoch 4 gains over GRU96 were
+0.000381 / +0.004433 / +0.003122 (mean +0.002645). Adding the epoch-4 model
at a fixed 20% to the stronger three-GRU + split + wide ensemble raised the
folds by +0.001137 / +0.002096 / +0.002321 (mean +0.001851); 25% averaged
+0.002021. These are promising exploratory numbers, but epoch 4 was selected
after inspecting all epoch histories: epoch 5 averaged only +0.000680 and was
negative on fold 3. The rerun verifies determinism, not independent selection.

The higher-priority validation-half checks failed:

- A zero-initialized context adapter on the C71 SWA, with every existing
  recurrent/head parameter frozen, reproduced the C71/v24-like baseline
  exactly at epoch 0. After three epochs on half 0, replacing C71 lowered the
  set by 0.000037; adding the adapter at 10% changed it by only +0.000008.
  Half 1 was cancelled by the +0.0005 gate.
- A jointly trained width-128 warm-up student used the exact G71 gen-2
  teacher, 6,000-sequence pool, alpha 0.75, seed 71 and schedule. On half 0 it
  scored 0.679093 at epoch 1 versus G71 0.678744 (+0.000349), then 0.681023
  at epoch 2 versus 0.681168 (−0.000145). The remaining epochs and half 1
  were cancelled.
- Mapping the same summary only to the initial GRU hidden state peaked at
  0.318727 in epoch 3 but ended at 0.314223 versus 0.316064 (−0.001841) in
  the locked pilot.

A cross-fitted recurrent residual corrector was also screened separately.
On 128/128 sequence halves its best post-selected gains were +0.000265 and
+0.000418, far below the +0.005 pilot gate, so it was not scaled.

Conclusion: the complete warm-up contains a small diversity signal for weak
FIT-only ensembles, but it does not survive the v24 lineage's mask-aware
distillation. No package is built and v24 remains the official best.

## Round 12: leaderboard forensics (2026-10-05)

Public leaderboard: 31 of the top 60 posted their best on Oct 4–5. Several
accounts registered on Oct 4 reached the prize line within a day and 5–7
submissions (iaeste rank 9, 0.66658, 5 submissions; ryan_cooper rank 14,
0.66629, 5; arseniykan rank 15, 0.66611, 7). A public repository for this
challenge, `dodler/wunder_connectome`, was indexed by search engines with
training scripts, ONNX models and checkpoints, and now returns 404. This fits
a complete solution having been public for a short time. The rules forbid
sharing complete solutions and require prize winners to hand over their own
training code and pass a code review, so that material is not used here, and
no copy was sought. Only general ideas from public discussion are allowed.

**Strong optimizer recipe for a label-only model: running.** The first
challenge's strong public model used AdamW (weight decay 1e-4), a OneCycle
cosine schedule with 10% warm-up, gradient clipping at 1.0 and dropout between
layers. `ensdistill.py` now takes `--adamw WD --onecycle --clip C --dropout P`
(`pilot.make_gru` takes `dropout`). Test: label-only GRU128 (alpha 0, full
train pool, 12 epochs, lr 1.5e-3, seed 71) on both halves, against the
label-only GRU128 trained with the current recipe (SWA 0.676943). A
label-only model at student level would be a strong, less correlated teacher
member; if the recipe gets there, a 2-layer GRU256 (about 30 minutes per epoch
on MPS) follows.

**Strong optimizer recipe (label-only GRU128): no gain.** AdamW 1e-4, OneCycle
(10% warm-up, peak lr 1.5e-3), clip 1.0, full train pool, 12 epochs, seed 71.
Held-out by epoch, h0: 0.662282 / 0.668244 / 0.670579 / 0.670466 / 0.67254 /
0.672634 / 0.677241 / 0.675874 / 0.677393 / 0.678322 / 0.677483 (e1–e11); h1:
0.665052 / 0.669727 / 0.668994 / 0.671001 / 0.669721 / 0.671956 / 0.673601 /
0.675002 / 0.676141 / 0.676574 / 0.675542. It plateaus at about 0.677, the
level of the label-only model trained with the current recipe (SWA 0.676943),
far from the 0.682 needed for a strong second lineage. The optimizer recipe
is not what separates label-only training from distillation.

**Loop conclusion (2026-10-05).** Every legitimate lever found or suggested
has been tested on both validation halves: width, depth, cell type and SSM
stacks, BPTT length, epochs, continuation, EMA, optimizer recipe, data size,
pooled-Pearson objective, distillation generations, lineage diversity,
current-row and integrated absolute-price inputs, i1 lead-lag, target
structure, mask structure, test-time calibration, mirror symmetry and feature
shift. None moves the ensemble beyond about ±0.0004 held-out. The public
prize-line surge on Oct 4–5 coincides with a briefly public complete solution
that the rules do not allow us to use. v24 (0.65914) remains the best
official result; v24 and v25 are the final candidates.

**Residual test with history features: negative.** 368 features per scored row:
10/100/1000/5000-row changes of a0–a7 and i0/i1 trade columns, 100/1000-row
changes of all 104 book columns, 100/1000-row rolling volatility of a0–a7 and
trades, i0/i1 price-event and trade rates over 20/200/2000 rows, and log time
since the last i0/i1 price event and trade. Cross-fit LightGBM on the v24-like
residual: h0 → h1 0.684340 → 0.684538 / 0.684548 / 0.683800 / 0.679492 (shrink
0.1 / 0.25 / 0.5 / 1); h1 → h0 0.686021 → 0.686077 / 0.685855 / 0.684680 /
0.679411. The splits use long-window activity and volatility (2,000-row i0
event rate, 1,000-row volatility of a2/a3/a6/a7): sequence-level regime
scale, which does not transfer. History features are exhausted as well.
Organizer text found by search ("preserve context across long event streams",
"adapt to noisy and non-stationary dynamics", "interconnected instruments")
describes the task generally and points to no specific method.
(Final epoch of the label-only strong-recipe runs: e12 h0 0.677636, h1 0.675128.)

**Distilled students with the strong recipe: running.** (a) 2-layer GRU192,
dropout 0.1, alpha 0.75, AdamW 1e-4, OneCycle (peak 1.5e-3), clip 1.0, 8
epochs, seed 71, h0 on MPS: does regularization make a larger distilled model
worth its inference cost (about 397 kMAC per row)? (b) GRU128 with the same
recipe on CPU, h0 then h1: a drop-in replacement for the current students,
compared with C71 (8 epochs including the continuation, SWA 0.684607).

**Strong-recipe distilled GRU128 (h0): first gain of the round.** AdamW 1e-4,
OneCycle (peak 1.5e-3, 10% warm-up), clip 1.0, alpha 0.75, 8 epochs, seed 71.
h0 by epoch: 0.677981 / 0.680315 / 0.679915 / 0.681377 / 0.684312 / 0.685039 /
0.685521 / 0.685808 (still rising). Against C71 (same seed, Adam 5 epochs +
3-epoch low-LR continuation, SWA 0.685150): final epoch 0.685808 (+0.00066),
SWA(e6–e8) 0.685587 (+0.00044) — with OneCycle the annealed final weights beat
SWA. Correlation with C71 0.994, so the gain is accuracy, not diversity. Pairs:
S8 + C72 0.686247 vs C71 + C72 0.685884 (+0.00036); v24-like set with S8 in
place of C71 0.686311 vs 0.686021 (+0.00029). h1 is training (CPU); the
2-layer GRU192 version is at e4 (0.681593, level with the GRU128 run).

**Strong-recipe GRU128, h1: negative.** h1 by epoch 0.675304 / 0.680037 /
0.677472 / 0.679566 / 0.680882 / 0.681717 / 0.682147 / 0.682127 against C71
0.684065 (−0.0019), after +0.00066 on h0: the recipe is not a reliable gain
for GRU128 students. **2-layer GRU192, strong recipe, h0**: 0.678799 /
0.6797 / 0.678991 / 0.681593 / 0.684567 / 0.685624 / 0.686256 / 0.686391 —
best single model so far on h0 (C71 0.685150, +0.0012; strong GRU128
+0.0006). Its h1 run (CPU) decides whether size pays off under
regularization. (Two seed-72 strong-recipe runs were launched twice by a
queueing slip; the later duplicates and the h1 run were stopped, the 22:39 h0
run continues; its log was truncated by the duplicate.)

**Seed-72 strong-recipe GRU128, h0:** epochs 1–8 0.677697 / 0.679618 / 0.681278 /
0.681464 / 0.683383 / 0.684884 / 0.68502 / 0.685361 (C72 0.684934, +0.0004).
h0 sets: S71 + S72 0.686505 vs C71 + C72 0.685884 (+0.00062); v24-like set
with strong students 0.686522 vs 0.686021 (+0.0005); 2-layer GRU192 alone
0.686391; GRU192 + S72 0.686946 (+0.0011 over C71 + C72); GRU192 correlates
0.992 with C71. Everything here is h0; seed-71 strong GRU128 lost 0.0019 on
h1, and several earlier changes also looked better on h0 than h1. Waiting for
seed-72 strong h1 (MPS) and the 2-layer GRU192 h1 (CPU).

**h1 results and verdict: neither the strong recipe nor the larger model
helps.** h1 by epoch — seed-72 strong GRU128: 0.678876 / 0.678644 / 0.680592 /
0.677982 / 0.681316 / 0.681972 / 0.682481 / 0.682396 (C72 0.681876, +0.0005);
2-layer GRU192: 0.672078 / 0.68038 / 0.677729 / 0.678156 / 0.680309 / 0.682201 /
0.682797 / 0.682674 (C71 0.684065, −0.0014; same-recipe GRU128 0.682127,
+0.0005). Both halves, final epochs:

| Set | h0 | h1 | Mean |
|---|---:|---:|---:|
| C71 + C72 (current recipe) | 0.685884 | 0.683896 | 0.684890 |
| S71 + S72 (strong recipe) | 0.686505 | 0.683251 | 0.684878 |
| v24-like: C71, C72 0.4; split 0.2 | 0.686021 | 0.684340 | 0.685181 |
| S71, S72 0.4; split 0.2 | 0.686522 | 0.683839 | 0.685180 |
| C71, C72, S71, S72 0.2; split 0.2 | 0.686528 | 0.684365 | 0.685446 |
| 2-layer GRU192 alone | 0.686391 | 0.682674 | 0.684532 |
| GRU192 0.3, C71, C72 0.25; split 0.2 | 0.686747 | 0.684598 | 0.685672 |

The strong recipe gains on h0 and loses on h1 (seed-level noise); the
2-layer GRU192 equals C71 on average (0.684532 vs 0.684607) and adds only
what an extra member adds, at four times the inference cost. The h0/h1
asymmetry recurs for several changes, so single-half results must not drive
decisions. Search for the top teams' method: no robust lever found.

## Round 13: following the untried-methods report (2026-10-06)

Report: `reports/커넥톰 대회 미시도 기법과 튜닝.md` (order: insurance checks → WiSE-FT →
queue-information kill test → teacher fine-tune regularizers → student bundle).

**Step 2, queue information: closed at identification.** On rows where only
one side's volumes change (no price event, no trade), all 11 volume columns of
that side change (median 11; never ≤ 2), bid and ask blocks change
independently, and no column dominates (largest |Δv| share ≈ 0.24). Change
vectors align with a few fixed directions only 13–38% of the time
(|cos| > 0.995). FastICA (11 components, 200k change vectors per side) does
not sparsify them: the median energy share of the largest coordinate stays
0.45–0.55 (raw 0.45–0.53), with under 1% of rows dominated by one component.
There is no linear invariant (smallest/largest covariance eigenvalue
0.03–0.05) and no constant per-row norm (CV 0.30–0.49). The volume encoding is
a nonlinear, non-sparse transform, so per-level queue sizes and best-queue
imbalance cannot be recovered; together with the earlier residual tests on the
raw volume columns, the queue-information bet is closed.

**Step 0a, fp16 weights: safe.** Rounding every weight of the per-half
v24-like set (C71, C72, split encoder) to float16 leaves the h0 set at
0.686021 (identical to six decimals; largest prediction change 1.1e-3), so a
half-precision kernel would be an accuracy-neutral way to halve weight
traffic if runtime ever binds. (The SVD check is re-running; MPS lacks
float64.)

**Step 3, teacher fine-tune regularizers: screened and closed.** `maskft.py` now takes
`--dropconnect P` (recurrent weight drop, one mask per chunk, gradient routed
through the mask), `--sam RHO` (two-pass sharpness-aware step on the same
input state), `--beta2 B` and `--input-noise S`. Grid on GRU128 s61 (paired
with its existing fine-tune, seed 7): DropConnect 0.15 / 0.3, SAM 0.05, β2
0.98, input noise 0.05, plus seeds 8 and 9 for a fine-tune soup; both halves.
The best regularizer was β2=0.98 on h0 (0.681926) and h1 (0.680777), but
neither half reached the matched control's +0.002 gate; SAM was clearly worse
(0.680481 / 0.680601). No regularized teacher is promoted to a package.

**Step 4, student bundle: queued.** `ensdistill.py` now takes `--chrono T`
(update-gate biases log U[1, T−1]), `--random-offset` (random first
truncated-BPTT boundary per batch) and `--alpha-train A` (teacher weight on
train rows; validation rows keep `--alpha`). Bundle: chrono 20000, random
offset, alpha 1.0 on train rows and 0.6 on validation rows, otherwise the G71
recipe; seeds 71/72 × both halves, judged at e5 against G71/G72.

**Step 0b, SVD of the students' stacked input matrix: costs accuracy.** h0
set 0.686021 → rank 32/48/64/80: 0.683897 / 0.684958 / 0.685498 / 0.685817.
fp16 weights are the safe route to runtime headroom; low rank is not.

### AFML skill applied (`afml-financial-ml`)

**Selection overfitting (CSCV, `selection_pbo.py`).** 27 configurations
evaluated on both halves (17 single models, 10 sets), held-out predictions
covering all 1,873 validation sequences, 16 sequence blocks, 12,870 splits:
PBO 0.000, median logit 3.30; in-sample winners are "big mix" (2-layer GRU192
0.3 + C71, C72 0.25 + split 0.2; 11,572 splits), "4 students" (C71, C72, S71,
S72, split 0.2 each; 996) and v26-like (294); the IS winner's OOS shortfall vs
the OOS best averages −0.00007. Full-validation ranking: big mix 0.685781,
4 students 0.685557, v26-like 0.685293, v24-like 0.685287, C pair 0.685006.
Within the validation period, choosing by validation is not luck, and more
members rank higher — consistent with v24 (four students) beating v25 (two)
publicly. This says nothing about shift to the test period.

**Completed:** clustered permutation importance for the continued student pair
(`cluster_mda.py`: 14 clusters swapped across sequences, both halves). The
largest mean drops were i0 bid/ask volumes (+0.05163/+0.05100) and i0 bid
price (+0.04863); i1 price/volume groups were only +0.001--0.004. This is
evidence that the first instrument carries most of the signal, but it is not
an ablation result: correlated clusters can substitute for one another, so
the result is used for representation prioritization, not feature deletion.

**Uniqueness weighting: closed after the group-CV check.** The competition does
not expose explicit label start/end spans, so exact AFML average uniqueness is
not identifiable. We therefore tested inverse target-persistence run length as
a train-only proxy (`uniqueness_weight_screen.py`), with no weighting at
inference. On the locked pilot, power 0.5 and 1.0 looked unusually strong:
e5 gains were +0.01428 and +0.01064 over the matched control. The full
sequence-disjoint 1,024-train/512-test check removed that effect for power
0.5:

| Fold | Control | Fixed e5 | Gain |
|---:|---:|---:|---:|
| 1 | 0.405980 | 0.406515 | +0.000535 |
| 2 | 0.427730 | 0.421405 | −0.006325 |
| 3 | 0.412565 | 0.411959 | −0.000607 |
| **mean** | — | — | **−0.002132** |

Only one of three folds was positive; sample standard deviation of the gains
was 0.003676. Power 1.0 was already −0.011917 on fold 1. The pilot gains are
therefore selection/optimization noise or a non-transferable reweighting, not
a prize-range lever; no submission or full refit is justified.

**Clustered permutation importance (`cluster_mda.py`): no harmful cluster.**
Continued student pair, each cluster swapped with another sequence's values,
WP drop h0 / h1: i0 bid price +0.0575 / +0.0398, i0 ask price +0.0336 / +0.0335,
i0 bid volume +0.0567 / +0.0465, i0 ask volume +0.0529 / +0.0491, i0 trade price
+0.0039 / +0.0029, i0 trade volume +0.0092 / +0.0095, i1 bid price +0.0041 / +0.0044,
i1 ask price +0.0023 / +0.0023, i1 bid volume +0.0015 / +0.0014, i1 ask volume
+0.0018 / +0.0005, i1 trade price +0.0025 / +0.0012, i1 trade volume +0.0078 /
+0.0074, a0–a4 +0.0063 / +0.0054, a5–a7 +0.0071 / +0.0052. Every cluster helps;
the obfuscated i0 volumes matter as much as i0 prices, so the GRUs already use
the volume information despite its nonlinear encoding. Nothing to drop.

**Step 1, WiSE-FT (h0).** Split encoder, θ = base + a (fine-tuned − base):
a = 0 / 0.6 / 0.8 / 0.9 / 1 / 1.1 → alone 0.675469 / 0.682164 / 0.682910 /
0.683024 / 0.682971 / 0.682758; in the v24-like set 0.685369 / 0.685846 /
0.685949 / 0.685989 / 0.686021 / 0.686045; DEV (train period, 266 sequences)
0.428875 / 0.430455 / 0.428823 / 0.427599 / 0.426097 / 0.424267. GRU192 teacher
member: alone 0.676030 / 0.681334 / 0.681991 / 0.682104 / 0.682069 / 0.681882;
DEV 0.430585 / 0.429750 / 0.427519 / 0.426000 / 0.424197 / 0.422104. a ≈ 0.9
keeps validation-period accuracy (±0.00005) while recovering 0.0015–0.0018 on
the train-period proxy: a free robustness hedge, not a score lever.
WiSE-FT h1 (same a grid): split alone 0.675077 / 0.681209 / 0.681974 / 0.682124 /
0.682128 / 0.681990, set 0.683666 / 0.684137 / 0.684249 / 0.684297 / 0.684340 /
0.684379, DEV 0.428875 / 0.430427 / 0.429033 / 0.427941 / 0.426558 / 0.424888;
GRU192 alone 0.676696 / 0.680532 / 0.680670 / 0.680500 / 0.680168 / 0.679676,
DEV 0.430585 / 0.432167 / 0.430777 / 0.429625 / 0.428164 / 0.426369. Both halves
agree: a = 0.9 is free in-period (≤ 0.00005) and +0.0014 on the train-period
proxy; GRU192 at a = 0.8 is −0.0001 / +0.0005 in-period and +0.0026 DEV. Use it
for the private-set hedge (split encoder at a = 0.9), not as a score lever.

**Step 3, teacher fine-tune regularizers (GRU128 s61, SWA(e2, e3), paired with
seed 7): no lever.** DropConnect 0.15 +0.00001 / +0.00027; DropConnect 0.3
−0.00041 / +0.00017; SAM 0.05 −0.00118 / +0.00015; β2 0.98 +0.00021 / +0.00017;
input noise 0.05 −0.00003 / +0.00001; 3-seed fine-tune soup −0.00001 / −0.00027
(output average −0.00003 / −0.00024). Data order alone moves a fine-tune by
−0.0013 / −0.0024 (seed 8) and −0.0016 / −0.0005 (seed 9) against seed 7, so
seed 7 was a good draw and single fine-tunes carry ±0.002 of order noise; the
soup recovers the average of the seeds, not more. Nothing reaches the
+0.0005 re-distillation bar.

**Uniqueness weighting (AFML ch. 4) in fine-tuning: no lever.** Row weight ×
(i0 price-block length)^−P, SWA(e2, e3): P = 0.5 0.681798 / 0.681979 (mean
0.681888, +0.0001); P = 1 0.679149 / 0.679955 (−0.0022). Equal weight per block
fights the row-counting metric; the compromise is noise-level.

**AFML summary.** PBO 0.000 (validation selection is reliable), no harmful
feature cluster, WiSE a = 0.9 a free private-set hedge, uniqueness weighting
and fine-tune regularizers noise-level. The one measured score lever is
ensemble size: 4-member and larger sets rank highest (big mix 0.685781 vs
v24-like 0.685287 on all validation), which needs runtime headroom (fp16
weights are accuracy-neutral).

**Step 4, student bundle: no lever.** Chrono init (T = 20,000), random first
truncated-BPTT boundary, teacher weight 1.0 on train rows and 0.6 on
validation rows, otherwise the G71/G72 recipe. e5 held-out vs the same seeds:
s71 h0 0.683651 vs 0.684673 (−0.0010), s72 h0 0.684195 vs 0.683589 (+0.0006),
s71 h1 0.683034 vs 0.683096 (−0.0001), s72 h1 0.682522 vs 0.681621 (+0.0009);
mean +0.0001 with 2 of 4 positive (bar: +0.001 and 3 of 4). Not continued or
packaged.

**Round 13 conclusion.** Every step of the untried-methods report has been
run: fp16 weights safe, SVD costly, WiSE a = 0.9 a free hedge, queue
information unrecoverable, fine-tune regularizers and the student bundle
noise-level, AFML diagnostics (PBO 0.000, no harmful feature cluster,
uniqueness weighting) without a score lever. The only measured lever left is a
larger ensemble, which needs runtime headroom (an fp16 kernel); expected about
+0.0005 held-out.

## Round 14: final-selection insurance and a float16 kernel (2026-10-07)

**Runtime guard (multi_solution.py).** The final dataset is re-scored on a
separate hidden set, so a 51–53 minute package has little margin (v18 and v24
differ by two minutes with identical structure). 54 minutes after loading, the
callback keeps only the first plain member (its hidden state carries over
exactly) with its own unweighted head (`fallback_out`, written by the
packager); after 58.5 minutes it returns zeros without running the network.
threadpoolctl (if installed) limits BLAS to one thread, since numpy is loaded
by the harness before the environment variables are set. The packager checks a
mid-sequence switch against the lone member (8×10⁻⁷) and the zero stage.

**WiSE hedge, v27.** `split128_..._tb_all_swa23_wise0.9.pt` = base + 0.9 ·
(SWA(e2, e3) − base); `submission_v27.zip` = v24 with this split member
(NumPy path with the guard; PyTorch parity 5.7×10⁻⁷). Expected public score
equal to v24 (in-period −0.00004), +0.0014 for the split member on the
train-period proxy.

**float16 kernel (fast_kernel.c).** The whole row runs in one C call: weights
in float16 panels of 32 rows (F16C conversion + FMA on x86, `__fp16` on
AArch64), float32 state and arithmetic, Cephes-style exp for the gates. The
callback compiles it at load time (`gcc -O3 -march=native`; build-essential is
in the scoring image), runs a self-test in a child process (an illegal
instruction cannot kill the run), and falls back to the NumPy path otherwise.
Checks: AVX2 path under Rosetta equals the AArch64 path to 4.8×10⁻⁷; kernel vs
PyTorch 7×10⁻⁴; scored WP on 13 validation sequences 0.767649 (kernel) vs
0.767647 (float32). This Mac: 0.376 s vs 0.686 s per sequence for the v24
members. `submission_v28.zip` = v24's members on the kernel, the calibration
submission: its score should equal v24 and its runtime gives the official
speed-up.

**Two-layer members.** Layer 0 joins its width group with a zero head; layer 1
is a "deep" group reading that member's [h, 1] (NumPy path and kernel).
Packaging the h0 two-layer GRU192 in either order matches PyTorch (NumPy
7.6×10⁻⁷, kernel 9.4×10⁻⁴).

**Which members the headroom is for.** Deployable additions to v24-like
(C .8, split .2 = 0.685181, both halves): SD students −0.00005 to −0.00015,
GRU192 teacher member +0.0001 at 0.2, two GRU192s +0.00009, g128c −0.00009;
none is worth runtime. The measured gains need new all-validation members:
2-layer GRU192 (big mix +0.0005) and strong-recipe GRU128 students (C .4,
S .4 +0.00027). Training: 2-layer GRU192 all (seed 71, MPS, about 6 hours) and
strong GRU128 all (seeds 71, 72, CPU, about 3 hours each).

**Official v28 (2026-10-07): 0.6591 in 25m07s** (v24: 0.65914 in 53m02s, the
same members on the NumPy path). The float16 kernel compiles and runs on the
scoring machine, keeps the score and takes 2.1× less time; v28 replaces v24
as the safe final candidate. One kernel point cannot separate fixed overhead
from per-kMAC cost: 25.1 min = F + 482·c, so 1,066 kMAC would take about 37
(F = 15) to 50 minutes (F = 5, or the extra weights spill from L2 to L3).

**v29, preview of the larger ensemble at the planned compute (1,066 kMAC).**
v24's four students 0.1 each; strong-recipe GRU128 s71 (h1) and s72 (h0) 0.1
each, so the two together saw all validation; 2-layer GRU192 (h0) 0.2; split
encoder 0.2. Held-out analogue (C .4, S .2, B192 .2, split .2): +0.00053 over
v24-like, positive on both halves (h0 +0.0008, h1 +0.0003). PyTorch parity
4.3×10⁻⁷ (NumPy) and 6.6×10⁻⁴ (kernel), 3,954,255 bytes. Its runtime
calibrates the kernel cost model; its score previews the gain before the
all-validation members finish.

**All-validation members finished.** 2-layer GRU192 (seed 71) final training
loss 0.3214 (h0 run 0.3165, h1 run 0.3259); strong GRU128 seeds 71 and 72
0.3231 and 0.3232. Health check on 8 validation sequences: each correlates
0.99 with its h0/h1 versions on scored rows (0.97 on all rows) with the same
prediction spread. **v30** = v29 with these members and the WiSE split
encoder: C75–C78 0.1 each, strong s71 and s72 0.1 each, 2-layer GRU192 0.2,
WiSE split 0.2 (1,066 kMAC, 3,955,006 bytes; PyTorch parity 5.8×10⁻⁷ NumPy,
5.9×10⁻⁴ kernel).

A label-only member is not the missing diversity: earlier tests (round 11)
added label-only models to the v24-like set at 5 / 10 / 20% for +0.00004 /
0.0000 / −0.0002 despite their 0.98 correlation.

**Out-of-period check (`dev_check.py`).** DEV minus the 6,000-sequence student
pool and the split base's FIT groups leaves 451 train-period sequences that no
member has seen; v28's and v30's sets are scored there as a second,
independent check for the final choice.
Result (Global WP, all rows after warm-up): members C75–C78 0.4337–0.4360,
strong s71 / s72 0.4362 / 0.4344, 2-layer GRU192 0.4358, split 0.4318, WiSE
split 0.4333 (+0.0016, as in step 1). Sets: v28 0.438161, v28 with WiSE split
0.438332, v30 with plain split 0.439046, **v30 0.439216 (+0.00106)**. Paired
sequence bootstrap (2,000 resamples) of v30 − v28: +0.00106, 95% interval
[+0.00036, +0.00171], P(≤ 0) 0.0015; plain-split v30 − v28 +0.00089
[+0.00021, +0.00151]. v30 wins on both validation halves (held-out analogue
+0.00053) and on this independent train-period check, and WiSE adds about
+0.00017 out of period at no in-period cost: v30 is the preferred final
candidate, with v28 as the fast, safe second choice.

**Official v29 (2026-10-07): 0.6593 in 45m37s** (+0.0002 over v28's 0.6591,
in line with the expected +0.0002 to +0.0004). Two kernel points give the
kernel cost model: official minutes ≈ 8.2 + 35.1 × (kMAC / 1000) (NumPy path:
about 19 + 59–70). v30 has v29's compute (about 45.6 minutes); one more
GRU128 student would take it to about 49 minutes and a second 2-layer GRU192
to about 60, so v30 is the largest ensemble with a margin for the final
re-run (a final set 20% larger reaches about 54.7 minutes, where the guard
switches to one member).

**Official v30 (2026-10-07): 0.6594 in 45m32s** (0.659412; predicted 45.6
minutes). New best: +0.0001 over v29 and +0.0003 over v24/v28, matching the
two local checks in direction. Leaderboard the same day: rank 121 of 4,436;
rank 1 0.67109, rank 8 (last prize) 0.66744 (0.66582 on Oct 4), rank 10
0.66723, rank 50 0.66246, rank 100 0.66009. The prize line is 0.0080 above
v30 and still rising. Final selection plan: v30 (best on validation halves,
unseen train-period sequences and the public board) and v28 (v24's members
in 25 minutes, the safe choice).

## Round 15: representation and structure screens (2026-10-07)

A proposed priority list (queue/OFI restoration, two-instrument GRU, causal
normalization and domain adaptation, PLE feature calibration, Mogrifier,
multi-horizon auxiliary heads, distillation variants) was checked against this
log first:
- Queue/OFI restoration (best-queue pairing, queue imbalance, microprice) is
  closed at identification (Round 13, step 2): a one-side queue change moves
  all 11 volume columns, ICA does not sparsify them, and no linear invariant
  exists; the price-volume mirror test also failed (Round 12).
- Next-event auxiliary losses and event pretraining failed (Rounds 8 and 10);
  a target-split dual GRU lost 0.0018 in the pilot.
- Mogrifier and step-coupled cross-gated two-state GRUs need a per-step
  Python recurrence (no fused kernel); at 20,000 rows per sequence this is
  not trainable at our scale. The two-instrument idea is screened in its fused
  form instead.

Screens, all as distilled students with G71's exact recipe and seed (gen-2
teacher, alpha 0.75, 6,000 train sequences per epoch, lr 1e-3, 5 epochs), h0
on MPS and h1 on CPU, against G71 h0 0.678744 / 0.681168 / 0.681218 /
0.684181 / 0.684673 and h1 0.678729 / 0.681329 / 0.679707 / 0.683866 /
0.683096 (epoch-to-epoch noise about ±0.002, so only a gain on both halves at
e5 counts):
1. `--cell ple`: per-feature piecewise-linear calibration of the standardized
   inputs (7 knots at per-feature quantiles 2–98%, slopes start at zero, so
   the GRU starts exactly as G71's).
2. `--causal 500`: every input's causal EWMA z-score (half-life 500 rows)
   appended to the inputs (224 columns).
3. `--cell twin`: separate i0 and i1 GRUs (96 and 48 units, each also reading
   a0–a7), then a joint GRU96 (about 131 kMAC).

**1. PLE calibration: no gain.** Identical start to G71 (epoch 0 equal on both
halves). h0 0.679487 / 0.681331 / 0.68109 / 0.684058 / 0.684692 (e5 +0.00002);
h1 0.678553 / 0.68123 / 0.680899 / 0.682719 / 0.682775 (e5 −0.00032). Learned
per-feature nonlinearities add nothing the GRU's gates do not already do.

**2. Causal EWMA z-scores (half-life 500): worse.** h0 0.678455 / 0.680587 /
0.679471 / 0.68256 / 0.683854 (e5 −0.00082); h1 0.675747 / 0.679875 /
0.678651 / 0.682317 / 0.682065 (e5 −0.00103); final training loss 0.3259 /
0.3361 against PLE's 0.3240 / 0.3339 on the same data. Sequence-relative
inputs do not help the period shift; the GRU's own state already carries the
recent context.

**3. Two-instrument GRU (`--cell twin`, i0 GRU96 + i1 GRU48, joint GRU96,
about 131 kMAC): an ensemble member, not a better single model.** h0 0.680484 /
0.68244 / 0.681643 / 0.685724 / 0.686185 (e5 +0.00151 over G71); h1 0.677756 /
0.682499 / 0.679311 / 0.68337 / 0.68316 (e5 +0.00006) — the recurring h0-only
gain. Held-out predictions (`pred_twin_e5_h*.npz`): alone h0 0.686185 / h1
0.683160 (C71 0.685150 / 0.684065); correlation with C71 0.9926 / 0.9918
(C71–C72 0.9936 / 0.9930), with the 2-layer GRU192 0.9917 / 0.9908. Sets:

| Set | h0 | h1 | Mean vs v24-like |
|---|---:|---:|---:|
| C .6, twin .2, split .2 | 0.686477 | 0.684648 | +0.00038 |
| v30-like (C .4, S .2, B192 .2, split .2) | 0.686827 | 0.684597 | +0.00053 |
| v30-like + twin (C .3, S .15, B192 .15, twin .2, split .2) | 0.687020 | 0.684772 | +0.00072 |
| twin in place of S (C .4, twin .2, B192 .2, split .2) | 0.686941 | 0.684798 | +0.00069 |

Replacing the two strong students by the twin gains +0.00011 / +0.00020 over
v30-like on both halves and costs 131 instead of 186 kMAC (about 43.7 instead
of 45.6 official minutes). All-validation twin (5 epochs, then G→C-style
continuation 1e-4, 1e-4, 3e-5) is training on MPS; the continuation is
checked on h0/h1 on CPU.

**Twin export.** `package_ft.twin_layers` rewrites a twin as an equivalent
two-layer GRU: layer 0 = the i0 and i1 GRUs side by side (block-diagonal,
width 160 with 16 zero units that stay at zero), layer 1 = the joint GRU96
reading [h, 1]; the existing two-layer paths (NumPy and kernel, including
different layer widths) then run it unchanged. Test package (C75, twin h0,
split): PyTorch TwinGRU parity 5.2×10⁻⁷ (NumPy) and 6.2×10⁻⁴ (kernel).
Two packager fixes came with it: layer 0's zero head takes layer 0's width,
and a twin's joint `rnn` no longer sets the plain input width.

**Twin continuation (1e-4, 1e-4, 3e-5 from e5): +0.0006 to +0.0008 on both
halves.** h0 0.686185 → 0.686662 / 0.686948 / 0.686816; h1 0.683160 →
0.683989 / 0.6838 / 0.683678 (the G→C student pattern). SWA over the three
continuation epochs ("twinC"): alone h0 0.686965 / h1 0.683943, correlation
with C71 0.9934 / 0.9927. Sets: **v31-like (C .4, twinC .2, B192 .2, split .2)
0.687037 / 0.684868, mean 0.685952 (+0.00077 over v24-like, +0.00024 over
v30-like, positive on both halves)**; with the uncontinued twin +0.00069; twin
at 0.3 (C .3) +0.00088, not adopted (weights chosen on the same halves, one
twin seed). The all-validation twin is at epoch 4 of 5 (MPS was slowed 1.5–2×
while the CPU checks ran); its continuation follows.

**v31 and v32 (all-validation continued twin).** All-validation twin seed 71:
e5 training loss 0.3282, continuation 0.3244 / 0.3252 / 0.3228 (between the
h0 and h1 runs). `submission_v31.zip` = C75–C78 0.1 each, twin SWA(cont e1–e3)
0.2, 2-layer GRU192 0.2, WiSE split 0.2 (parity 5.9×10⁻⁷ / 5.6×10⁻⁴,
3,753,206 bytes). In the kernel the twin is a width-160 block-diagonal layer
plus a GRU96 (about 206 kMAC with the padding), so v31 is about 1,089 kMAC,
about 46.4 official minutes.

Unseen train-period check (`dev_check.py`, 451 sequences): **the twin alone
scores 0.439717, above every other member (0.4318–0.4362) and above the whole
v30 ensemble (0.439216)**; v31 0.439961 (+0.00074 over v30, bootstrap 95%
[+0.00026, +0.00121], P(≤ 0) 0.002; +0.0018 over v28). Three separately
trained twins (h0, h1, all) agree, so the structure, not one lucky model,
carries the gain; it also generalizes best out of period.

Twin weight: validation halves (C .6 − t, twin t, B192 .2, split .2) t = 0.2 /
0.3 / 0.4 → mean 0.685952 / 0.686057 / 0.686106 (h1 flat from 0.3); DEV
0.439961 / 0.440314 / 0.440555 (0.3 − 0.2: +0.00035, 95% [+0.00016,
+0.00055]). 0.3 is where both checks still agree: **`submission_v32.zip`** =
C75–C78 0.075 each, twin 0.3, 2-layer GRU192 0.2, WiSE split 0.2 (parity
5.8×10⁻⁷ / 5.2×10⁻⁴, 3,753,236 bytes, about 46.4 minutes). Twin seed 72 is
training (all-validation on MPS; h0 and h1 on CPU) for a two-seed twin share.

**Official v32 (2026-10-08): 0.6597 in 46m11s** (0.659661; predicted 46.4
minutes). +0.0003 over v30 and +0.0006 over v24/v28: the twin structure, not
more of the same members, gave the largest step since v24, matching both
local checks. Leaderboard: rank 124 of 792 scored; rank 1 0.67125, rank 8
0.66767 (+0.0080), rank 50 0.66246, rank 100 0.66040. Final pair now v32 and
v28.

**Twin seed 72.** h0: e5 0.686212 (s71 0.686185), continuation e2 / e3
0.687449 / 0.68735 (s71 0.686948 / 0.686816). All-validation: e5 loss 0.3285,
continuation 0.3251 / 0.3241 / 0.3229 (s71 0.3228). (The laptop slept on
battery with the lid closed from 13:46 to 17:52 on Oct 8, pausing both runs;
`caffeinate -s` only holds on AC power.) Unseen train-period check: twin s72
alone 0.438290 (s71 0.439717; non-twin members ≤ 0.4362). Sets: v32 0.440314;
two twins at 0.15 each with GRU192 0.440246 (−0.00007); two twins at 0.2 each
with GRU192 and students 0.05 each 0.440536 (+0.00022); two twins at 0.2 each
without GRU192, students 0.1 each 0.440189 (−0.00012). Seed 71 was the better
draw out of period; splitting the twin share over two seeds costs nothing
there and halves the dependence on one model. Runtime constraint: four
students + two twins + GRU192 + split is about 1,296 kMAC (53.7 minutes, too
long); two students + two twins + GRU192 + split about 1,109 kMAC (47.1);
four students + two twins + split about 897 kMAC (39.7). Waiting for h1.

**Twin seed 72, both halves.** Continuation h0 0.68731 / 0.687449 / 0.68735,
h1 0.683535 / 0.683056 / 0.683181; SWA alone h0 0.687470 / h1 0.683341 (s71
0.686965 / 0.683943); correlation between the two twins 0.9947 / 0.9943
(higher than twin–student 0.993: a second seed adds robustness, little
diversity). Sets against v32-like (C .3, t71 .3, B192 .2, split .2: 0.687191 /
0.684923):

| Set | h0 | h1 | Unseen DEV vs v32 |
|---|---:|---:|---:|
| two twins 0.15 each, B192 .2, C .3 | +0.00014 | −0.00003 | −0.00007 |
| two twins 0.2 each, B192 .2, C .2 | +0.00031 | −0.00002 | +0.00022 |
| two twins 0.2 each, no B192, C .4 | −0.00007 | −0.00008 | −0.00012 |

None beats v32 on both halves (the rule set before looking); the second row
ties on h1 and wins on h0 and out of period. Built as an optional candidate:
`submission_v33.zip` = C75, C76 0.1 each, twins s71 and s72 0.2 each, 2-layer
GRU192 0.2, WiSE split 0.2 (about 1,109 kMAC, 47.1 minutes; parity
5.5×10⁻⁷ / 5.3×10⁻⁴, 3,553,051 bytes). v32 stays the primary final candidate.

**Official v33 (2026-10-09): 0.6598 in 46m37s** (predicted 47.1 minutes):
+0.0001 over v32, within noise but in line with the local checks (h0 +0.00031,
DEV +0.00022, h1 tie) and less dependent on one twin. Final candidates: v33
(primary), v28 (fast, twin-free hedge), v32.

**4. Wider twin (`--scratch 128 --cell twin`: i0 GRU128 + i1 GRU64, joint
GRU128; 221 kMAC native, about 300 in the padded kernel form): running.** Same
recipe and seed as twin96 (5 epochs at 1e-3, then continuation 1e-4, 1e-4,
3e-5), h0 on MPS and h1 on CPU. Question: does a wider twin beat twin96 on
both halves, enough to replace the 2-layer GRU192 (399 kMAC)?
**Result: worse; stopped.** h1 (complete): e1–e5 0.678122 / 0.680583 /
0.681213 / 0.682218 / 0.681984, continuation 0.682019 / 0.681645 / 0.681526
(twin96: e5 0.68316, continuation 0.683989 / 0.6838 / 0.683678), so about −0.002
with lower training loss (overfitting). h0: e5 0.686051 (twin96 0.686185),
continuation e1 0.686514 (twin96 0.686662). On h1, twin128 SWA alone 0.681879
(twin96 0.683943), correlation with twin96 0.9939; in the v33-like set
(0.684907) replacing the 2-layer GRU192 gives 0.684617 and adding it at 0.1
gives 0.684800. It fails h1 outright, so the last two h0 continuation epochs
were stopped. Twin capacity is not the lever; twin96 stays.
(The laptop slept on battery 09:40–14:23 on Oct 9, stretching epoch 1 to
about 4 hours.)

## Round 16: which metric does the leaderboard use? (2026-10-10)

`global_wp.py` (written 2026-09-17 from the website's description) pools all
selected rows of all sequences. The starter pack's own reference (METRIC.md,
spec `v2_wp_block_clip2_v1`; `utils.BlockAccumulator`) instead takes the
equal mean of per-sequence two-target weighted Pearson over eligible
sequences. Every local number in this log is pooled.

Held-out sets re-scored with the per-sequence metric keep the same order with
slightly larger gaps (mean of halves, pooled → per-sequence): C71 0.68461 →
0.64174, B192 0.68453 → 0.64221, split 0.68255 → 0.63977, twin s71 0.68545 →
0.64398; v12-like 0.68418 → 0.64194, v24-like 0.68518 → 0.64307, v30-like
0.68571 → 0.64376, v32-like 0.68606 → 0.64441, v33-like 0.68620 → 0.64458
(v24 → v33: +0.0010 pooled, +0.0015 per-sequence). Past selections stand under
either metric. Frozen v1 on all validation: pooled 0.675613, per-sequence
0.629293.

Offsets to the public board (LB − local): pooled baseline −0.024, v1 −0.027,
v12 −0.026, v24 −0.026, v33 −0.026; per-sequence +0.004 (README baseline
0.589595), +0.019, +0.016, +0.016, +0.015. The public board tracks the pooled
metric much more tightly, contradicting the starter-pack documentation.

Decisive test: `cache/packages/submission_diag_metric.zip` = v28 with each
sequence's predictions multiplied by 0.25 + 0.75·frac(1000·Σ|first row|)
(from data, not `seq_ix`). Locally (v24-like halves, random scales in [0.25, 1])
this leaves the per-sequence metric unchanged (0.64184 → 0.64184, 0.64349 →
0.64348) and drops the pooled metric by 0.044 (0.68602 → 0.64164, 0.68434 →
0.64055). Expected public score: about 0.659 if per-sequence, about 0.615 if
pooled. Checked: outputs are exactly v28's times one constant per sequence.

**Result: the public board is pooled.** `submission_diag_metric.zip` scored
**0.6158** in 26m47s (v28 0.6591): −0.0433, matching the local pooled drop
(−0.044) and not the per-sequence prediction (unchanged, about 0.659). The
public leaderboard uses global pooled weighted Pearson over all selected rows,
like `global_wp.py`, not the starter pack's per-sequence `v2_wp_block_clip2_v1`
spec. Every local comparison in this log was therefore on the leaderboard's
metric. A per-sequence objective would optimize the wrong target for the public
board. Whether the final private scoring follows the same scorer is not stated;
v33 is the best set under both metrics, so the final choice does not depend on it.

**5. Triple-stream GRU (`--cell triple`): running.** Separate recurrent states
for i0 prices (book prices 0–21 and trade prices 44–47), i0 volumes (book
volumes 22–43 and trade volumes 48–51) and i1 (52–103), each also reading
a0–a7, widths 64 / 64 / 48, then a joint GRU96 (about 133 kMAC native, 260 in
the padded kernel form). Same recipe and seed as twin96, h0 on MPS and h1 on
CPU, with the continuation. Gate: beat twin96 in the v33-like set on both
halves and not lose on the unseen train-period check. Export:
`package_ft.stream_layers` generalizes the twin conversion to any streams (in
the module's joint-input order); a random triple and the twin both match
PyTorch (NumPy 4.3×10⁻⁷ / 5.8×10⁻⁷, kernel 4.9×10⁻⁴ / 7.7×10⁻⁴).
**Result: no gain; stopped.** h1 (complete): e1–e5 0.677783 / 0.682565 /
0.680486 / 0.682545 / 0.682411, continuation 0.683235 / 0.683094 / 0.682965
(twin s71 0.683989 / 0.6838 / 0.683678; twin s72 0.683535 / 0.683056 /
0.683181). h0 through e4: 0.682898 (e2) / 0.682247 / 0.686024 (twin s71
0.68244 / 0.681643 / 0.685724). On h1 the triple correlates 0.9942 with twin
s71, the same as the two twin seeds with each other (0.9943): it is another
twin, not a new view. In the v33-like set on h1 (0.684907): replacing twin
s72 −0.00002, replacing the 2-layer GRU192 −0.00006, replacing both students
−0.00009, adding it at 0.15 −0.00001. Unseen train-period check with the h1
models: triple 0.439494, twin s71 0.439718, twin s72 0.439844; twin s71 +
twin s72 0.441462 vs twin s71 + triple 0.441243. It fails in period and out
of period, so the remaining h0 epochs were stopped. Splitting i0 into price
and volume streams adds nothing beyond the instrument split.

## Round 17: more twin seeds (2026-10-11)

Twin seed averaging measured on the two seeds: one twin (mean of seeds) h0
0.687217 / h1 0.683642 → average of the two 0.687899 / 0.684379 (+0.00068 /
+0.00074, both halves; +0.0017 on the unseen train-period sequences). A
1/k seed-noise extrapolation gives about +0.0010–0.0011 for four seeds and
+0.0012–0.0013 for eight, over one. Runtime with the padded twin (206 kMAC):
four twins + split 41.1 minutes, five 48.3; the plan needs a native layout.
Training all-validation twins seeds 73 (MPS) and 74 (CPU), same recipe.

**Native multi-stream layout.** Each stream is its own first-layer member
(zero head, input weights only on its columns, width padded to a multiple of
32 with zero units) and the joint layer is a deep member with several sources
(`g{j}_source` is now a list of (group, member) pairs; its input is the
sources' [h, 1] in order, the input bias riding on the last constant). NumPy
path: the deep member concatenates its sources' states. Kernel: a deep
member's sources run right before it, so their states are adjacent and it
reads one contiguous slice; plain members using under 80% of the input
columns gather only those (new kind 3, `columns` in the context). Twin: about
144 kMAC instead of 206 (i0 stream 45.5k, i1 stream padded to 64 units 24.2k,
joint 74.6k). Checks: twin, triple (random weights) and the 2-layer GRU192
match PyTorch (NumPy 4.8×10⁻⁷ / 4.8×10⁻⁷ / 6.0×10⁻⁷; kernel 6.2×10⁻⁴ /
5.6×10⁻⁴ / 7.2×10⁻⁴); the AVX2 path under Rosetta equals the AArch64 path
to 3.6×10⁻⁷ on the twin package. Planned v34: four twins + 2-layer GRU192 +
split ≈ 1,089 kMAC ≈ 46.4 minutes.
Seed 74 (CPU): e5 loss 0.3281, continuation 0.3252 / 0.3256 / 0.3230; unseen
train-period check alone 0.439072 (s71 0.439717, s72 0.438290). Seed 73 on MPS
runs about 3,550 s per epoch (CPU about 2,600 s for this model). Pre-set v34
candidate: four twins 0.15 each, 2-layer GRU192 0.2, WiSE split 0.2 (weights
0.5 and 0.7 for the twins shown for reference only).
Seed 73 (MPS): continuation 0.3252 / 0.3252 / 0.3226; alone 0.440233 on the
unseen train-period sequences (best twin). Sets there: v33 0.440521; **v34
(four twins 0.15 each, GRU192 0.2, WiSE split 0.2) 0.441381, +0.00086 over
v33, bootstrap 95% [+0.00037, +0.00131], no resample ≤ 0**; twins 0.5 total
0.441009, 0.7 total 0.441661 (not chosen: weights were fixed beforehand).
In-period evidence is the two-seed result (no students: h0 +0.0002, h1
−0.0001 with two twins) plus the measured seed-averaging gain.
`submission_v34.zip` (3,840,153 bytes): PyTorch parity 5.8×10⁻⁷ (NumPy) and
5.6×10⁻⁴ (kernel); kernel layout four × [i0 stream (gather 61), i1 stream
(gather 61), joint (162)], GRU192 (plain + deep), split; 1,089 kMAC, about
46.4 official minutes. No single-layer member comes first, so the guard has
only its zero stage (58.5 minutes); v33 keeps the full guard as the
alternative.

**Official v34 (2026-10-12): 0.6599 in 51m31s** (+0.0001 over v33). The
runtime model said 46.4 minutes: v32 had the same 1,089 kMAC in nine kernel
members and took 46m11s, v34 runs fifteen smaller members (four twins as
three members each), so per-member cost (about 0.9 official minutes per extra
member, or slower gathered and small members) is no longer negligible; the
kMAC-only model does not apply to many small members. Margin: 8.5 minutes to
the limit, and the guard has only its zero stage at 58.5 minutes, so a final
set more than about 13% larger than the public test would partly score zeros.

**v35: four twins + split, without the 2-layer GRU192.** Unseen train-period
check: four twins 0.2 each + WiSE split 0.2 0.441625 (v34 0.441381, v33
0.440521); validation halves with the two-seed analogue (twins 0.8, split
0.2) against v34-like: h0 −0.00013, h1 −0.00015. The GRU192 is roughly
neutral (slightly helpful in period, slightly harmful out of period) while
costing about 400 kMAC, two kernel members and about 15 official minutes.
`submission_v35.zip` = twins s71–s74 0.2 each, WiSE split 0.2 (2,358,160
bytes; parity 4.5×10⁻⁷ NumPy, 5.9×10⁻⁴ kernel; about 690 kMAC in 13 kernel
members, expected about 33–37 official minutes). Purpose: v34's accuracy
with a large runtime margin for the final re-run.

## Round 18: developing the twin structure (2026-10-12)

Two variants, each screened on h1 (5 epochs + continuation, seed 71) in
period and on the 451 unseen train-period sequences with the h1 models,
against twin96 seeds 71/72 under the same conditions:
1. Narrow twin (`--scratch 64 --cell twin`: 64 / 32 / 64, 64k parameters,
   about 65 kMAC native): if close to twin96, eight seeds would fit in the
   runtime of four.
2. Siamese twin (`--cell siamese`): one GRU64 shared by i0 and i1 (two states,
   same weights), joint GRU96; 89k parameters, about 114 kMAC. Export: the
   shared weights become both streams (`package_ft.streams` recognizes
   `shared.*`); a random siamese model matches PyTorch (NumPy 4.3×10⁻⁷, kernel
   7.1×10⁻⁴).

**1. Narrow twin: rejected.** h1 e1–e5 0.676981 / 0.680936 / 0.679678 /
0.682706 / 0.682428, continuation 0.683497 / 0.683317 / 0.683171 (CPU 1,430 s
per epoch). SWA alone 0.683466 (twin96 s71 0.683943, s72 0.683341),
correlation with s71 0.9943 (as s71–s72). In period it averages like a twin96
seed (s71 + narrow 0.684426 vs s71 + s72 0.684379; three twins + split
0.684762 vs two 0.684670). Out of period it loses the twin's advantage: alone
0.437453 vs 0.439718 / 0.439844; s71 + narrow 0.440203 vs s71 + s72 0.441462;
all three 0.441209. Twin96 sits between overfitting (twin128, in period) and
losing out-of-period robustness (twin64).

**2. Siamese twin: rejected.** h1 e1–e5 0.677038 / 0.681725 / 0.679375 /
0.682465 / 0.682645, continuation 0.683079 / 0.683019 / 0.682923. SWA alone
0.683135 (twin96 s71 0.683943, s72 0.683341), correlation with s71 0.9941. In
period: s71 + siamese 0.684288 vs s71 + s72 0.684379; as a third twin in the
v35-like set 0.684686 vs 0.684670. Out of period: alone 0.438801 (twin96
0.439718 / 0.439844), s71 + siamese 0.440925 vs s71 + s72 0.441462, all three
0.441697. Shared instrument weights give a slightly weaker twin, not a new
view. Twin96 (separate weights, 96 / 48 / 96) remains the structure; seeds
are the only lever left in this family.

**Official v35 (2026-10-12): 0.6598 in 35m51s** (estimate 33–37 minutes):
within 0.0001 of v34 (0.6599) and 15m40s faster, best on the unseen
train-period check (0.441625). Final pair by the rule set before
submission: **v35 + v33** (v33: 0.6598, 46m37s, full guard and a different
composition).

**v35 composition: no change justified.** Variants (unseen DEV with four
twins; h0 / h1 with the two-twin analogue), against v35 (0.441625 / 0.687603
/ 0.684670): student first (twins 0.7, C 0.1, split 0.2) −0.00024 / −0.00006
/ +0.00021; no split (twins 1.0) +0.00032 / +0.00030 / −0.00029; split 0.1
+0.00025 / +0.00019 / −0.00010; strong student (twins 0.7, S 0.1, split 0.2)
−0.00006 / +0.00002 / +0.00002. Each trades h0 and DEV against h1.

**Runtime with per-member cost.** Fitting the kernel packages (v29–v35):
minutes ≈ 0.6 + 0.0347 · kMAC + 0.88 · kernel members; a twin is about 7.6
minutes (144 kMAC, three members). Five twins + split ≈ 43.5 minutes; six
twins + split ≈ 51; six twins without split ≈ 46.4. Twin seeds 75 (MPS) and
76 (CPU) are training for a five- or six-twin v36.
Seed 76 (CPU): e5 loss 0.3287, continuation 0.3260 / 0.3243 / 0.3227; alone on
the unseen train-period sequences 0.440845 (best twin so far; s71 0.439717,
s72 0.438290, s73 0.440233, s74 0.439072). Candidates fixed before seed 75
finishes: v36a five twins (0.16 each) + split 0.2 (about 43.5 minutes), v36b
six twins 1/6 each without split (about 46.4), and six twins + split for
reference (about 51, too slow).
Seed 75 (MPS): continuation 0.3246 / 0.3241 / 0.3229; alone 0.438667 on the
unseen train-period sequences. Sets there: v35 0.441625; v36a (five twins +
split) 0.441643 (+0.00002); **v36b (six twins, no split) 0.442383
(+0.00076)**; six twins + split 0.441926 (+0.0003, about 51 minutes). The
split member's role is disputed: dropping it was +0.0003 (h0) / −0.0003 (h1)
with two twins, +0.0005 out of period with six. `submission_v36.zip` = twins
s71–s76, 1/6 each, no split (2,950,573 bytes; parity 4.6×10⁻⁷ NumPy,
3.3×10⁻⁴ kernel; 18 kernel members, 866 kMAC, about 46.5 minutes; guard zero
stage only). A leaderboard check on top of two out-of-three local checks.

**Official v36 (2026-10-13): 0.6603 in 43m32s** (estimate 46.5): +0.0004 over
v34, +0.0005 over v35. The unseen train-period check predicted it (+0.00076);
the halves were split. Across v30–v36 the leaderboard follows that check's
direction (about 1/3–2/3 of its size) more closely than the in-period halves:
the public test behaves like out-of-period data, which is where the twins
win. Runtime refit with v36: minutes ≈ 0.6 + 0.0376 · kMAC + 0.3 · members
(twin about 6.3 minutes, GRU192 about 15.6, split about 4.5); a seventh twin
would be about 50 minutes. Final pair now **v36 + v33** (v33: different
members and the full guard).

## Round 19: more twins (2026-10-13)

**Tail panels.** `gemv` now takes any row count that is a multiple of 8: full
32-row panels, then one panel of the remaining 8, 16 or 24 rows (AVX2 path
with up to three accumulators). `panels()` pads rows to a multiple of 8, the
packager pads stream widths to a multiple of 8, the kernel accepts widths
that are multiples of 8, and the split encoder runs its 72 rows instead of 96.
The twin's i1 stream runs at its true 48 units: about 131 kMAC per twin
instead of 144 (joint input 146 instead of 162). Checks: six twins match
PyTorch exactly as before (NumPy 4.6×10⁻⁷, kernel 3.3×10⁻⁴); student +
GRU192 + split 9.4×10⁻⁷ / 6.1×10⁻⁴; AVX2 under Rosetta equals AArch64 to
3.6×10⁻⁷ on the six-twin package.

Runtime refit on v34–v36: minutes ≈ 7.2 + 0.0382 · kMAC + 0.18 · kernel
members (v34 51.5, v35 35.9, v36 43.5 reproduced). With tail panels a twin is
about 5.6 minutes: six twins about 40.5, seven about 46, eight about 52.

**i1 inference ablation (three all-data twins on the unseen train-period
sequences; twin s71 half models in period).** i1 volumes zeroed: DEV −0.0022,
h0 −0.0003, h1 −0.0010; swapped between sequences: −0.0044 / −0.0021 /
−0.0020; all i1 zeroed: −0.024 / −0.024 / −0.022; a0–a7 swapped: −0.027 /
−0.018 / −0.014. The most period-shifted inputs (i1 volumes) are used more
out of period, not less: the twin's inputs stay as they are. More twins:
seeds 77 (MPS) and 78 (CPU) training; v37 = seven twins (s71–s77, by seed
order, not chosen by score) with tail panels, about 46 minutes.
Seeds 77 (MPS) and 78 (CPU): continuation final loss 0.3231 / 0.3237; alone on
the unseen train-period sequences 0.438962 / 0.439224. Sets there: v36 (six
twins) 0.442383, **seven twins 0.442388 (+0.000005), eight twins 0.442421
(+0.00004)**. Seed averaging has saturated at six twins (pairwise correlation
about 0.994: the shared error dominates), so v37 is not built. With tail
panels v36's predictions are unchanged (removed units carried zero weights)
and its runtime would drop from 43.5 to about 40.5 minutes; v36 already has
16 minutes of margin, so it is not resubmitted. Final pair: v36 + v33.

## Round 20: toward a different error (2026-10-13)

The six-twin average has saturated, so the remaining question is whether the
models' shared error can be reduced. Candidates, in order: (1) teacher
dependence, (2) a column-interaction encoder, (3) row plus event memory, (4)
conditional representation alignment (CORAL). The halves and the unseen
train-period sequences have been used for many selections, so decision rules
are fixed before each result.

**1. Teacher dependence (running).** Twins learn 0.25·label + 0.75·teacher.
From the trained twin s71 (h0/h1 e5), the three-epoch continuation (1e-4,
1e-4, 3e-5, same seed and data order) is repeated with teacher weight 0.25
and 0, against the existing 0.75 continuation (h0 0.686662 / 0.686948 /
0.686816; h1 0.683989 / 0.6838 / 0.683678). Rule: a lower teacher weight
counts only if it beats 0.75 on both halves (SWA of the three epochs) and is
not worse on the unseen sequences (half models).

**2. Column-interaction encoder (`--cell twinenc`, queued after 1).** Each of
the eight order-book groups (i0/i1 × bid/ask × price/volume levels, 11
columns) gets an MLP 11 → 32 → 16 (tanh); each twin stream reads its raw
inputs plus its instrument's four group codes (64 more inputs). 166k
parameters (twin 131k); about 34.5 kMAC more per twin (encoders 6.9k, wider
stream inputs 27.6k). Same recipe as twin96 s71 on h0 (MPS) and h1 (CPU).
Rule: beat twin96 s71 alone on both halves and on the unseen sequences, or
improve a v36-like set when it replaces one twin.
**1. Result: lowering the teacher weight hurts; rejected.** Continuation from
twin s71 e5. h1: teacher 0.25 → 0.68173 / 0.678719 / 0.678032, teacher 0 →
0.679934 / 0.674331 / 0.673319 (0.75: 0.683989 / 0.6838 / 0.683678). h0:
teacher 0.25 → 0.686783 / 0.686403 / 0.685657, teacher 0 → e1 0.686 (0.75:
0.686662 / 0.686948 / 0.686816); the teacher-0 h0 run was stopped after e1.
More label weight fits label noise and falls further each epoch: the teacher
regularizes the twin rather than holding it back. Candidate 2 started on
both halves (h0 on MPS directly; h1 on CPU after its queue).

**2. Column-interaction encoder, h1: large out-of-period gain.** h1 e1–e5
0.677034 / 0.682423 / 0.680713 / 0.683055 / 0.68309, continuation 0.683686 /
0.68381 / 0.683943 (twin96 s71: e5 0.68316, continuation 0.683989 / 0.6838 /
0.683678); h0 so far e3–e5 0.68254 / 0.686321 / 0.687238 (twin96 s71
0.681643 / 0.685724 / 0.686185, e5 +0.0011), continuation running. h1 SWA
alone 0.683945 (twin96 s71 0.683943, s72 0.683341), correlation with s71
0.9944; pairs in period: enc + s71 0.684654, enc + s72 0.684374 (s71 + s72
0.684379). **Unseen train-period sequences with the h1 models: twin + encoder
alone 0.442327 vs twin96 0.439718 / 0.439844 (+0.0025)**; enc + s72 0.442774
vs s71 + s72 0.441462 (+0.0013); all three 0.442843. One half-trained
twin + encoder matches the all-data six-twin ensemble (0.442383) out of
period. Pending: the h0 model's out-of-period check. All-validation twin +
encoder seed 71 started on CPU in advance.

**Twin + encoder export.** The npz gains `event_inputs`, `n_encoders` and per
encoder e{k}_w1 (8, 32, 11), b1, w2 (8, 16, 32), b2, cols (8, 11) and slot.
Each encoder writes its 128 codes into an extended input vector [raw 112, 1,
events, codes …]; twin streams read them as extra input columns
(`stream_members(state, inputs, codes)` places the stream's 64 code weights
at its instrument's slot). The callback folds standardization into each first
encoder layer. NumPy path: two einsums per encoder per row. Kernel: new kind
4 (group MLP: gather 11 + bias, 32 tanh units, 16 codes per group) that runs
right after the runtime-guard member and before every member that reads
codes. Checks: student + two twin-encoders + split matches PyTorch (NumPy
4.8×10⁻⁷, guard 8.3×10⁻⁷; kernel 6.2×10⁻⁴, guard 1.1×10⁻³); twin-encoders
alone 5.4×10⁻⁷ / 7.8×10⁻⁴; AVX2 under Rosetta equals AArch64 to 3.6×10⁻⁷; six
twins unchanged. A twin-encoder is about 166 kMAC in four kernel members
(twin 131 in three): five about 42.5 minutes, six about 49.6.
**h0 confirmation: twin + encoder adopted.** h0 continuation 0.687546 /
0.687755 / 0.687682 (twin96 s71 0.686662 / 0.686948 / 0.686816). SWA alone
0.687817 (twin96 s71 0.686965, s72 0.687470), correlation with s71 0.9947;
pairs enc + s72 0.688325 vs s71 + s72 0.687899. Unseen train-period sequences
with the h0 models: alone 0.441134 vs twin96 s71 0.440788 / s72 0.439363;
enc + s71 0.442508, enc + s72 0.441843 vs s71 + s72 0.441628. Against twin96
s71 alone it is +0.00085 (h0) / tie (h1) in period and +0.0026 (h1 model) /
+0.00035 (h0 model) out of period, so it passes the rule fixed beforehand;
mean out-of-period gain over the four single comparisons about +0.0018. All-
validation twin + encoder seeds 71 (CPU) and 72 (MPS) training for a v37 of
twin-encoders.
All-validation twin + encoder seed 71 (CPU): e5 loss 0.3285, continuation
0.3247 / 0.3255 / 0.3231. Unseen train-period check: alone 0.440444 vs the
all-validation twin s71 0.439717 (+0.00073); with twin s72: 0.441100 vs twin
s71 + twin s72 0.440741 (+0.00036). Twin + encoder minus twin96 for the same
seed 71, out of period: h1 model +0.0026, h0 model +0.00035, all-validation
+0.00073 (mean +0.0012): always ahead, by a seed-dependent amount. Seeds 72
(MPS) and 73 (CPU) training.
Seed 72 (MPS): continuation 0.3253 / 0.3244 / 0.3231; unseen check alone
0.440961 vs the all-validation twin s72 0.438290 (+0.0027). Pairs: twin-enc
s71 + s72 0.442354 vs twin s71 + s72 0.440741 (+0.0016) — two twin-encoders
already match the six-twin v36 (0.442383). Seeds 73 (CPU, continuation) and
74 (MPS) training; seeds 75/76 follow for a five- or six-member v37
(four twin-encoders about 35.5 minutes, five about 42.5, six about 49.6).
Seed 73 (CPU): continuation 0.3253 / 0.3253 / 0.3227; alone 0.439797 vs twin
s73 0.440233 (−0.00044). Per-seed twin-enc minus twin (all-validation):
s71 +0.0007, s72 +0.0027, s73 −0.0004 (mean +0.001). Three-member sets:
twin-encoders s71–73 0.442606 vs twins s71–73 0.441750 (+0.00086) and v36
(six twins) 0.442383. Seeds 74 (MPS) and 75 (CPU) training; v37 (five
twin-encoders, about 42.5 minutes) once both finish.
Seeds 74 (MPS) and 75 (CPU): continuation final loss 0.3231 / 0.3227 (the
laptop slept on battery during their last epochs, which took about 4 hours);
alone on the unseen sequences 0.441013 / 0.441765. Twin-encoders alone:
s71–75 0.440444 / 0.440961 / 0.439797 / 0.441013 / 0.441765 (twins s71–76
0.4383–0.4408). Four twin-encoders (s71–73, 75) 0.443270. **v37 = five
twin-encoders (s71–75, equal weight): 0.443487 vs v36 0.442383, +0.0011,
bootstrap 95% [+0.00006, +0.0021], P(≤ 0) 0.016.** `submission_v37.zip`
(3,136,231 bytes): PyTorch parity 4.1×10⁻⁷ (NumPy) and 4.3×10⁻⁴ (kernel);
20 kernel members (five encoders of kind 4, gathered streams, joints), 831
kMAC, about 42.6 official minutes; guard zero stage only.

**Official v37 (2026-10-14): 0.6604 in 43m37s** (estimate 42.6): +0.0001 over
v36, within board noise; the unseen-sequence gain (+0.0011) transferred at
about a tenth this time (v36: +0.00076 → +0.0005). Final pair: v37 + v33.
**Ensemble size (unseen train-period sequences, equal weights).** Twin-
encoders × 1–5: 0.440444 / 0.442354 / 0.442606 / 0.443044 / 0.443487 (still
rising, +0.0004 from four to five); twins × 4 / 6 / 8: 0.441941 / 0.442383 /
0.442421 (saturated). Adding plain twins to the five twin-encoders lowers it
(+1 twin 0.443454, +2 0.443229, +8 0.443124). Runtime caps the count at six
twin-encoders (about 50 minutes; seven about 57). Twin-encoder seed 76
training on CPU for v38 (six twin-encoders), the next seed in order, no
selection by score.
Seed 76 (CPU): continuation 0.3260 / 0.3242 / 0.3226; alone 0.441287. **Six
twin-encoders 0.443680 vs five 0.443487: +0.00019, bootstrap 95% [−0.00013,
+0.00051], P(≤ 0) 0.11** — no longer distinguishable from zero.
`submission_v38.zip` = twin-encoders s71–76, equal weight (3,765,900 bytes;
parity 4.0×10⁻⁷ NumPy, 5.2×10⁻⁴ kernel; about 49.6 minutes by the model, v37
ran 1 minute over its estimate, so about 50.5). Expected public change within
noise; final pair stays v37 + v33 unless v38 is clearly ahead.

## Round 21: side encoders (2026-10-15)

`--cell twinenc2` = TwinEncGRU plus four side encoders: each book side's 11
price and 11 volume levels together (22 columns) through an MLP 22 → 48 → 16,
so a price level and its volume can interact before the GRU; stream inputs
raw 60 + group codes 64 + side codes 32 = 156 (188k parameters; about 187
kMAC per model, five about 47 minutes). Screening h1 (CPU) and h0 (MPS), seed
71, 5 epochs + continuation, against twin-encoder s71. Rule fixed now: adopt
if the unseen train-period check is not worse for either half model and
better on their mean, and neither half loses more than 0.0003 in period.
Export generalized: a model may carry several encoder blocks (groups, side),
each a kind-4 kernel member (the kernel's gather buffer now sized by the
block's column count); codes per model 128 or 192. A random side-encoder
model mixed with a trained twin-encoder matches PyTorch (NumPy 3.8×10⁻⁷,
kernel 6.0×10⁻⁴); AVX2 under Rosetta equals AArch64 to 2.4×10⁻⁷.

**Disk cleanup (2026-10-15, with approval).** Deleted 9.5 GB of caches the
current pipeline does not read: `stack/valid_hidden.npz` (round 1 hidden
states), the six `teacher*_y4607.npy` segments (train pool beyond 6,000,
rejected), the pilot arrays (`fit_*`, `dev_*`; `pilot.py prepare` rebuilds
them), `advance/fold_1–3`, `head_screen`, `sgru`, `warmup_adapter`, `eventgru`
and test packages. Kept: `improve/` (the packager's standardizer),
`advance/full_fit/distill/groups.npy` and `teacher.npy`, gen-1 and gen-2
teacher caches, `maskft`, `distill_mps`, `dev_check`, all submission packages.
Free disk 5.6 GB → 30 GB.
**h1 result: fails the out-of-period rule.** h1 e1–e5 0.677199 / 0.681763 /
0.681375 / 0.683609 / 0.683461, continuation 0.684159 / 0.68406 / 0.683932
(twin-encoder s71 0.683686 / 0.68381 / 0.683943); h0 so far e3–e5 0.682408 /
0.686031 / 0.687141 (twin-encoder 0.68254 / 0.686321 / 0.687238). h1 SWA alone
0.684160 vs twin-encoder 0.683945 (+0.0002), correlation 0.9947; with twin
s72 0.684468 vs 0.684374. Unseen train-period sequences (h1 models): side
0.441405 vs twin-encoder 0.442327 (−0.0009); with twin s72 0.442338 vs
0.442774; side + twin-encoder 0.443423 (a diversity pair, +0.0006 over
twin-encoder + twin). Waiting for the h0 model before the final call.
**h0 and verdict: side encoders rejected.** h0 continuation 0.686902 /
0.687368 / 0.687316 (twin-encoder 0.687546 / 0.687755 / 0.687682); SWA alone
0.687337 vs 0.687817 (−0.0005); with twin s72 0.688083 vs 0.688325. Unseen
train-period sequences (h0 models): side 0.442132 vs twin-encoder 0.441134
(+0.0010); with twin s72 0.442369 vs 0.441843; side + twin-encoder 0.443130.
Out of period −0.0009 (h1) / +0.0010 (h0), mean about 0; in period +0.0002 /
−0.0005. It fails the rule set beforehand on two counts and adds no more
diversity than another twin-encoder seed (correlation 0.9947). The per-group
encoder's gain does not extend to joint price-volume side encoders. v37
(five twin-encoders) stays the best package; final pair v37 + v33.

## Round 22: delayed target estimate at student scale (2026-10-16)

`delayed_feature.py`: the Round 16 reconstruction (ridge on i0 book and trade
prices at the completed interval's last row, at the event row and their
difference) fitted on 1,500 train sequences outside the student pool and DEV
(6.5 million events); held until the next event, zeros before the first;
self-checked against the pilot's reconstruction and for causality. Pre-check
before any training, on h1 scored rows: delayed estimate alone 0.481526,
twin-encoder alone 0.683945, correlation 0.636; output blends (1 − a) · model
+ a · delayed, scored on odd / even held-out sequences: a = 0 0.678933 /
0.689285, 0.05 0.678661 / 0.688991, 0.1 0.678021 / 0.688323, 0.2 0.675366 /
0.685588 — lower on both at every weight. The pilot's +0.0015 came from a
label-only GRU96 that lacked this information; the twin-encoder already has
it. The student-scale training run is not started.

## Round 23: recalling the same book state (2026-10-16)

Pre-check (twin-encoder s71, h1, 468 held-out sequences, every row): 92.5% of
scored rows have an earlier visit to their exact i0 price state, median gap 48
rows; the prediction at the previous visit's last row scores 0.407 there (model
0.662) and output blends lower WP at every weight. By gap: 100–500 rows (21.6%
of scored rows) blend 0.1 −0.0019; 500–2000 (6.9%) +0.0023; over 2000 (3.5%)
+0.0003 — the GRU forgets only a little, beyond about 500 rows.

**B, rule-based recall (`recall_rule.py`, no training): rejected.** Rule fixed
beforehand: when the state's previous visit ended at least 500 rows ago, output
0.9 · current + 0.1 · prediction at that visit's last row. Unseen train-period
sequences (v37 on every row): +0.00059, bootstrap 95% [−0.0006, +0.0020];
validation halves (twin-encoder s71 half models): h0 −0.00023, h1 −0.00010
(a = 0.05: −0.00006 / +0.00001; gap ≥ 1000: −0.00012 / −0.00015). It fails
the in-period part of the rule. The learned memory (A) would chase the same
small pocket (about 10% of scored rows, little information left).

**Official v38 (2026-10-15): 0.66049 in 53m32s** (v37 0.66045 in 43m37s):
+0.00004, as the unseen-sequence check predicted (+0.0002, not significant).
It ran 3.9 minutes over its 49.6-minute estimate (v37: +1.0), leaving 6.5
minutes to the limit; a final set about 9% larger would reach the zero stage.
Final pair stays **v37 + v33**.

## Round 24: spread component S re-examined (2026-10-16)

S = −(t0 + t1) in ticks, 313 validation sequences. Tick 0.657 (117
sequences): median 8 i0 price states per sequence, scored S 1 tick 76%, 2
ticks 19%; S never changes off events; unchanged at 36% of events; within-
sequence state purity 0.865; modal S of a state agrees across sequences 0.908.
Tick 0.469: 23 states, S 1 (56%) / 2 (31%), purity 0.684. Tick 0.241: 202
states, S from −9 to +5, purity 0.537. Tick 0.125: 1,047 states, S from −16 to
+9 with about 35% of scored rows at S ≤ 0 — not a spread: t0 and t1 cannot
share one future price there.

Decisive test: a state → mean S lookup from 1,200 train sequences (states with
≥ 50 rows), applied to the twin-encoder s71 h1 model on its held-out half (true
tick as an upper bound), replacing the model's S at weight w and keeping its D.
Correlation of S with the model's S / the lookup: tick 0.125 0.331 / 0.248,
0.241 0.322 / 0.260, 0.469 0.234 / 0.356, **0.657 0.044 / 0.199**. WP at w =
0 / 0.25 / 0.5 / 1: 0.125 0.79756 / 0.79707 / 0.79552 / 0.78924; 0.241
0.78252 / 0.78219 / 0.78118 / 0.77707; 0.469 0.72091 / 0.72048 / 0.71921 /
0.71434; 0.657 0.42310 / 0.42244 / 0.42039 / 0.41237; all 0.683945 /
0.683525 / 0.680341 / 0.664238. The model carries almost no S information in
the 0.657 group, yet S at correlation 0.2 only adds noise to t0 and t1: the
+0.038 oracle needs near-exact S, which is only revealed after the fact. The
0.657 group's low score comes from D, the future direction. C1: no-go.

## Round 25: third-generation teacher (2026-10-17)

Students learn 0.75 from the gen-2 teacher (gen-1 fine-tuned GRUs, split
encoder, distilled GRU128, GRU students), which predates the twin family.
`--gen 3`: teacher3 = 0.5 · teacher2 + 0.5 · mean(twin-family students),
cached the same way (only the added models are run): per half twin s71, twin
s72 and twin-encoder s71 (half models); for all validation twin-encoders
s71–75 and twins s71–75 (all SWA over the continuation epochs). Gen-1 → gen-2
gave students +0.0003. Test: twin-encoder s71 distilled from teacher3 on h0
(MPS) and h1 (CPU), same recipe, against the gen-2 twin-encoder s71. Rule:
adopt if the unseen train-period check is not worse for either half model and
better on their mean, and neither half loses more than 0.0003 in period.
**Result: not adopted (fails the in-period part of the rule).** h0 e1–e5
0.680719 / 0.682988 / 0.682244 / 0.686796 / 0.688114, continuation 0.68831 /
0.688487 / 0.68821 (gen-2 twin-encoder s71 0.687546 / 0.687755 / 0.687682); h1
e1–e5 0.677155 / 0.682171 / 0.680701 / 0.683224 / 0.682689, continuation
0.683027 / 0.683568 / 0.683214 (gen-2 0.683686 / 0.68381 / 0.683943). SWA of
the continuation epochs, gen-3 vs gen-2: h0 in period 0.688490 vs 0.687817
(+0.00067), unseen train-period sequences 0.441608 vs 0.441134 (+0.00047); h1
in period 0.683393 vs 0.683945 (−0.00055), unseen 0.443416 vs 0.442327
(+0.00109). Out of period both half models improve (mean +0.00078), but h1
loses more than the 0.0003 tolerance in period. The all-data gen-3 students
(about 3 hours of teacher cache plus about 6.5 hours per seed) are not started
under the rule; that is a separate decision.

**Official v39 (2026-10-19): 0.66052 in 51m02s** (v38 0.66049 in 53m32s, v37
0.66045 in 43m37s). `submission_v39.zip` (built 2026-10-19; 3,765,695 bytes;
parity 4.0×10⁻⁷ NumPy, 3.9×10⁻⁴ kernel) = the gen-3 twin-encoder s71 half
models (h0 and h1, which together saw all validation) and the all-data gen-2
twin-encoders s71–74, equal weights. Unseen train-period check (`dev_check.py`
now also predicts the seed-71 half models of both teacher generations; paired
sequence bootstrap, 2,000 resamples): v39 0.444477 vs v38 0.443680 (+0.00080,
95% [−0.00006, +0.00169], P(≤ 0) 0.038) and v37 0.443487 (+0.00099, [+0.00020,
+0.00176], P(≤ 0) 0.008). The same set with the gen-2 half models scores
0.444035: the gen-3 teacher adds +0.00044 ([+0.00004, +0.00081], P(≤ 0) 0.015);
half models in place of all-data seeds 75/76 add +0.00035 (P(≤ 0) 0.17). The
board kept the check's order (v39 > v38 > v37) at under a tenth of its size.
Runtime: about 7.5 minutes to the zero stage (v38, the same six-member
structure, ran 2.5 minutes longer), so v39 tolerates a final set about 10–14%
larger; v37 about 30%. The rules do not state the final dataset's size.
Proposed final pair: v39 (best on both checks) + v37 (same family, 16 minutes
of margin); v33 stays the guarded, different-composition alternative.
