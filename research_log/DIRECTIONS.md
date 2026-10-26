# Alpha Connectome direction map (2026-10-03)

Goal: improve the best official score (v24, 0.65914) under one CPU, 60 minutes,
16 GB and causal row-by-row inference. The promotion rule is improvement on
both validation halves; tiny gains are treated as uncertain until an official
submission confirms them.

## Information and features

| Direction | Evidence | Status |
|---|---|---|
| Raw 112 features | Strongest reusable input | keep |
| Standardization/clipping | Stable across all strong GRUs | keep |
| i0/i1 shared split encoder | +0.0001–0.0002 in ensembles | keep as diversity member |
| Fixed lags / EMA history | ≤+0.00046 FIT-CV | closed |
| i0/i1 event-flow features | negative in cross-fit residual tests | closed |
| Event-compressed sequence | 0.5918 vs ≈0.684 teacher | closed |
| Tick-group specialists/calibration | unstable or negative across halves | closed |
| Price-state lookup | validation corr only 0.22; blend unstable | closed |
| Validation-like train selection | below validation-only fine-tuning | closed |
| Explicit step/time calibration | 0.68458 vs 0.68481 control, both halves down | closed |
| Delayed target reconstruction feature | pilot 0.31754 vs GRU96 0.31606 (+0.00148) | below +0.005 scale-up gate |
| Complete warm-up summary (mean/std/movement) | post-selected e4 FIT-CV blend +0.00202; validation adapter ≈0 and gen-2 student e2 −0.00015 | closed for deployment |
| Scoring-mask prediction | useful for loss weights, not as an inference input | keep only in training |
| Next i0 event oracle | 0.7525 with future event, diagnostic only | no inference use |
| External market data | prohibited by competition FAQ | forbidden |

## Model families

| Direction | Evidence | Status |
|---|---|---|
| GRU96/128/192 | best established family | saturated but keep |
| Split-encoder GRU128 | strongest diverse conventional member | keep |
| TCN48 | −0.0226 FIT-CV | closed |
| LSTM128 | epoch 1 below matched GRU, much slower | closed |
| Vanilla RNN128 | pilot 0.3021 vs GRU96 0.3161 | closed |
| Two-layer GRU64 | pilot 0.3180, only +0.0019 | below +0.005 scale-up gate |
| Target-specific dual GRU64 | pilot 0.3143 | closed |
| Event-time GRU | weak alone and negative in blends | closed |
| Causal Conv residual | below control | closed |
| Interval-reset nonlinear residual | below control | closed |
| Distribution head | small diversity gain, extra runtime | optional, low priority |
| Mask-gated experts | +0.0002 vs head control, no teacher gain | closed |
| Nonlinear hidden readout | severe cross-fit overfit | closed |
| Selective SSM pilot | CPU kernel feasible; 0.2993 vs GRU96 0.3161 | closed at pilot gate |
| Full Mamba-2 reference | optimized package requires CUDA/Triton | not deployable here |
| Local causal attention 64×4 | pilot 0.3039 vs GRU96 0.3161 | closed at pilot gate |
| Causal multi-scale window mixer | pilot passed, but fold-1 0.3864 vs GRU96 0.4060; best blend only +0.00048 | closed |
| Two-layer GRU128 | h0 0.6810; 5% strong-proxy blend −0.00001 | closed |
| Three-layer minGRU128 | h0 0.68372; best strong-proxy blend +0.00008 | closed |
| GRU96 with 2,048-row BPTT | pilot 0.31381 vs 0.31606 | closed |
| Warm-up-initialized GRU96 | pilot e3 +0.00266, fixed e5 −0.00184 | closed |

## Objectives and training

| Direction | Evidence | Status |
|---|---|---|
| abs(target)-weighted MSE | matches metric well | keep |
| Target-balanced loss | +0.00023 on split encoder, both halves | keep |
| Pearson loss | no gain | closed |
| Target smoothing | strongly negative | closed |
| Next-event auxiliary loss | below matched control | closed |
| Next-event pretraining | −0.00164 converged pilot; early warm-start only | closed |
| Predictive contrastive pretraining (1/8/32 rows) | 0.31379 vs matched GRU96 0.31606 | closed |
| Dedicated D = t0−t1 GRU | single-GRU gain +0.0025–+0.0040 on all CV folds; strong-ensemble blend −0.00160 | closed for deployment |
| Shared-trunk D auxiliary head | peak +0.00056; frozen epoch-5 −0.00244 | closed |
| Future-event privileged distillation | early +0.00646; converged −0.00070; one-epoch curriculum +0.00009 | closed |
| Hard event bottleneck | −0.00355 pilot | closed |
| Distillation | largest efficient gain; scratch students are strongest singles | keep |
| Second-generation teacher | ≈+0.00015 package-level, seed-dominated | saturated |
| SWA | repeatable gain | keep |
| Weight EMA | +0.0004 single, only +0.00004 in ensemble | closed |
| Low-LR continuation | +0.00038 for continued gen-2 pair package proxy | promote |
| Validation 3× continuation | epoch 1 −0.00023 vs ordinary continuation | closed |
| More width / label-only training | overfits | closed |

## Ensemble and post-processing

| Direction | Evidence | Status |
|---|---|---|
| Equal-weight diverse members | robust default | keep |
| Global fitted weights | +0.00001 over v21 proxy | not material |
| Tick-specific fitted weights | no gain over v21 proxy | closed |
| Runtime-constrained cross-fit subset | 0.68494 vs v21 proxy 0.68481 | useful structure |
| GRU192 + split + gen1/gen2 student | 0.68508, both halves; 465 kMAC | packaged as v23 |
| GRU192 + split + continued gen-2 pair | 0.68521, both halves; 465 kMAC | next package |
| More than five members | exceeds safe runtime for tiny gain | closed |

## Runtime and packaging

- Width-grouped NumPy GRUs are the measured Pareto implementation.
- v17 used 464 kMAC and 54m26s; 480 kMAC is a practical upper boundary.
- Callback/PyTorch agreement must remain below 5e-5; current packages are
  around 1e-6.
- A NumPy selective-state update benchmark was fast (0.205 seconds per 20k
  rows), but the locked pilot missed GRU96 by 0.0168, so it is not exported.

## Methodology audit

### Critical errors

None found in the new screens. Future event values are training labels only,
DEV is never used to fit pilot models, and cached half predictions are
sequence-disjoint.

### Material weaknesses

1. The starter-pack `README.md` and `METRIC.md` still describe legacy
   per-sequence aggregation, while the current official site specifies one
   Global WP pooled over selected rows. Research selection correctly uses
   `GlobalWP`; legacy `BlockAccumulator` results must not drive decisions.
2. Many ideas have been selected on the same two validation halves. Gains
   below roughly 0.0005 are exposed to multiple-testing selection bias.
3. Train and validation periods are highly separable (AUC ≈0.98). Validation
   fine-tuning transferred to v10, but a new family can still fail on test.
4. Final all-validation students are separate stochastic runs from per-half
   evidence; seed variance is about 0.0015 for a single student.
5. Public leaderboard submissions are another adaptive validation channel.

Sequence bootstrap makes the uncertainty explicit. Against the v21 proxy,
v23's gain is +0.00021 with a 95% interval of [-0.00023, +0.00067]
(82% positive); the continued-student structure is +0.00040 with
[-0.00011, +0.00090] (94% positive). Neither is conclusive without the
leaderboard, although continuation is the stronger bet.

### What is correct

- Causal sequence boundaries and state resets are preserved.
- Normalizers are fitted on training/FIT data.
- Validation-half fitting and scoring are disjoint by sequence.
- The current Global Weighted Pearson matches the official web specification.
- Packages are checked against PyTorch and the callback contract.

## Priority queue

1. Keep v24 as the official baseline and v25 as the faster fallback; do not submit more variants of this
   GRU/distillation family.
2. Do not scale the causal multi-scale window mixer: its first larger fold
   failed and its best 10% blend gained only +0.00048.
3. The next candidate must use a new learned representation; fixed-window,
   lag and EMA variants are now closed alongside the GRU tuning branch.
4. Delayed target encoding is the only new pilot with a positive converged
   result (+0.00148), but its nonlinear reconstruction diagnostic reached
   only 0.5365 event WP and it remains below the scale-up gate.

Predictive contrastive pretraining was the next learned-representation screen.
Its self-supervised loss fell 2.3585 → 1.7109 and target epoch 1 improved by
0.00685, but its converged best was 0.313787 versus 0.316064 for the matched
GRU96. Like next-event pretraining, it is only an optimization warm start and
is closed without larger runs.

A dedicated future-direction model `D = t0 − t1` was the first target-
structure screen to generalize beyond the pilot. With epoch 5 and 50% D
replacement frozen, its gains over a normalized GRU96 were +0.004022,
+0.002515 and +0.002504 on the three 1,024-train/512-test folds. However, it
added only +0.000536 to the weaker three-GRU ensemble and reduced the strongest
fixed fold-1 ensemble by 0.001601. Strong distilled/split ensembles already
capture its D signal, so full-validation refitting and v27 packaging are not
justified.

With two final continuations available, package both Pareto points:

- safe: `submission_v25.zip`, continued pair at 0.4 each + split at 0.2
  (proxy 0.685181, 290 kMAC);
- score: `submission_v26.zip`, continued pair + GRU192 + split, equal weights
  (proxy 0.685207, 465 kMAC, same runtime shape as the already accepted v17).

Adding two uncontinued gen-2 students to the safe set lowers the proxy to
0.685055, so G77/G78 are not fillers for the final package.

## Official v25/v26 decision (2026-10-04)

| Package | Held-out proxy | Official | Runtime | Decision |
|---|---:|---:|---:|---|
| v25: continued pair + split | 0.685181 | **0.6590** | **39m21s** | keep as the public baseline |
| v26: v25 + GRU192 | 0.685207 | 0.6589 | 54m37s | reject |

The 0.000026 proxy advantage of v26 reversed by 0.0001 publicly, while its
runtime rose by 15m16s. This is below both sequence-bootstrap uncertainty and
leaderboard resolution: the extra GRU192 supplies no demonstrated test
diversity. Relative to v18, v25 gains only 0.0003, confirming that continuation
improves the existing family but does not change its ceiling.

Do not submit more seed, width, continuation or ensemble-weight variants of
this GRU family. A new package is warranted only after an orthogonal candidate
beats a matched control by at least 0.005 on the locked pilot or by at least
0.002 on both full validation halves. The next research question is a new
information representation, not another blend of the existing predictions.

Warm-up conditioning was the next new-information screen. Appending a summary
of rows 0--98 to every GRU input looked positive in FIT-CV, but epoch 4 was
chosen after inspecting the fold histories. The stronger validation-half
tests did not reproduce it: a context-only adapter changed the v24-like set by
at most +0.000008 on half 0, and a jointly trained gen-2 student was +0.000349
at epoch 1 but −0.000145 at epoch 2 versus matched G71. Initializing the hidden
state from the same summary ended −0.001841 on the locked pilot. The family is
closed unless a genuinely untouched split is introduced.

The first such pilot passed after this decision. A 96-wide causal mixer uses
the current row plus changes between rolling means at 4, 16 and 64 rows. On
the locked 128-FIT/32-DEV pilot, seeds 7/8/9 all peaked at epoch 3: 0.328040,
0.322057 and 0.331760 versus the frozen GRU96 control 0.316064. The mean gain
was +0.011222. On the first larger 1,024-train/512-test fold it fell to
0.386436, versus 0.405980 for the normalized GRU96. Adding 10% of the mixer
to the strongest fixed ensemble moved 0.418084 to only 0.418564 (+0.000480),
below the +0.002 gate. The other folds and submission build are cancelled.
The `u=0.02` baseline improves alone by 0.000144 but also lowers both final
ensemble proxies, so it is not refitted for deployment.

## AFML uniqueness-weighting audit (2026-10-06)

Exact AFML average uniqueness cannot be computed here because the competition
does not provide label event start/end spans. A train-only proxy based on
inverse target-persistence run length was screened in
`uniqueness_weight_screen.py`; validation never used the weights. Its locked
pilot result was misleadingly positive (power 0.5: +0.01428, power 1.0:
+0.01064 at e5), but the first sequence-disjoint group fold did not support it:

| Weight power | Group-fold control | Fixed e5 | Gain | Gate |
|---:|---:|---:|---:|---|
| 0.5 | 0.405980 | 0.406515 | +0.000535 | fail (+0.002) |
| 1.0 | 0.405980 | 0.394063 | −0.011917 | fail |

The extended power-0.5 check is even weaker:

| Fold | Control | Fixed e5 | Gain |
|---:|---:|---:|---:|
| 1 | 0.405980 | 0.406515 | +0.000535 |
| 2 | 0.427730 | 0.421405 | −0.006325 |
| 3 | 0.412565 | 0.411959 | −0.000607 |
| **mean** | — | — | **−0.002132** |

Only one of three folds was positive (sample SD 0.003676). This is a closed
direction. The correct AFML conclusion is not that uniqueness weighting is
harmful in general, but that this persistence proxy is not reliable evidence
for this data and should not be promoted to a full training run or submission.

The companion clustered MDA screen (`cache/step01/cluster_mda.json`) is useful
for prioritization: swapping i0 bid/ask volume clusters reduced pooled WP by
about +0.0516/+0.0510 on average across the two halves, while i1 clusters
were +0.001--0.004. Because correlated clusters substitute for one another,
this is not permission to delete i1 features. If another model family is
attempted, it should allocate capacity to a learned i0 representation and
test any i1 simplification only on a fresh sequence-disjoint split.
