# Sequential improvement screen

This screen reuses the locked FIT/DEV split and the three fixed group-CV folds in `group_cv.py`. Each fold fits on 1,024 FIT sequences and scores 512 disjoint FIT sequences with the current Global Weighted Pearson metric. No cross-sequence clock order is inferred. All input transformations are causal within one sequence; each normalization is estimated only from that fold's fitting groups. Existing artifacts and public OOS scores are reference values, not selection data.

Experiments, in order:

1. Repeat GRU widths 64 and 96 at seeds 7, 17, and 37. Seed 7 uses the saved original fold checkpoints; the other seeds are refitted from scratch. Compare paired width gains across folds and seeds.
2. At GRU width 96, compare training-input standardization; five epochs at the original 0.001 learning rate; five epochs with learning rate 0.001 for epochs 1–3 then 0.0003 for epochs 4–5; and standardization plus the five-epoch decay schedule. Keep the original three-epoch GRU as reference.
3. Compare ridge with causal lags `(1, 8)`, `(1, 8, 32)`, and `(1, 8)` plus a span-32 causal EMA against the saved lag-1 ridge. Every lag is `x[t]-x[t-lag]` and is zero before the lag is available. The EMA starts at the first row of each sequence.
4. Compare normalized MLP at four epochs and at causal lag sets `(1, 8)` and `(1, 8, 32)` against the saved normalized two-epoch MLP.
5. Compare simple averages: original GRU96 three-seed, best screened GRU96 plus the other two seeds, best GRU96 plus GRU64, and best GRU96 plus 10% of the best screened ridge, MLP, or both. Check all required rows, early/late rows, and a fixed random 20% mask. Inspect output scales 2, 4, 8 and shifts ±0.2 after scaling by 4; their effect can arise only through clipping by the official metric.
6. Compare a width-96 GRU trained with a 0.05 minibatch weighted-correlation auxiliary loss against the original weighted-MSE GRU. This is an approximation to, not an exact optimization of, Global Weighted Pearson.

These are exploratory comparisons because multiple candidates share the same three folds. Full-FIT refitting and the fixed DEV check follow model selection. The already-viewed public OOS file remains excluded from the screen and any weight fitting.

## 1. Seed stability

All scores below are three-fold FIT-only Global Weighted Pearson means on all required rows. The saved seed-7 checkpoints reproduced their earlier fold scores exactly.

| Seed | GRU64 | GRU96 | GRU96 − GRU64 |
|---:|---:|---:|---:|
| 7 | 0.397298 | 0.403263 | +0.005965 |
| 17 | 0.401642 | 0.406662 | +0.005020 |
| 37 | 0.401222 | 0.406990 | +0.005767 |

All nine paired fold/seed width differences were positive. This supports width 96 as the next GRU reference; the values are still exploratory FIT-fold results, not official OOS results. Exact fold scores are in ignored `cache/improve/stage1_summary.json`.

## 2. GRU training changes

The width-96 seed-7 three-epoch reference is 0.403263. Fold-only feature standardization at the same width and training schedule scored 0.397020 across folds (−0.006243); it was lower on all three folds. Five epochs at the original 0.001 learning rate scored 0.409488 (+0.006225), improving all three folds. Five epochs with learning rate 0.001 for the first three epochs and 0.0003 for the last two scored 0.411999 (+0.008735 versus three epochs, +0.002510 versus five fixed-rate epochs), improving all three folds over the three-epoch reference. Standardization plus this five-epoch schedule scored 0.415425 (+0.003427 versus unstandardized five-epoch decay), improving all three folds over unstandardized five-epoch decay. Thus the sign of the standardization effect depended on training duration in this screen.

## 3. Causal Ridge history

The saved lag-1 ridge reference is 0.352280. Adding lag 8 scored 0.359464 (+0.007184), and adding lag 32 as well scored 0.363155 (+0.010875 versus lag 1). Both changes were positive on all three folds. Lag 1 and 8 plus span-32 EMA scored 0.381317 (+0.029037 versus lag 1), the best Ridge screen. NumPy emitted arithmetic warnings during matrix multiplication. Coefficients and required-row features were finite, and the scorer rejected any nonfinite required-row predictions. A second pass used float64 multiplication and clipped predictions on only required rows; all three EMA fold scores matched the original scores within 5×10⁻⁹. The source of the warnings remains to be isolated before packaging this Ridge variant.

## 4. Normalized MLP

The saved two-epoch MLP reference is 0.343203. Four epochs scored 0.351790 (+0.008587), two epochs with causal lags 1 and 8 scored 0.355991 (+0.012788), and two epochs with lags 1, 8, and 32 scored 0.358487 (+0.015284). Each step was higher on all three folds. Combining lag 32 with four epochs scored 0.365753 (+0.007266 versus the two-epoch lag-32 model), with gains on two of three folds.

## 5. Simple ensembles and output calibration

The best single GRU was the normalized five-epoch GRU96 at 0.415425. Equal averaging with the two original-schedule GRU96 seeds 17 and 37 scored **0.422280** (+0.006855), higher on all three folds. The same three-seed average without the five-epoch model scored 0.414090. Averaging the best GRU with the original GRU64 scored 0.417543. Mixing 10% of the best Ridge or MLP into the best GRU scored 0.416987 or 0.417092; mixing 10% of each scored 0.418136. All of these use FIT-fold predictions only.

On the best single GRU, output scales 2, 4, and 8 scored 0.410783, 0.384261, and 0.359988, versus 0.415425 at scale 1. Adding ±0.2 after scale 4 scored 0.386377 and 0.381603. These operations altered the score through clipping and provided no gain. Early, late, and fixed 20% row-mask scores for all blends are recorded in ignored `cache/improve/stage5_summary.json`; the winning three-seed blend scored 0.425763, 0.418782, and 0.422485 on those masks, respectively.

The three-seed blend is the FIT-fold winner. Its three models were refitted on all 9,546 fixed FIT sequences for a locked DEV check. The public OOS file did not enter this screen.

## 6. Weighted-correlation auxiliary loss

Adding a 0.05 minibatch weighted-correlation penalty to the original three-epoch GRU96 scored 0.403634, only +0.000371 above its weighted-MSE reference. It did not approach the 0.415425 normalized five-epoch GRU, so this auxiliary loss was not selected.

## Complete FIT refit and fixed DEV check

The selected Ridge EMA, MLP lag-32/four-epoch, and three GRU ensemble members were refitted from scratch on all 9,546 locked FIT sequences. The same frozen artifacts were then scored on all 1,061 fixed DEV sequences and 21,114,961 required rows. Values below are Global Weighted Pearson, not the older block metric.

| Model | All DEV | Early | Late | Fixed random 20% |
|---|---:|---:|---:|---:|
| Ridge EMA32 | 0.382176 | 0.380419 | 0.383849 | 0.382550 |
| MLP lag32, 4 epochs | 0.393341 | 0.392250 | 0.394375 | 0.393820 |
| GRU96 normalized, 5 epochs | 0.433573 | 0.434398 | 0.432759 | 0.433863 |
| GRU96 seed17, 3 epochs | 0.420845 | 0.422388 | 0.419143 | 0.420967 |
| GRU96 seed37, 3 epochs | 0.424894 | 0.422878 | 0.426766 | 0.425356 |
| Frozen 1:1:1 GRU ensemble | **0.443056** | **0.443733** | **0.442275** | **0.443388** |

Relative to the previous complete-FIT selected variants on this same DEV set, Ridge improved from 0.353303 by +0.028873, MLP improved from 0.369475 by +0.023866, and the best single GRU improved from 0.414437 by +0.019136. The ensemble added +0.009483 over the new single GRU on DEV. These DEV values confirm the direction of the FIT-fold selection but are not an untouched test because DEV was inspected in earlier work. The public OOS comparison follows with no changes to weights or blend coefficients.

## Frozen public OOS comparison

The five refitted models and the frozen 1:1:1 GRU blend were scored on all 1,873 official validation sequences and exactly 3,528,273 rows selected by `need_prediction AND is_scored`. Their weights and blend were unchanged after the FIT-fold selection and DEV check. The independent baseline and previous selected-model scores below are from the prior complete Global WP replay on this same file and mask.

| Model | t0 | t1 | Global WP | Versus baseline | Versus previous family model |
|---|---:|---:|---:|---:|---:|
| Supplied baseline | 0.587913 | 0.646192 | 0.617052 | — | — |
| Ridge EMA32 | 0.621886 | 0.632448 | 0.627167 | +0.010115 | +0.029095 |
| MLP lag32, 4 epochs | 0.621509 | 0.637281 | 0.629395 | +0.012343 | +0.020369 |
| GRU96 normalized, 5 epochs | 0.665532 | 0.678660 | 0.672096 | +0.055044 | +0.010598 |
| GRU96 seed17, 3 epochs | 0.648615 | 0.668871 | 0.658743 | +0.041690 | — |
| GRU96 seed37, 3 epochs | 0.652235 | 0.667292 | 0.659764 | +0.042711 | — |
| Frozen 1:1:1 GRU ensemble | **0.667863** | **0.680077** | **0.673970** | **+0.056918** | **+0.012472 vs previous GRU** |

The ensemble added +0.001874 over its best single member on this public OOS set, smaller than its +0.006855 FIT-fold and +0.009483 DEV gains. No further weight or blend tuning follows from this observation. The public OOS labels and mask had been inspected in earlier work, so this is a public validation comparison, not an untouched hidden-test estimate or an official leaderboard submission. Scores used batched local inference; the new model callbacks and Linux 1-vCPU time limit have not been verified. Existing NumPy GRU callback timing suggests three serial GRUs could be close to the 60-minute test limit, so submission feasibility needs a separate runtime check.

Exact scores are in ignored `cache/improve/full_fit/dev_scores.json` and `cache/improve/full_fit/oos_scores.json`. Full-FIT artifact SHA-256: normalized GRU `a13499f213e8389c2893dc9029d32ec8ba6bfc830911afe05e50171b0ad334ce`, seed17 GRU `37c720383ff9ce729ad2914ab82652b9c40fcda4427affae2a4f02707fe037c6`, seed37 GRU `760150e4dbb9655c56d168b9e5d673819e634c8d33684683d0f4de444aa49348`, Ridge `076eed7a5c5aefac800a06979bb6b00ce9a8faf37aa6d25cd75c7d8970ddbbd1`, MLP `7ef50a34d98e8ca5f394b090702913b2e98a4a0a49d6f1e234ff4d0c124bd701`.
