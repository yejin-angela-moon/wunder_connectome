# Advanced model screens

All model selection in this report uses the existing locked FIT-only three-fold group screen. Each fold trains on 1,024 complete FIT sequences and scores 512 disjoint FIT sequences with Global Weighted Pearson. DEV and public OOS were not read until after the distilled candidate was promoted and refitted on complete FIT. Sequence order is shuffled, so these are group-CV results rather than chronological walk-forward results.

The reference is the normalized GRU96 trained for five epochs with the existing learning-rate decay: **0.415425099** mean FIT-CV Global WP. The predeclared promotion rule remains a mean gain of at least 0.005 with gains in at least two folds.

| Screen | Fold 1 | Fold 2 | Fold 3 | Mean | Gain vs family reference | Decision |
|---|---:|---:|---:|---:|---:|---|
| GRU96 + lags 1/8/32/128 + gradient clipping | 0.403737 | 0.418208 | 0.409996 | 0.410647 | -0.004778 | Reject |
| GRU96 + lags + EMA residuals 8/32/128 | 0.407552 | 0.423698 | 0.416411 | 0.415887 | +0.000462 | Reject |
| Shared i0/i1 encoder + difference/product interactions | 0.410931 | 0.430733 | 0.411375 | **0.417680** | **+0.002255** | Preserve for blend check; below promotion rule |
| Common/difference target head | 0.406511 | 0.425671 | 0.413370 | 0.415184 | -0.000241 | Reject |
| GRU96, eight epochs | 0.402855 | 0.425331 | 0.412542 | 0.413576 | -0.001849 | Reject |
| GRU128, five epochs | 0.409208 | 0.428527 | 0.412789 | **0.416841** | **+0.001416** | Preserve for blend check; below promotion rule |
| GRU96 + gradient clipping | 0.406236 | 0.428306 | 0.412572 | 0.415705 | +0.000280 | Reject |
| GRU96 random windows with 128-row burn-in | 0.391873 | 0.414876 | 0.389924 | 0.398891 | -0.016534 | Reject |
| Causal TCN48, 511-row receptive field | 0.386358 | 0.403420 | 0.388728 | 0.392835 | -0.022590 | Reject |
| GRU96 after causal EMA centering | 0.402530 | 0.419306 | 0.406199 | 0.409345 | -0.006080 | Reject |
| GRU96 after causal EMA variance normalization | 0.395349 | 0.412786 | 0.405393 | 0.404509 | -0.010916 | Reject |
| Shared GRU96 + low/high-volatility output heads | 0.405901 | 0.427898 | 0.410459 | 0.414752 | -0.000673 | Reject |
| MLP lag32, random sample offset each epoch | 0.353748 | 0.378193 | 0.366477 | 0.366139 | +0.000386 vs MLP reference | Reject |

The strongest new standalone result was the structured i0/i1 encoder at 0.417680, followed by GRU128 at 0.416841. Neither met the existing +0.005 promotion rule. Longer training reduced training loss but reduced validation WP, and random-window training materially damaged full-sequence performance. The small TCN was both slower and less accurate than GRU96.

## Volatility diagnostic

The existing normalized GRU96 was also scored separately above and below a causal volatility threshold. Each threshold was the median calculated from inputs in 32 groups belonging only to that fold's fitting side.

| Fold | Low-volatility WP | High-volatility WP | Low rows | High rows |
|---|---:|---:|---:|---:|
| 1 | 0.303330 | 0.468664 | 5,171,560 | 5,017,752 |
| 2 | 0.312418 | 0.496078 | 5,189,509 | 4,999,803 |
| 3 | 0.297195 | 0.475610 | 4,779,178 | 5,410,134 |

The performance gap is large and consistent, but a hard two-head expert did not improve the pooled metric. The regime diagnostic therefore does not justify further expert complexity yet.

## FIT-only row-level blend check

The two weak positive standalone candidates were combined from their actual validation-row predictions with the existing three-GRU ensemble. The existing ensemble score was reproduced exactly before comparison.

| Blend | Fold 1 | Fold 2 | Fold 3 | Mean | Gain vs 3-GRU ensemble | Decision |
|---|---:|---:|---:|---:|---:|---|
| Existing 3-GRU ensemble | 0.413768 | 0.435500 | 0.417570 | 0.422280 | — | Reference |
| 75% ensemble + 25% split encoder | 0.416751 | 0.438450 | 0.420212 | 0.425138 | +0.002858 | Below promotion rule |
| 75% ensemble + 25% GRU128 | 0.416511 | 0.437754 | 0.420748 | 0.425005 | +0.002725 | Below promotion rule |
| 80% ensemble + 10% each candidate | 0.416565 | 0.438111 | 0.420477 | 0.425051 | +0.002771 | Below promotion rule |
| Equal ensemble/split/GRU128 | **0.418084** | **0.438683** | **0.421509** | **0.426092** | **+0.003812** | Consistent, but below +0.005 rule |

All five-model blend variants improved all three folds, so the two new models contain useful diversity. The best gain did not meet the predeclared threshold and would expand inference from three GRUs to five, so full-FIT refitting is not justified.

## Three-GRU distillation

A single normalized GRU96 student was trained with an equal mixture of the original absolute-target-weighted label loss and weighted imitation loss against the existing three-GRU ensemble. Teacher predictions were generated only from each fold's fitting side; the fixed validation groups remained unseen by both student fitting and selection.

| Model | Fold 1 | Fold 2 | Fold 3 | Mean | Gain vs single normalized GRU |
|---|---:|---:|---:|---:|---:|
| Single normalized GRU reference | 0.405980 | 0.427730 | 0.412565 | 0.415425 | — |
| Three-GRU teacher | 0.413768 | 0.435500 | 0.417570 | 0.422280 | +0.006855 |
| Distilled single GRU96 | **0.415383** | **0.434057** | **0.415967** | **0.421802** | **+0.006377** |

The student improved all three folds over the single-model reference and retained all but 0.000478 of the teacher's mean score, so it passes the promotion rule while reducing inference from three GRUs to one. In the absolute-target error diagnostic, the student was worse on small targets but had comparable or lower mean error for both targets in the score-dominant `|target| > 1` bin. The existing weighted loss already emphasizes these rows; the diagnostics do not justify another auxiliary-loss sweep, especially after the common/difference target head failed.

The already fitted full-FIT teacher ensemble was also exported as a callback package. Row-wise NumPy inference reproduced PyTorch within `7.79e-06`; repeated current-host checks took 1.167–1.206 seconds per 20,000-row sequence, projecting to 38.31–39.60 minutes for 1,970 sequences. The current ZIP is 681,959 bytes. This is a host projection, not a Linux 1-vCPU benchmark.

## Frozen full-FIT confirmation and package

After promotion, the distilled student was trained on all 9,546 FIT sequences with the same fixed loss, seed, five epochs, and learning-rate schedule. It was then checked once on fixed DEV and once on the already-viewed public OOS set.

| Frozen model | Fixed DEV Global WP | Public OOS Global WP | Local projected runtime | ZIP bytes |
|---|---:|---:|---:|---:|
| Three-GRU teacher | 0.443056 | 0.673970 | 38.31–39.60 min | 681,959 |
| Distilled single GRU96 | **0.444501** | **0.675613** | **13.25–14.38 min** | **229,735** |
| Supplied baseline | — | 0.617052 | — | — |

The student exceeded the teacher by 0.001445 on fixed DEV and 0.001643 on public OOS, while reducing the local projected callback runtime by about 66%. Its public OOS gain over the supplied baseline is 0.058560. These OOS numbers are local public-validation replays, not hidden-test estimates; the public set had already been viewed during earlier work.

The NumPy callback reproduced the full-FIT PyTorch student within `1.07e-06` over a complete 20,000-row sequence. The extracted ZIP passed reset/warm-up smoke tests and is at `research/cache/packages/gru96_distill.zip`.

## Remaining external checks

1. A true Linux 1-vCPU benchmark in the competition runner; the local timing range is not a time-limit guarantee.
2. Competition upload and hidden scoring. No package was uploaded from this workspace.

Conditional expansions were also not run because their prerequisite screen failed: two-layer GRU after the weak GRU128 result, larger TCN after TCN48 underperformed, combinations of the split encoder and target head after the target head failed, and EMA-span/expert sweeps after causal normalization and the hard regime head failed.

Exact ignored artifacts and scores are under `research/cache/advance/`. The executable screens are `research/advance.py` and `research/distill.py`.
