# Alpha Connectome: research results

The [locked split](SPLIT.md) separates fit, internal development, and official OOS by complete sequence. The latest improvement screen did not select candidates using official OOS labels or mask; the public OOS results had nevertheless been viewed during earlier work.

The ordered FIT-only architecture, blend, and distillation screens are reported in [ADVANCE.md](ADVANCE.md). The standalone architecture changes did not meet the +0.005 rule, but a single distilled GRU retained the three-GRU ensemble gain and passed promotion: 0.421802 mean FIT-CV versus 0.415425 for the single-model reference. After complete-FIT refitting, it scored 0.444501 on fixed DEV and 0.675613 on the already-viewed public OOS set. The user submitted `submission_v1.zip`; it completed in 22m16s and scored 0.648658200077 on the official leaderboard (rank 227 when reported).

## Stronger-teacher distillation (2026-09-24)

The frozen v1 student was blended 50:50 with each teacher component on FIT-CV. No deployable two-GRU blend met the predeclared +0.002 gain and mask-robustness gate; the best was student plus normalized teacher at +0.001617. A diagnostic four-GRU student/teacher blend gained +0.003510, motivating a single-model compression test with the teacher-loss weight raised from 0.50 to 0.75. This was the only new distillation weight tested.

The alpha=0.75 student improved all three FIT-CV folds. Mean Global WP rose from 0.421802 to 0.423252 (+0.001449); early, late, and deterministic random-20% means changed by +0.002486, +0.000405, and +0.001370. After a frozen refit on the same 9,546 FIT sequences, it scored 0.445220 on fixed DEV (+0.000719 versus v1) and 0.675752 on the already-viewed public OOS (+0.000140). The NumPy callback matched PyTorch within 8.35e-7 on a full sequence, produced a 229,895-byte ZIP, and projected to 13.76 local minutes for 1,970 sequences. This v2 candidate has not been submitted, so no official-score improvement is claimed.

## Sequential improvement screen (2026-09-19)

The ordered FIT-only experiments and frozen full-FIT/DEV/public-OOS comparison are in [IMPROVE.md](IMPROVE.md). On the current Global WP metric, the supplied baseline scored 0.617052; the updated Ridge, MLP, and best single GRU scored 0.627167, 0.629395, and 0.672096, respectively. The fixed three-GRU average scored 0.673970. These are local public-validation replays, not official submissions or untouched hidden-test results.

## Metric audit and train-only global rescore (2026-09-17)

The current [competition data overview](https://wundernn.io/connectome/docs/data_overview) says the official metric is **Global Weighted Pearson over all selected rows**, while this local starter pack's `utils.BlockAccumulator` reports the equal mean of sequence-wise correlations. These are different quantities. `research/global_wp.py` implements streaming Global WP and checks it against the direct pooled calculation. The local kit's scorer remains unchanged so earlier results are reproducible. The newer official ZIP uses `GlobalAccumulator`; its three Parquet files and baseline ONNX have the same sizes and ZIP CRC32 values as the local files. A complete local baseline replay under Global WP reproduced the current published 0.617052 score (details below).

Both frozen model generations were rescored on the **same complete 1,061-sequence train-development holdout**, using all 21,114,961 required rows. No `valid.parquet` access, retraining, or new model selection occurred in this pass.

| Model | Pilot Global WP | Expanded Global WP | Pilot local block WP | Expanded local block WP |
|---|---:|---:|---:|---:|
| Ridge | 0.325818 | 0.346408 | 0.234007 | 0.256050 |
| MLP | 0.318087 | 0.363782 | 0.230002 | 0.272995 |
| GRU | 0.330414 | 0.412326 | 0.223079 | 0.331816 |

The recomputed local block scores match all six previously saved values exactly. With fixed train-only 20% random row masks, expanded Global WP was 0.346649 (ridge), 0.364092 (MLP), and 0.412445 (GRU); first-half versus second-half step results are saved in ignored `research/cache/train_dev_global_rescore.json`. These are sensitivity checks, not the hidden official scoring mask, a chronological backtest, or an OOS result.

## Fixed three-fold group screen (2026-09-17)

The [predeclared screen](SCALE.md) used 1,536 fitting-pool sequences: three disjoint folds with 1,024 fitting and 512 validation sequences each. Models were refitted from scratch per fold; the locked 1,061-sequence development holdout and official `valid.parquet` were excluded. Values below are the mean of the three fold **Global WP** scores using all required rows; `±` is the across-fold standard deviation, not a confidence interval.

| Family | Existing rule | Tested change | Change minus existing |
|---|---:|---:|---:|
| Ridge | 0.345604 ± 0.013477 | Causal one-step differences: 0.352280 ± 0.013063 | +0.006675 |
| MLP | 0.326287 ± 0.012136 | Every 8th row: 0.326585 ± 0.011924 | +0.000298 |
| MLP | 0.326287 ± 0.012136 | Fit-fold-only standardization: 0.343203 ± 0.008985 | +0.016916 |
| GRU, epoch 3 | 0.397298 ± 0.009323 | Hidden width 96: 0.403263 ± 0.010557 | +0.005965 |

The 64-wide GRU's epoch-1 and epoch-2 means were 0.369726 and 0.389152, so this screen did not support early stopping before epoch 3. Ridge differences, MLP standardization, and GRU width 96 improved all three paired folds and met the predeclared +0.005 mean-gain rule; the MLP sampling change did not. Fixed train-only 20% random-mask means closely tracked all-row means (details in ignored `research/cache/group_cv/summary.json`). These are exploratory, non-chronological, compute-bounded results: only 1,536 of 9,546 eligible fitting sequences entered the screen. The three qualifying variants were then refitted on the complete fitting pool; no claim of a final model winner follows.

## Selected variants: complete FIT refit and fixed DEV check (2026-09-18)

Ridge with one-step causal differences, MLP with FIT-only feature standardization, and 96-wide GRU were each refitted on all 9,546 locked fitting sequences. The table compares them with the previously frozen expanded models on the **same 1,061 train-development sequences and 21,114,961 required rows**. Scores are Global WP using all required rows; early and late use steps 99–9,999 and 10,000–19,999, respectively. Random 20% is a fixed per-sequence train-only row mask, not the official hidden test mask.

| Family | Previous Global WP | Selected Global WP | Change | Early | Late | Random 20% | Selected local block WP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge + causal difference | 0.346408 | 0.353303 | +0.006896 | 0.350788 | 0.355692 | 0.353517 | 0.263967 |
| MLP + FIT standardization | 0.363782 | 0.369475 | +0.005693 | 0.365838 | 0.372996 | 0.369714 | 0.278569 |
| GRU hidden width 96 | 0.412326 | 0.414437 | +0.002111 | 0.411847 | 0.416756 | 0.414583 | 0.332765 |

All three changes remain positive on this fixed DEV check, though the GRU gain (+0.002111) is smaller than its group-CV mean gain (+0.005965). Per-sequence local block WP 10th/50th/90th percentiles were 0.076761/0.246676/0.484508 (ridge), 0.113684/0.265461/0.491481 (MLP), and 0.138811/0.326696/0.536803 (GRU). These are **not** chronological or OOS scores. The DEV set was inspected during earlier experiments; this pass confirms selected changes but cannot serve as an untouched final test. `valid.parquet` was not read during metric audit, CV, refitting, packaging, or this scoring pass. The supplied baseline's masked OOS score is not directly comparable to these unmasked DEV results.

Each candidate was exported as a ZIP containing `solution.py`, `numpy_gru.py`, and `model.npz`. The ZIP was extracted and its no-argument callback smoke-tested for warm-up and sequence reset under both local Python 3.13 and Python 3.11. The row-by-row predictions were compared with the source batch model over one complete 20,000-row FIT sequence. Timings below are single-sequence callback measurements on this Mac, excluding data I/O, then simply multiplied by the currently documented 1,970 test sequences; they are **not** a Linux 1-vCPU benchmark or a time-limit guarantee.

| ZIP | Bytes | Seconds / 20,000 rows | Projected local minutes / 1,970 sequences | Max batch difference |
|---|---:|---:|---:|---:|
| `ridge_lag.zip` | 4,297 | 0.110 | 3.62 | 2.38×10⁻⁶ |
| `mlp_norm.zip` | 17,408 | 0.186 | 6.11 | 5.36×10⁻⁷ |
| `gru96.zip` | 228,783 | 0.488 | 16.03 | 1.18×10⁻⁵ |

All three extracted ZIPs also completed a full 20,000-row train-feature callback under local Python 3.11 with finite outputs (0.103, 0.177, and 0.447 seconds respectively on this Mac). This still does not replicate the official Linux resource limit.

For GRU, four additional 20,000-row train-file sequences showed at most 2.23×10⁻⁵ prediction difference from batched PyTorch due to float32 recurrent round-off; the inspected sequence-wise WP differences were at most 2.4×10⁻⁶. The parity gate is 5×10⁻⁵ for GRU and 1×10⁻⁵ for the nonrecurrent models. All ZIPs are below the documented 20 MB limit, but have not been uploaded or run in the competition container. Full measurements and local ZIPs remain in ignored `research/cache/selected/fixed_dev_scores.json` and `research/cache/packages/`.

## Expanded candidates: train data only

All three pilot model families were retrained using **all 9,546 locked fitting sequences** from `train.parquet`; evaluation used **all 1,061 fixed internal development sequences** from that same file. The previous pilot artifacts were trained on 128 fitting sequences and scored on this *same* complete development set. Each score below uses 21,114,961 required rows, with all 1,061 blocks eligible. The official `valid.parquet` OOS file was not read for this experiment.

| Model | Pilot mean WP | Expanded t0 WP | Expanded t1 WP | Expanded mean WP | Change vs pilot |
|---|---:|---:|---:|---:|---:|
| Weighted ridge | 0.234007 | 0.250642 | 0.261459 | 0.256050 | +0.022043 |
| 32-wide MLP | 0.230002 | 0.269348 | 0.276642 | 0.272995 | +0.042993 |
| 64-wide stateful GRU | 0.223079 | 0.332513 | 0.331118 | 0.331816 | +0.108736 |

The [predeclared training protocol](SCALE.md) used every 32nd required fitting row for ridge and MLP, and all fitting rows for the GRU. The MLP ran two epochs (mean weighted training loss 1.070028 → 1.043594); the GRU ran three (1.010298 → 0.982083 → 0.974267). These losses are training diagnostics, not the competition metric. All three internal development scores improved relative to their small-data pilots; this is not an official OOS result or a final model choice. The supplied baseline's legacy block score of 0.589595 was measured on the *masked official OOS* set and is not directly comparable to the unmasked train-development scores above. Its training provenance is unknown, so a score on the train holdout would not establish a fair baseline comparison either.

The ignored local artifacts are `research/cache/scaled_ridge.npy`, `scaled_mlp.pt`, `scaled_gru.pt`, and `scaled_dev_scores.json`. SHA-256 for the three model artifacts, respectively: `e3b68d19d5a09bc05862e3196956e5700c46b40d66555bb2a92ea7f311bff05b`, `8495860bae7bad52fcb0a4f31c38438ca3ad6415b52ef3e5f5aa07a08c3b482f`, `d18f612179c1ffcaf2e2fac4d261bec26a4932486544ba9fd8ceda4694889005`.

## Official masked OOS re-score under Global WP (2026-09-18)

All artifacts below were frozen before this pass. They were scored on the **same complete 1,873-sequence `valid.parquet`**, with 3,528,273 rows selected by `need_prediction AND is_scored`. Models were not refitted, reweighted, or selected using these OOS scores. The baseline was run row by row; candidate models were scored in batches (with prior row-wise parity checks for pilot and selected candidates). The current [official quick start](https://wundernn.io/connectome/docs/quick_start) reports baseline Global WP 0.617052; the independent local replay returned **0.617052289**.

| Frozen model | Global t0 | Global t1 | Global mean WP | Versus baseline |
|---|---:|---:|---:|---:|
| Supplied ONNX baseline | 0.587913 | 0.646192 | 0.617052 | — |
| Pilot ridge (128 FIT sequences) | 0.556983 | 0.589684 | 0.573333 | -0.043719 |
| Pilot MLP (128 FIT sequences) | 0.547177 | 0.588623 | 0.567900 | -0.049152 |
| Pilot GRU (128 FIT sequences) | 0.566483 | 0.588035 | 0.577259 | -0.039793 |
| Expanded ridge (9,546 FIT sequences) | 0.583693 | 0.593442 | 0.588568 | -0.028485 |
| Expanded MLP (9,546 FIT sequences) | 0.596904 | 0.612869 | 0.604886 | -0.012166 |
| Expanded GRU (9,546 FIT sequences) | 0.654867 | 0.663076 | 0.658972 | +0.041919 |
| Selected ridge + causal differences | 0.592792 | 0.603352 | 0.598072 | -0.018980 |
| Selected MLP + FIT-only standardization | 0.599100 | 0.618953 | 0.609026 | -0.008026 |
| Selected GRU width 96 | 0.657386 | 0.665609 | 0.661498 | +0.044446 |

Relative to the unchanged expanded model in its own family, the selected variant's OOS Global WP changed by +0.009504 (ridge), +0.004140 (MLP), and +0.002526 (GRU). These are public, masked validation measurements, **not** a hidden-test estimate or a new hyperparameter-selection signal. The official OOS set had already been viewed during the pilot; it is not an untouched holdout. The latest selected GRU's source batch predictions were checked against its NumPy submission callback over complete 20,000-row train sequences (maximum difference observed 2.23×10⁻⁵), so the submission ZIP has not itself been run over the entire OOS file. Scores and selected-row counts are saved separately in ignored `research/cache/oos_global_baseline.json` and `research/cache/oos_global_candidates.json`.

## Legacy block-score OOS baseline reproduction

The supplied ONNX GRU scored **0.589594511 legacy block WP** (`t0` 0.571491710, `t1` 0.607697312) on the complete masked `valid.parquet`: 1,873 blocks seen, 1,872 eligible, 1 excluded, 3,528,273 selected rows, and 3,527,577 selected rows in eligible blocks. The old kit's row-by-row scorer required 37,274,573 predictions and took 849.87 seconds locally. This matched the then-published 0.589595 score after rounding; the current Global WP baseline is 0.617052289 above. It is a fixed historical reference, not a model-selection result.

## Legacy block-score OOS comparison of frozen pilots

The three unchanged pilot artifacts below were scored on the same complete `valid.parquet` and official `is_scored` mask as the supplied baseline. Every row after warm-up had a finite prediction. Each run saw 1,873 blocks, of which the same 1,872 were eligible; 3,528,273 rows were selected in total and 3,527,577 were in eligible blocks.

| Model | t0 WP | t1 WP | Mean WP |
|---|---:|---:|---:|
| Supplied ONNX baseline | 0.571492 | 0.607697 | 0.589595 |
| Weighted ridge | 0.488268 | 0.521114 | 0.504691 |
| MLP | 0.471661 | 0.521354 | 0.496508 |
| GRU | 0.512438 | 0.524030 | 0.518234 |

These candidate artifacts were trained on only 128 sequences and were not retrained after the OOS request. Batched local scoring was checked against row-by-row inference: over one complete 20,000-row sequence, the GRU maximum prediction difference was `7.2e-7`; the MLP difference over the first 1,024 rows was `4.8e-7`. The three-model OOS pass took 251.20 seconds locally; its batched runtime is not a submission callback benchmark. The OOS scores are now visible and cannot serve as an untouched holdout for future tuning.

## Complete internal development holdout

The frozen pilot models were evaluated on **all 1,061 held-out train sequences** after fitting only 128 sequences from the other 9,546 groups. Every block was eligible. Each model produced 21,114,961 scored predictions, with all required steps selected because train has no scoring mask.

| Model | t0 WP | t1 WP | Mean WP |
|---|---:|---:|---:|
| Current-row weighted ridge | 0.220309 | 0.247706 | **0.234007** |
| 32-wide current-row MLP | 0.213734 | 0.246271 | 0.230002 |
| 64-wide stateful GRU | 0.214686 | 0.231473 | 0.223079 |

The 32-sequence pilot and the full development holdout produced the same ordering, but the differences are small and these are models trained on only 128 sequences. No final model choice follows from this screen.

## Pilot on 32 fixed internal development sequences

All 32 pilot development sequences were eligible. The score uses all 19,901 required rows per sequence (636,832 rows total), because `train.parquet` has no `is_scored` column. Thus these numbers are directly comparable **to each other**, but not to the masked official OOS score.

| Model | t0 WP | t1 WP | Mean WP |
|---|---:|---:|---:|
| Current-row weighted ridge | 0.244990 | 0.227101 | 0.236046 |
| 32-wide current-row MLP | 0.245122 | 0.225600 | 0.235361 |
| 64-wide stateful GRU | 0.231202 | 0.205762 | 0.218482 |

The ridge and MLP saw one of every 32 required rows from 128 fitting sequences. The GRU saw all required rows from those 128 sequences for three epochs, carrying hidden state between 512-row chunks and resetting between sequences. All three used clipped targets and absolute-target-weighted squared error; only development scores determined the comparison. The weighted ridge needs float64 normal-equation accumulation because the book columns are correlated; float32 gave unstable coefficients and an incorrect 0.109020 score. The initial random-window GRU reached 0.075069 WP; carrying state through full sequences raised it to 0.218482. The supplied ONNX baseline has no released training provenance and may have seen this train holdout, so its train-holdout score is excluded from candidate comparison.

## Reproduce

Use a Python environment with the root `requirements.txt` plus PyTorch. The pilot used Python 3.13.2, NumPy 2.2.6, PyArrow 19.0.1, ONNX Runtime 1.23.2, and PyTorch 2.13.0. From the starter-pack root:

```bash
python research/pilot.py prepare
python research/pilot.py train
python research/pilot.py score
python research/pilot.py score_all
python research/pilot.py score_oos
python research/pilot.py selfcheck
```

`prepare` verifies the train group order before writing ignored NumPy caches; `train` writes model weights and predictions; `score` checks the 32-sequence pilot, `score_all` streams all 1,061 held-out train groups, and `score_oos` evaluates the frozen artifacts on the official validation mask with the **old kit's** `BlockAccumulator`. The historical baseline block score was produced with:

```bash
python baseline/solution.py --validation datasets/valid.parquet
```

For the current Global WP rule, replay all frozen models and the baseline without modifying the older score files:

```bash
python research/oos_global.py candidates
python research/oos_global.py baseline
```

The baseline ONNX SHA-256 is `321ec67b93f3c5dde2c0901ca4532072dad5f65753c2e6519e93ea3555f6a6fd`.

Frozen pilot artifacts used for the user-requested OOS comparison (SHA-256): `ridge_weights.npy` `eabe5a8598ba6380092f91482e23ee0da40248d425f23e8ac0b06bd2a40874ce`, `mlp.pt` `1794fc403194ab174caffe46ffc03dd154ae32b4f83ccc1b195fa7239f29756d`, `gru.pt` `ea943951b91a4a0511a8ddba820feb43f694c1d1be2748cc28623f0315230d75`.

## Phase 4 status

The full fitting-split experiment, three train-only variants, fixed DEV check, three inference ZIP checks, and Global WP OOS re-score are complete. No ZIP has been submitted. The local starter pack retains its old `BlockAccumulator` for reproducibility; the current official pack uses `GlobalAccumulator`, and the published baseline score is locally reproduced under the new rule. Keep `valid.parquet` out of fitting and tuning; both old pilot scores and the new re-scores have been viewed, so it is not an untouched final holdout. Linux-container runtime remains unverified. Real chronological walk-forward remains pending verified sequence order.

Selected full-fit artifact SHA-256: ridge `74f2fc2ce2580bf81535498d976ac7cb531cbbc696eb2288717a5cc0590bfce1`, MLP `916f8ee85749dd6af5a86406c1f2afaedade0689518d667bd05429679956d2c5`, GRU `ebc53014bff439531c11e1cc627f6256a287896edaa9f4bc1c2047e00dd40211`.
