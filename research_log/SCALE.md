# Full fitting-split experiment (locked before training)

Only `datasets/train.parquet` is read. Fit groups are the 9,546 row groups whose zero-based index is not divisible by 10; development groups are the other 1,061. The three candidates use the same split and the pilot architectures. No `valid.parquet` labels, mask, or scores enter training or model selection.

| Candidate | Fitting input | Fixed training rule |
|---|---|---|
| Weighted ridge | Every 32nd required row from every fit group | Float64 weighted sufficient statistics; regularization scaled from the pilot by fitting-group count |
| MLP (112→32→2) | Same sampled rows from every fit group | Two passes, Adam 0.001, seed 7 |
| GRU (112→64→2) | Every row of every fit group | Three passes, 64 independent sequences per batch, 512-row truncated backpropagation with state carried across chunks, Adam 0.001, seed 7 |

All models use clipped `[-2, 2]` targets and absolute-target-weighted squared error. Development scoring uses every required step and the kit's block Weighted Pearson. The training script saves artifacts only after complete passes; `scale.py score` evaluates all three on the full fixed development holdout. The OOS file remains unused in this experiment.

From the starter-pack root, use a Python environment with the root requirements and PyTorch, then run:

```bash
python research/scale.py selfcheck
python research/scale.py ridge
python research/scale.py mlp
python research/scale.py gru
python research/scale.py score
```

The final internal-development measurements are recorded in [RESULTS.md](RESULTS.md). The score command writes `research/cache/scaled_dev_scores.json`; training artifacts remain under the ignored `research/cache/` directory.

## Predeclared train-only comparison after the metric audit (2026-09-17)

The current competition web documentation describes **Global** Weighted Pearson, unlike the bundled block-average scorer. Before any more model selection, rescore frozen pilot and expanded artifacts on the same 1,061 train-development sequences with both metrics. The expanded models and the earlier scores remain frozen; no official `valid.parquet` access is needed.

For a compute-bounded screen, select 1,536 groups evenly from the locked 9,546 fitting groups, shuffle this list with NumPy seed 20260917, then make three disjoint 512-group folds. Each fold fits on 1,024 groups and scores on 512, all from the original fitting pool. This is **group cross-validation, not chronological walk-forward**. The fixed 1,061-group development holdout stays outside this screen. Rank experiments by the mean of the three fold Global WP scores, while reporting fold variation and fixed train-only mask sensitivity.

Compare the existing architecture/training baseline with one-variable screens: ridge plus all-feature one-step causal differences; MLP with every 8th rather than every 32nd required fitting row; MLP with fit-fold-only feature standardization (sampling unchanged); and GRU with 96 rather than 64 hidden units. For GRU also score the saved epoch 1/2/3 checkpoints to test early stopping. Prefer a change over its simpler baseline only if mean fold Global WP improves by at least 0.005 and at least two of three folds improve; otherwise retain the baseline. Do not treat any improvement as a time-forward result. Only after this screen should a chosen variant be refit on all 9,546 groups and scored once on the fixed development holdout.

The screen selected causal differences for ridge, fitting-fold-only standardization for MLP, and hidden width 96 for GRU. The MLP every-eighth-row test missed the threshold. On a fresh cache, reproduce the train-only screen and selected full-fit checks with:

```bash
python research/global_wp.py
python research/group_cv.py selfcheck
python research/group_cv.py run
python research/group_cv.py summary
python research/refit_selected.py ridge_lag
python research/refit_selected.py mlp_norm
python research/refit_selected.py gru96
python research/refit_selected.py score
OPENBLAS_NUM_THREADS=1 python research/package_candidates.py all
```

The three ZIPs under ignored `research/cache/packages/` are inference-only candidates, not submitted entries. Package checks use train features only. `refit_selected.py score` evaluates all required rows on the same locked train-development groups, including fixed early/late and random-20%-row sensitivity checks; it does not read official validation. Since that development holdout has been inspected in prior experiments, its new scores are confirmation diagnostics, not an untouched test. Chronological walk-forward remains unavailable without verified sequence order.
