# Results: official submissions and every tested variant

**Source and conventions.** All numbers are copied from `research_log/MASK.md`, which has the full context and the rule set before each test.
- "h0 / h1": held-out Global WP on the other validation half's `is_scored` rows.
- "Mean": the mean of h0 and h1.
- "DEV": the 451 unseen train-period sequences, scored on all rows. Only differences matter; levels cannot be compared with validation.
- "Pilot": the locked 128-FIT / 32-DEV train-only screen (GRU96 control 0.316064).

Earlier train-only stages (ridge/MLP/GRU, group-CV, first distillation) are in `research_log/RESULTS.md`, `IMPROVE.md` and `ADVANCE.md`.

## 1. Official public leaderboard

The public score uses the pooled metric. Runtime is the official wall time; the limit is 60 minutes.

**Member abbreviations:**

| Name | Model |
|---|---|
| ED | distilled GRU128 α0.75 |
| SD | gen-1 GRU128 student |
| C | gen-2 GRU128 student + low-LR continuation |
| S | strong-recipe GRU128 |
| B192 | 2-layer GRU192 |
| split | split encoder |
| twin | TwinGRU |
| twin-encoder | TwinEncGRU |

Package files are in `packages/`; member lists with exact checkpoints are in `packages/*.check.json`.

| Package | Members (ensemble weights) | Public score | Runtime |
|---|---|---:|---:|
| v1 | distilled GRU96 trained on FIT only (no validation FT) | 0.648658 | 22m16s |
| v10 | v1, GRU192, GRU128 α0.75, GRU192 α0.75; mask-weighted validation FT, SWA(e2, e3); padded to width 192 | 0.6582 | 56m07s |
| v12 | v10 with GRU128 α0.75 replaced by ED | 0.6583 | 56m01s |
| v15 | SD71 0.5; GRU128 α0.75 s61, ED, GRU192 1/6 each (width groups, 452 kMAC) | 0.6579 | 46m11s |
| v16 | SD71, SD72 0.3 each; ED, GRU192 0.2 each | 0.6586 | 45m32s |
| v17 | SD71, SD72 0.3 each; GRU192, split 0.2 each | 0.6586 | 54m26s |
| v18 | SD71–SD74, split (0.2 each) | 0.65870 | 51m01s |
| v24 | C75–C78, target-balanced split (0.2 each) | 0.65914 | 53m02s |
| v25 | C75, C76 0.4 each; split 0.2 | 0.65898 | 39m21s |
| v26 | C75, C76, GRU192, split (0.25 each) | 0.6589 | 54m37s |
| v28 | v24's members on the float16 C kernel | 0.6591 | 25m07s |
| v29 | C75–C78 0.1 each; S71 (h1 model), S72 (h0 model) 0.1 each; B192 (h0 model) 0.2; split 0.2 | 0.6593 | 45m37s |
| v30 | C75–C78 0.1 each; S71, S72 0.1 each; B192 0.2; WiSE split 0.2 | 0.659412 | 45m32s |
| v32 | C75–C78 0.075 each; twin s71 0.3; B192 0.2; WiSE split 0.2 | 0.659661 | 46m11s |
| v33 | C75, C76 0.1 each; twins s71, s72 0.2 each; B192 0.2; WiSE split 0.2 | 0.6598 | 46m37s |
| v34 | twins s71–s74 0.15 each; B192 0.2; WiSE split 0.2 | 0.6599 | 51m31s |
| v35 | twins s71–s74 0.2 each; WiSE split 0.2 | 0.6598 | 35m51s |
| v36 | twins s71–s76, equal | 0.6603 | 43m32s |
| **v37** | **twin-encoders s71–s75, equal** | **0.66045** | 43m37s |
| v38 | twin-encoders s71–s76, equal | 0.66049 | 53m32s |
| v39 | twin-encoder s71 distilled from the gen-3 teacher, its h0 and h1 half models (together they saw all validation), plus all-data twin-encoders s71–s74; equal | **0.66052** (best) | 51m02s |
| diag | v28 × one constant per sequence (metric test) | 0.6158 | 26m47s |

**Final selection (still open).** Before v39 the pair was v37 + v33. The current proposal is v39 + v37.
- v39 is the best on both the public board and unseen DEV. It has about 7.5 minutes to the zero stage, so it tolerates a final set about 10–14% larger than the public one.
- v37 is from the same family, with 16 minutes of margin; it tolerates about 30% more data.
- v33 has different members and the full runtime guard (reduce stage at 54 minutes, zeros at 58.5 minutes).
- v38 is +0.00004 over v37, within noise, and leaves only 6.5 minutes of margin.
- The rules do not state the size of the final dataset.

**Built but not submitted:** v2–v9, v11, v13, v14, v19–v23, v27, v31. Why each was built is in the log.

**Biggest steps:**

| From → to | Change | Public gain |
|---|---|---:|
| v1 → v10 | mask-weighted fine-tuning on validation | +0.0095 |
| v18 → v24 | gen-2 teacher + continued students + target-balanced split | +0.0004 |
| v30 → v32/v36 | twin structure | +0.0003 / +0.0009 |
| v36 → v37 | twin-encoder | +0.0001 |

The public board tracked the direction of the unseen-DEV check better than that of the in-period halves.

## 2. Main members and sets (local)

| Model (SWA as deployed) | h0 | h1 | DEV (all-data model) |
|---|---:|---:|---:|
| v1, frozen | — | — | mean of halves 0.675590 |
| v1 FT (u 0.06, SWA e2–e3) | 0.681944 | 0.679740 | — |
| split encoder FT, target-balanced | 0.682971 | 0.682128 | 0.4318 (WiSE 0.4333) |
| SD71 (gen-1 student) | 0.684212 | 0.683506 | — |
| G71 (gen-2 student) | 0.684436 | 0.683962 | — |
| C71 (gen-2 + continuation) | 0.685150 | 0.684065 | C75–C78: 0.4337–0.4360 |
| S71 strong recipe (e8) | 0.685808 | 0.682127 | S71 / S72: 0.4362 / 0.4344 |
| B192 2-layer GRU192 (e8) | 0.686391 | 0.682674 | 0.4358 |
| twin s71 (continuation SWA) | 0.686965 | 0.683943 | 0.439717 |
| twin s72 | 0.687470 | 0.683341 | 0.438290 |
| twin-encoder s71 | 0.687817 | 0.683945 | 0.440444 |
| twin-encoder s71, gen-3 teacher | 0.688490 | 0.683393 | — (half models: 0.441608 / 0.443416) |

All-data twins s71–s78 alone on DEV: 0.4383–0.4408. Twin-encoders s71–s76: 0.440444 / 0.440961 / 0.439797 / 0.441013 / 0.441765 / 0.441287.

**Unseen-DEV scores of the package sets:**

| v28 | v30 | v31 | v32 | v33 | v34 | v35 | v36 | v37 | v38 | v39 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.438161 | 0.439216 | 0.439961 | 0.440314 | 0.440521 | 0.441381 | 0.441625 | 0.442383 | 0.443487 | 0.443680 | **0.444477** |

Bootstrap tests (paired over sequences, 2,000 resamples):

| Comparison | DEV difference | 95% interval | P(≤ 0) |
|---|---:|---|---:|
| v37 − v36 | +0.0011 | [+0.00006, +0.0021] | 0.016 |
| v38 − v37 | +0.00019 | [−0.00013, +0.00051] | 0.11 |
| v39 − v38 | +0.00080 | [−0.00006, +0.00169] | 0.038 |
| v39 − v37 | +0.00099 | [+0.00020, +0.00176] | 0.008 |
| v39 − (same set with the gen-2 half models): effect of the gen-3 teacher | +0.00044 | [+0.00004, +0.00081] | 0.015 |

**Ensemble size on DEV (equal weights):**
- Twins ×4 / ×6 / ×8: 0.441941 / 0.442383 / 0.442421. Saturated at 6; pairwise correlation is about 0.994.
- Twin-encoders ×1–×6: 0.440444 / 0.442354 / 0.442606 / 0.443044 / 0.443487 / 0.443680.
- Adding plain twins to five twin-encoders lowers DEV: +1 → 0.443454, +2 → 0.443229.

**First FT ensembles** (mean of halves; all members FT u 0.06, SWA):

| Package | Members | Mean of halves |
|---|---|---:|
| v1 | frozen | 0.675590 |
| v3 | v1 + v2 | 0.681516 |
| v4 | v1 + v2 + t_norm | 0.682493 |
| v6 | v1, v2, GRU128, GRU192 | 0.683549 |
| v7 | v1, GRU128 α0.5, GRU192, GRU128 α0.75 | 0.683816 |
| v10 | v1, GRU192, GRU128 α0.75, GRU192 α0.75 | 0.684038 |

## 3. Variants that scored differently

### 3a. Validation fine-tuning (stage 3, `maskft.py`), mostly on v1

| Variant | Result | Decision |
|---|---|---|
| Unscored-row weight u = 1 (uniform control) vs u = 0.06, e3 | 0.677191 vs 0.680127 (frozen 0.675590) | **mask weighting adopted** |
| u = 0.02 | single +0.000144; lower in both deployable sets | not adopted |
| SWA(e2, e3) vs e3 alone | 0.680842 vs 0.680127 | **adopted** |
| lr 1e-4 (e3 / e4) | 0.680274 / 0.680260 | no gain |
| 2 epochs at 2e-4, then 5e-5 to e6, various SWA windows | 0.680681–0.680773 | keep 3 epochs |
| Train sequences mixed in with mask-net weights (`maskmix.py`) | 0.679914 (h0 e3) / 0.678054 (h1 e2) | rejected |
| 1,500 most validation-like train sequences mixed in | peak 0.679523 (e1) | rejected |
| Target-balanced loss (split encoder) | +0.000227, both halves | **adopted** for split |
| Chunk 1024 instead of 512 | +0.000019 | no |
| Frozen recurrent weights (GRU192) | single SWA 0.681299 vs 0.681119; nothing in ensemble | no |
| L2-SP λ 0.01 | e3 0.679579 | rejected |
| Weighted-Pearson loss term (per chunk) | 0.680837 vs 0.680842 | rejected |
| Exact pooled Global-WP gradient (1, 1) / (3, 0.2) | +0.00006 / −0.00026 | rejected |
| Next-event auxiliary loss 0.1 | below control every epoch | rejected |
| Causal Conv1D residual; event-reset interval MLP residual | below control | rejected |
| DropConnect 0.15 / 0.3, SAM 0.05, β2 0.98, input noise 0.05, 3-seed FT soup | all within ±0.0012, none +0.0005 on both halves | none |
| AFML uniqueness weight (block length^−P), P 0.5 / 1 | +0.0001 / −0.0022 | no |
| WiSE-FT a = 0.9 (split encoder) | in period within ±0.00005; train-period proxy +0.0014 | **adopted as hedge** (v27, v30–v35) |
| Per-tick-group recalibration; 0.657-group specialist FT | +0.0004; lower | no |
| Distribution head mixed 25% into the split member | +0.000116 (v20) | not submitted |

### 3b. Students (stage 5, `ensdistill.py`)

| Variant | Result | Decision |
|---|---|---|
| From-scratch GRU128 student (gen-1, α 0.75, 6,000 pool) vs best FT single | SD71 0.684212 / 0.683506 vs split 0.682895 / 0.681752 | **adopted** |
| Gen-2 teacher vs gen-1 (same seed) | +0.00034 (s71), −0.00019 (s72); sets about +0.00015 | adopted |
| Low-LR continuation (1e-4, 1e-4, 3e-5), SWA of 3 epochs | +0.00041 (s71), +0.00078 (s72); set +0.00038, both halves | **adopted** |
| Second continuation (5e-5, 3e-5, 1e-5) | +0.00003 | no |
| Weight EMA 0.9998 | single +0.0004 (h0 only); in a set +0.00004 | no |
| Validation repeated 3× per epoch | −0.0002 to +0.00005 | stopped |
| Full train pool (10,607 instead of 6,000) | no change | rejected |
| GRU256 student | e5 0.684456 / 0.681232 vs 0.684673 / 0.683096 (overfits) | rejected |
| 2-layer GRU128 | SWA +0.0003, 2× inference cost | not packaged |
| minGRU 3×128 | SWA 0.684287 vs 0.684436; corr 0.993 with G71 | no |
| LSTM128 | e1 −0.0006 | stopped |
| BPTT 2,048 rows, batch 8 | −0.0002 | no |
| Label-only (α 0) | 0.676943 vs 0.684199 | rejected |
| Strong recipe (AdamW 1e-4, OneCycle peak 1.5e-3, clip 1, 8 epochs), GRU128 | s71 h0 +0.00066, h1 −0.0019; pair 0.684878 vs 0.684890 | members in v29/v30 only |
| Strong recipe 2-layer GRU192, dropout 0.1 (B192) | mean 0.684532 vs C71 0.684607 | member in v29–v34 |
| Chrono init + random BPTT offset + α 1.0 on train rows | mean +0.0001, 2 of 4 positive | no |
| PLE input calibration (`--cell ple`) | e5 +0.00002 / −0.00032 | no |
| Causal EWMA z-scores, half-life 500 (`--causal 500`) | e5 −0.00082 / −0.00103 | rejected |
| **Twin GRU** (`--cell twin`, i0 96 / i1 48 / joint 96) | e5 +0.00151 / +0.00006; with continuation 0.686965 / 0.683943; best DEV member | **adopted** (v32–v36) |
| Wider twin (128 / 64 / 128) | h1 about −0.002 (overfits) | rejected |
| Narrow twin (64 / 32 / 64) | in period like a seed; DEV 0.437453 vs 0.439718 | rejected |
| Siamese twin (shared stream weights) | SWA 0.683135 (h1); DEV 0.438801 vs 0.439718 | rejected |
| Triple stream (i0 prices / i0 volumes / i1) | h1 continuation 0.683235 / 0.683094 / 0.682965 vs twin 0.683989 / 0.6838 / 0.683678; DEV 0.439494 vs 0.439718 | rejected |
| Continuation teacher weight 0.25 / 0 (instead of 0.75) | h1 e3 0.678032 / 0.673319 vs 0.683678 | rejected |
| **Twin + per-group encoders** (`--cell twinenc`) | h0 0.687817 vs twin 0.686965 (+0.00085); h1 tie; DEV +0.0026 (h1 model), +0.00035 (h0 model) | **adopted** (v37/v38) |
| Twin-encoder + side encoders (`--cell twinenc2`) | in period +0.0002 / −0.0005; DEV −0.0009 / +0.0010 | rejected |
| Gen-3 teacher (0.5 · gen-2 + 0.5 · twin family) | in period +0.00067 / −0.00055; DEV +0.00047 / +0.00109. Its two half models inside v39: +0.00044 on DEV vs the gen-2 half models; v39 is the best public score | all-data gen-3 students not trained (h1 fails the −0.0003 tolerance); half models used in v39 |

### 3c. Other families and features

Pilot screens need +0.005 to scale up; validation screens need +0.002 on both halves, or a pre-set rule.

| Variant | Result | Decision |
|---|---|---|
| Event-time GRU (steps only at i0 price events) | FT mean 0.591786; 5% blend −0.00037 | rejected |
| Dedicated spread model S (`sgru.py`), S swap or stacking | better corr(S), set lower (0.684126 vs 0.684352) | rejected |
| Price-state → S lookup (rounds 6 and 24) | validation corr 0.22; any weight lowers WP | rejected |
| Dedicated D = t0 − t1 model | group-CV +0.0025 to +0.0040 alone; −0.0016 in the strong ensemble | not deployed |
| Vanilla RNN128 / 2-layer GRU64 / target-specific dual GRU64 (pilot) | −0.014011 / +0.001931 / −0.001795 | none passes |
| Selective SSM (Mamba-like) / local causal attention (pilot) | −0.016803 / −0.012156 | rejected |
| Multi-scale window mixer | pilot +0.011; 1,024/512 fold 0.386436 vs 0.405980 | rejected |
| Next-event pretraining / hard event bottleneck / contrastive pretraining (pilot) | −0.001643 / −0.003548 / −0.002277 | rejected |
| Warm-up (rows 0–98) context | pilot +0.006924, FIT folds +0.0026; adapter on C71 −0.000037 | no |
| Uniqueness weighting in train-only training | pilot +0.014; group-CV mean −0.002132 | rejected |
| Smoothed targets (forward means 10 / 50 / 200 rows) | 0.6746 / 0.6619 / 0.6127 vs 0.6775 | rejected |
| Delayed target estimate (ridge reconstruction) as input / blend | blends lower at every weight | rejected |
| Rule-based recall of a revisited book state | DEV +0.00059 [−0.0006, +0.0020]; h0 −0.00023, h1 −0.00010 | rejected |
| LightGBM residual on current-row, history or decoded-track features | all lower on held-out halves | rejected |
| Per-sequence test-time calibration (online) | far below base | rejected |
| Bid/ask mirror augmentation | no mirror preserves the signal | rejected |
| Queue/OFI reconstruction from volumes | not identifiable (nonlinear, non-sparse encoding) | closed |
| fp16 weights / low-rank SVD of inputs | fp16 identical to 6 decimals / rank 64 −0.0005 | fp16 kernel adopted |
| Fitted ensemble weights vs equal | 0.684018 vs 0.684038 | equal weights |

### 3d. Data facts that shaped the design (round 3–4, 9)

- About 8–9% of required validation rows are scored. Scored rows arrive in bursts and look like a clustered random process modulated by activity; the mask net reaches AUC about 0.905.
- Targets are tick-quantized and constant between i0 price-block changes; t0 = F − ask, t1 = bid − F, with a persistent future price F.
- There are four tick groups. The 0.657 group holds about 49% of validation scored rows and scores far lower (v1 0.420 vs 0.72–0.79).
- Validation is a different period from train: AUC 0.976 for telling train from validation. That is why validation fine-tuning helps and train mixing hurts.
