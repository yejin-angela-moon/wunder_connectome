# Alpha Connectome: training pipeline, data handling and results

This folder holds the whole pipeline behind the leaderboard entries v1–v39:
- training scripts,
- the exact row sampling and mask weighting,
- the loss functions,
- the train/validation splits,
- run configs and checkpoints,
- every model variant we tested, with scores.

The best public score is **v39 0.66052** (v38 0.66049, v37 0.66045; the three are within leaderboard noise). The final selection is open: the pair was **v37 + v33** before v39, and **v39 + v37** is the current proposal (see RESULTS.md).

- `RESULTS.md`: official scores of every submission, plus local scores of the variants (adopted and rejected).
- `research_log/MASK.md`: the full experiment log (rounds 1–25) with every number. The other files in `research_log/` cover the earlier train-only stages.

## Folder map

| Path | Contents |
|---|---|
| `code/` | Every research script (`*.py`) and the inference kernel `fast_kernel.c`. The core pipeline files are listed in section 3. |
| `configs/cache/...` | Every JSON a run wrote: full argument list plus held-out score per epoch. Same layout as `research/cache/`. Examples: `configs/cache/ensdistill/<tag>.json`, `configs/cache/maskft/<tag>.json`. |
| `logs/` | Training and evaluation logs, with per-epoch loss and held-out WP. |
| `checkpoints/cache/...` | 316 PyTorch state dicts, same layout as `research/cache/`: base models, the teacher chain, every deployed member, and the final weights of each tested variant. Details in section 6. |
| `packages/` | Every officially scored submission ZIP, plus the packager's check report (`*.check.json`: members, weights, parity, local timing). |
| `research_log/` | Experiment logs: `MASK.md` (main), `SPLIT.md`, `SCALE.md`, `IMPROVE.md`, `ADVANCE.md`, `DIRECTIONS.md`, `RESULTS.md` (early stages). |

## Setup

1. Unpack the competition starter pack, with `datasets/train.parquet`, `valid.parquet` and its `utils.py`. The scripts import `utils.py` from the starter-pack root. Both the original `utils.py` (`BlockAccumulator`) and the newer official one (`GlobalAccumulator`) work; with the newer one, only the two legacy block scorers (`scale.score_dev`, `refit_selected.score`) are unavailable.
2. Copy `code/*` into `<starter-pack>/research/`.
3. Copy `checkpoints/cache/` into `<starter-pack>/research/cache/`. The scripts find files by fixed relative names (`pilot.CACHE = research/cache`, `pilot.ROOT = starter-pack root`).
4. Use Python 3.11 with torch 2.14, numpy 2.2.6, pyarrow 19.0.1 and scipy 1.17.1. Those are the versions used. Training ran on an Apple-silicon laptop: `--device mps`, or `--device cpu --threads 6`.

Run commands from the starter-pack root, as `python research/<script>.py ...`. Paths passed on the command line (`--init`, package members) are relative to the current directory.

## 1. Data, splits and the metric

**Data.**
- `train.parquet`: 10,607 sequences (one row group each) of 20,000 rows. Each row has 112 features and 2 targets (t0, t1). Train has no `is_scored` column.
- `valid.parquet`: 1,873 sequences, which also carry `is_scored`.
- Rows 0–98 are warm-up (`WARMUP = 99`). They are never scored and get zero loss weight everywhere.

**Locked train split** (`scale.py`):
- `FIT` = train groups with index % 10 ≠ 0 (9,546 sequences).
- `DEV` = index % 10 = 0 (1,061 sequences).
- The base GRUs and the first distillation were trained on FIT only.

**Validation halves** (`maskmix.split`, `maskft.py`):

| Split | Fits on | Scored on |
|---|---|---|
| `h0` | even validation groups (937) | the odd groups' `is_scored` rows |
| `h1` | odd groups | even groups |
| `all` | all 1,873 sequences | nothing held out (deployment refit) |

- A change counts only if it helps both halves, since several changes looked good on one half only.
- Every model that is deployed (`*_all_*`) is a separate run on all validation, with the recipe chosen on h0/h1.

**Student train pool** (`ensdistill.teacher_cache`):
- 6,000 train sequences: the base 2,000 from `np.random.default_rng(123).choice(10607, 2000, replace=False)`, plus 4,000 from `np.random.default_rng(124).choice(<the other 8,607>, 4000, replace=False)`.
- The pool is drawn from all of train, so it overlaps DEV.

**Unseen DEV** (`dev_check.py`):
- DEV minus the 6,000-sequence pool (and minus `FIT`) leaves 451 train-period sequences that no student has seen.
- They are scored on all rows ≥ 99, because train has no mask. DEV scores (about 0.44) therefore cannot be compared in level with validation scored-row scores (about 0.68); only differences between models are meaningful.
- DEV is the out-of-period check. From v30 to v38, the public leaderboard followed its direction more closely than the in-period validation halves (at about 1/10 to 2/3 of its size).

**Metric** (`global_wp.py`):
- Pooled |y|-weighted Pearson ("Global WP") over all selected rows of all sequences, computed per target and then averaged over t0 and t1.
- y and predictions are clipped to [−2, 2]; the weights are |y| after clipping.
- The public leaderboard uses this pooled form, not the per-sequence mean described in the starter pack's METRIC.md. A diagnostic submission (`packages/submission_diag_metric.zip`) proved it: it multiplied each sequence's predictions by its own constant and scored 0.6158 against 0.6591. That matches the local pooled drop and contradicts the unchanged per-sequence prediction (`research_log/MASK.md`, round 16).

## 2. Inputs

**Standardizer.** `cache/improve/full_fit/gru96_norm5ep/gru_standardizer.npz` holds mean and std computed on FIT, using every 32nd row from row 99 on.

**Model input.** Every model from v1 on reads `clip((raw − mean) / std, −10, 10)`, which is 112 columns. There are three exceptions:
- `t_s17` and `t_s37` read raw inputs.
- `gru192ev_a0.5_s59` also reads the 46 causal event-time features from `features.py`.
- The split encoder reads the standardized row through its own encoders.

**Column layout:**

| Columns | Contents |
|---|---|
| 0–51 | instrument i0: `i0_p0–p21` (book prices, 0–21), `i0_v0–v21` (book volumes, 22–43), `i0_dp0–dp3` (trade prices, 44–47), `i0_dv0–dv3` (trade volumes, 48–51) |
| 52–103 | instrument i1, same layout |
| 104–111 | a0–a7 |

**Order-book groups** used by the twin encoder (11 columns each). "Bid" and "ask" are our reading of the obfuscated halves (p0–p10 vs p11–p21, and likewise for volumes); the code uses only the column ranges.

| Group | i0 columns | i1 columns |
|---|---|---|
| bid prices | 0–10 | 52–62 |
| ask prices | 11–21 | 63–73 |
| bid volumes | 22–32 | 74–84 |
| ask volumes | 33–43 | 85–95 |

## 3. Pipeline, stage by stage

The core files and what each one does:

| File | Role |
|---|---|
| `pilot.py` | Constants (`ROOT`, `CACHE`, `N_ROWS`, `WARMUP`), `make_gru`, data checks |
| `scale.py` | Locked FIT/DEV split, `read_group` for train, GRU training loop |
| `improve.py` | Stage 0: full-FIT base GRU96s |
| `distill.py` | Stage 1: 3-GRU teacher → GRU96 student (v1, v2) |
| `distill_mps.py` | Stage 2: wider students of the same teacher; split encoder; `build()` (model from a state dict) |
| `maskft.py` | Stage 3: mask-weighted fine-tuning on validation |
| `maskmix.py` | Mask net, validation split, `evaluate` |
| `ensdistill.py` | Stages 4–5: teacher caches (gen 1/2/3), students, all student architectures |
| `package_ft.py` | Stage 6: export to a submission ZIP and parity/runtime checks |
| `multi_solution.py` | Submission callback (`solution.py` in the ZIP) |
| `fast_kernel.c` | float16 C kernel, compiled at load time |
| `features.py` | Event-time features (batch and per-row `EventState`) |
| `oos_global.py` | `read_group` for validation (returns `is_scored & need_prediction` as the mask) |
| `global_wp.py` | Metric |
| `dev_check.py` | Unseen-DEV scoring of members and sets |
| `selection_pbo.py` | CSCV / PBO selection-overfitting check |

### Stage 0: base GRU96s on FIT (`improve.py`, `scale.train_gru`)

```
python research/improve.py fit_best_gru                            # gru96_norm5ep: standardized, 5 epochs, lr 1e-3 -> 3e-4 from epoch 4
python research/improve.py fit_ensemble_seed --name gru96_seed17   # raw inputs, 3 epochs, lr 1e-3
python research/improve.py fit_ensemble_seed --name gru96_seed37
```

- Loss: `mean((p − y)² · |y|)` over rows ≥ 99, with y clipped to ±2.
- Batch of 64 whole sequences, truncated BPTT in chunks of 512 rows; the hidden state is carried across chunks and detached between them.

### Stage 1: distillation into one GRU96 (`distill.py`), giving v1 and v2

```
python research/distill.py full          # v1: alpha 0.5  -> cache/advance/full_fit/distill/model.pt
python research/distill.py full_alpha75  # v2: alpha 0.75 -> model_alpha75.pt
```

- Teacher: the mean of the three stage-0 GRUs, cached for FIT in `teacher.npy` (not shipped, 728 MB; rebuilt automatically). `groups.npy` lists the FIT groups.
- Loss: `(1 − α) · mean((p − y)²|y|) + α · mean((p − teacher)²|y|)`.
- Schedule: 5 epochs, lr 1e-3, then 3e-4 from epoch 4; seed 57.

### Stage 2: wider students of the same teacher (`distill_mps.py`)

Same loss and schedule as stage 1. Output goes to `cache/distill_mps/<tag>.pt`.

```
python research/distill_mps.py --width 128 --alpha 0.5  --seed 57          # gru128_a0.5_s57
python research/distill_mps.py --width 192 --alpha 0.5  --seed 57          # gru192_a0.5_s57
python research/distill_mps.py --width 128 --alpha 0.75 --seed 58          # gru128_a0.75_s58
python research/distill_mps.py --width 192 --alpha 0.5  --seed 59 --event  # gru192ev_a0.5_s59 (+46 event features)
python research/distill_mps.py --width 192 --alpha 0.75 --seed 60          # gru192_a0.75_s60
python research/distill_mps.py --width 128 --alpha 0.75 --seed 61          # gru128_a0.75_s61
python research/distill_mps.py --width 128 --alpha 0.75 --seed 81 --split  # split128_a0.75_s81 (split encoder)
```

### Stage 3: mask-weighted fine-tuning on validation (`maskft.py`)

```
python research/maskft.py run   --model v1 --half 0     # writes cache/maskft/v1_u0.06_lr0.0002_s7_h0_e{1,2,3}.pt
python research/maskft.py run   --model v1 --half 1
python research/maskft.py final --model v1              # all validation -> ..._all_e{1,2,3}.pt
# same for gru128_a0.5_s57, gru192_a0.5_s57, gru128_a0.75_s58, gru192ev_a0.5_s59, gru192_a0.75_s60, gru128_a0.75_s61
# split encoder: --model split128_a0.75_s81 --target-balance   (tag ..._tb_...)
```

The defaults are the adopted recipe:
- unscored weight `u = 0.06`, Adam lr 2e-4, 3 epochs, seed 7;
- batch of 32 whole validation sequences, chunk 512.

How rows are weighted, per chunk:
```
row_weight = 1.0 if is_scored else 0.06 ;  row_weight[:, :99] = 0
w = row_weight * |y|           (y clipped to ±2)
loss = Σ w (p − y)² / Σ w
```

- **Target-balanced** (`--target-balance`): each target's weighted MSE is normalized by its own weight and the two are then averaged.
- **SWA**: a member's weights are the average of epochs 2 and 3. Every FT member used downstream is SWA(e2, e3).
- **WiSE hedge** for the split member (v27, v30–v35). It reproduces the shipped file exactly:
  ```python
  base = torch.load("research/cache/distill_mps/split128_a0.75_s81.pt")
  ft = [torch.load(f"research/cache/maskft/split128_a0.75_s81_u0.06_lr0.0002_s7_tb_all_e{e}.pt") for e in (2, 3)]
  wise = {k: base[k] + 0.9 * ((ft[0][k] + ft[1][k]) / 2 - base[k]) for k in base}
  torch.save(wise, "research/cache/maskft/split128_a0.75_s81_u0.06_lr0.0002_s7_tb_all_swa23_wise0.9.pt")
  ```

### Stage 4: mask net and teacher caches (`maskmix.py`, `ensdistill.teacher_cache`)

Both are built automatically by the first `ensdistill.py` run for a given split and generation.

**Mask net** (`cache/maskmix/masknet_{h0,h1,all}.pt`):
- A causal GRU64 on the standardized 112 inputs predicts P(is_scored | past rows).
- It is trained on the fitting split's validation sequences: BCE over rows ≥ 99, 3 epochs, Adam 1e-3, batch 64, chunk 512, seed 11.
- On the other half it reaches AUC about 0.905.
- Train rows (no mask) get expected-mask weights from it.

**Teacher caches** (`cache/ensdistill/teacher{,2,3}_{h0,h1,all}_{v,t,x4000}.npy`):
- float16 teacher predictions for the fitting split's validation sequences (`v`) and the 6,000-sequence pool (`t` = base 2,000, `x4000` = extra 4,000).
- About 0.63 GB per split per generation, so they are not shipped. A generation-g cache is built on top of generation g−1, which is built first if missing.

| Generation | Teacher |
|---|---|
| gen 1 | mean of the 7 FT members, each SWA(e2, e3): `v1, gru128_a0.5_s57, gru192_a0.5_s57, gru128_a0.75_s58, gru192ev_a0.5_s59, gru192_a0.75_s60, gru128_a0.75_s61` |
| gen 2 | 0.5 · mean of 9 models + 0.5 · mean(gen-1 GRU128 students). The 9 models are the 7 FT members, the target-balanced split encoder SWA(e2, e3), and the distilled GRU128 "ED" SWA(e1–e4). The gen-1 students are seeds 71–72 for the halves and 71–74 for all; each is SWA(e3–e5). |
| gen 3 | 0.5 · gen 2 + 0.5 · mean(twin-family students). Tested, **not adopted** (RESULTS.md). |

The gen-2 inputs are produced by these commands:
```
python research/ensdistill.py --student gru128_a0.75_s58 --alpha 0.5 --half 0        # "ED": defaults t1872, pool 2000, 4 epochs, lr 2e-4, seed 7
python research/ensdistill.py --scratch 128 --alpha 0.75 --train-per-epoch 6000 --pool 6000 --lr 0.001 --epochs 5 --seed 71 --half 0   # gen-1 student
```

Two naming caveats:
- The ED checkpoints were written before the `_p{pool}` part was added to the tag, so they are named `gru128_a0.75_s58_a0.5_t1872_lr0.0002_s7_*`. A re-run writes `..._t1872_p2000_...`; rename it, or use the shipped files.
- The shipped files are what `load_gen2_extra` expects.

### Stage 5: students (`ensdistill.py`), the deployed models

**Exact sampling, weighting and loss** (`ensdistill.main`):

1. **Epoch contents.** Every fitting-split validation sequence once (`--valid-repeat 1`). Add `--train-per-epoch 6000` train sequences drawn without replacement from the 6,000 pool, which means the whole pool each epoch. Shuffle all of it with `np.random.default_rng(seed)`. Every item is a whole 20,000-row sequence.
2. **Batches.** 32 sequences per update (validation and train mixed). Truncated BPTT over chunks of 512 rows; the hidden state is carried and detached per chunk, and there is one optimizer step per chunk.
3. **Row weights.**
   - Validation rows: `1.0 if is_scored else 0.06`.
   - Train rows: `q + 0.06 · (1 − q)`, where q is the mask net's probability on the standardized row.
   - Rows < 99: 0.
   - Then `w = m · |y|`, with y clipped to ±2.
4. **Loss per chunk:**
   ```
   label = Σ w (p − y)² / Σ w
   mimic = Σ w (p − t)² / Σ w            (t = cached teacher prediction)
   loss  = (1 − α) · label + α · mimic,  α = 0.75
   ```
5. **Schedule.**
   - Adam lr 1e-3 for 5 epochs; from epoch 4 the lr is multiplied by 0.3 (`--scratch` runs).
   - Then a **continuation**: `--init <e5.pt>` with 3 more epochs at lr 1e-4, 1e-4, 3e-5, fresh Adam, same seed and data order.
   - The deployed weights are the **SWA of the 3 continuation epochs**.
6. **Held-out scoring.** After each epoch the script scores the other half's scored rows and stores them in `configs/cache/ensdistill/<tag>.json` → `history`. Split `all` has no held-out half.

**Commands for the deployed families** (replace `S` with the seed and `H` with 0, 1 or −1; −1 = all validation):
```
# twin-encoder GRU (v37/v38 members; seeds 71-76)
c="--scratch 96 --cell twinenc --gen 2 --alpha 0.75 --train-per-epoch 6000 --pool 6000 --seed S --half H"
python research/ensdistill.py $c --lr 0.001 --epochs 5
python research/ensdistill.py $c --lr 0.0002 --epoch-lrs 0.0001 0.0001 0.00003 \
    --init research/cache/ensdistill/scratchtwinenc96g2_a0.75_t6000_p6000_lr0.001_sS_all_e5.pt   # (_h0/_h1 for halves)

# twin GRU (v32-v36; seeds 71-78): same with --cell twin
# GRU128 "C" students (v24-v33; seeds 75-78): same with --scratch 128 (no --cell)
# strong-recipe GRU128 "S" (v29/v30; seeds 71, 72), final = e8, no continuation:
python research/ensdistill.py --scratch 128 --gen 2 --alpha 0.75 --train-per-epoch 6000 --pool 6000 \
    --lr 0.0015 --epochs 8 --adamw 0.0001 --onecycle --clip 1 --seed S --half H
# 2-layer GRU192 "B192" (v29-v34): as S plus --scratch 192 --layers 2 --dropout 0.1
```

**Time per all-data student** on the laptop: about 45–60 minutes per epoch, so about 6–8 hours for 5 + 3 epochs.

### Stage 6: packaging and inference (`package_ft.py`, `multi_solution.py`, `fast_kernel.c`)

```
E=research/cache/ensdistill/scratchtwinenc96g2_a0.75_t6000_p6000_lr0.0002
python research/package_ft.py submission_v37 $(for s in 71 72 73 74 75; do printf "${E}_s${s}_cont_all_e1.pt+${E}_s${s}_cont_all_e2.pt+${E}_s${s}_cont_all_e3.pt "; done)
```

**Member syntax.** `a.pt+b.pt+c.pt` means the weight average (SWA) of those checkpoints. A trailing `@w` sets the ensemble weight; weights are normalized to sum to 1.

**Reproducibility check.** Running the command above on a fresh starter pack that holds only this bundle's code and checkpoints rebuilds v37:
- `model.npz` is identical, array for array, to the submitted `packages/submission_v37.zip`, and so are `solution.py` and `features.py`.
- Parity matches the shipped check: 4.1e-7 NumPy, 4.3e-4 kernel.
- The only difference is one line in `fast_kernel.c`. The encoder gather offset was generalized from 16 to the block's column count for the side-encoder test after v37/v38 were built; outputs are unchanged.

**What the packager checks**, against batched PyTorch on a complete validation sequence:
- row-wise callback parity: the NumPy path must be within 5e-5; the float16 kernel's gate is 2e-2, and it typically lands at 3–8e-4,
- warm-up behaviour,
- sequence reset,
- the runtime guard,
- the 20 MB size limit.

**How the callback runs:**
- Members are stepped row by row, grouped by width, with one shared input matmul.
- Standardization and biases are folded into the weights.
- Twins are exported as streams feeding a joint layer; encoder blocks are "kind 4" kernel members.
- `fast_kernel.c` is compiled at load time (`gcc -O3 -march=native`) and self-tested in a child process. If that fails, the NumPy path runs.

**Runtime guard:** 54 minutes after load, only the first member keeps running; after 58.5 minutes the callback outputs zeros.

**Official runtime fit** (kernel packages): minutes ≈ 7.2 + 0.0382 · kMAC + 0.18 · kernel members. It underestimated the twin-encoder packages by 1–4 minutes (v37 43m37s, v38 53m32s).

## 4. Student architectures (`ensdistill.py`)

| Cell | Structure | Params / inference cost |
|---|---|---|
| `gru` (`--scratch W [--layers L]`) | GRU(112 → W), linear head | GRU128 ≈ 92 kMAC |
| `twin` | i0 GRU96 on cols 0–51 + a0–a7 (60 inputs); i1 GRU48 on cols 52–103 + a0–a7; joint GRU96 over [h_i0, h_i1]; head | 131k params, ≈ 131 kMAC |
| `twinenc` (**final**) | twin, plus 8 per-group MLP encoders 11 → 32 → 16 (tanh) over the bid/ask × price/volume groups of each instrument; each stream reads raw 60 + its instrument's 4 codes (64) = 124 inputs | 166k params, ≈ 166 kMAC |
| `twinenc2` | twinenc + 4 side encoders (22 → 48 → 16) | rejected |
| `triple` | i0 prices / i0 volumes / i1 streams + joint | rejected |
| `siamese` | one GRU shared by i0 and i1 | rejected |
| `ple`, `mingru`, `lstm`, `warmupgru` | see RESULTS.md | rejected |
| split encoder (`distill_mps.SplitGRU`) | shared 52 → 32 encoder for i0 and i1, [i0, i1, i0 − i1, i0·i1, a-encoder] → GRU128 | teacher / ensemble member |

## 5. Checks and decision rules

**Three checks:**
- **Validation halves** (in period): the SWA model of each half is scored on the other half's scored rows. Cached held-out predictions live in `cache/maskft/pred_*.npz` (not shipped; recomputed from the checkpoints).
- **Unseen DEV** (out of period, 451 sequences): `python research/dev_check.py`.
- **Paired sequence bootstrap** (2,000 resamples) for set differences on DEV.

**Rules used:**
- Each rule was written down before the result was seen.
- Adopt only if both halves improve, or if a pre-set tolerance holds (e.g. no half loses more than 0.0003 in period, and DEV does not get worse).
- A single half was never enough.
- Seed noise for one student is about ±0.0015, so recipe comparisons use the same seed and data order.

**Selection overfitting.** `selection_pbo.py` (CSCV, 16 blocks) gave PBO 0.000 for the choices made inside the validation period.

## 6. Which checkpoints are included

All paths are relative to `checkpoints/cache/`.

- **Stage 0–2 bases:**
  - `improve/full_fit/gru96_{norm5ep,seed17,seed37}/scaled_gru.pt`
  - the standardizer
  - `advance/full_fit/distill/model.pt` (v1), `model_alpha75.pt` (v2), `groups.npy`
  - `distill_mps/<base>.pt` for the 6 teacher bases and `split128_a0.75_s81`
- **Stage 3 FT members:** `maskft/<base>_u0.06_lr0.0002_s7_{h0,h1,all}_e{2,3}.pt` for the 7 teacher bases, the target-balanced split (`..._tb_...`) and the WiSE split.
- **Mask nets:** `maskmix/masknet_{h0,h1,all}.pt`.
- **Gen-2 teacher inputs:**
  - ED: `ensdistill/gru128_a0.75_s58_a0.5_t1872_lr0.0002_s7_{h0,h1,all}_e{1-4}.pt`
  - gen-1 students: `ensdistill/scratch128_a0.75_t6000_p6000_lr0.001_s*_{h0,h1,all}_e{3,4,5}.pt`
- **Students, every family and split:**
  - `*_lr0.001_*_e5.pt`: end of the 5-epoch phase. This is the init for the continuation, or the final model for variants that were not continued.
  - `*_cont*_e{1,2,3}.pt`: continuation epochs; SWA of these is the member.
  - `*_lr0.0015_*_e8.pt`: one-cycle runs (S students, B192, label-only strong recipe).
- **Not shipped:**
  - intermediate epochs of rejected screens (still on the original machine),
  - teacher prediction caches (rebuilt automatically),
  - cached held-out predictions,
  - pilot arrays.

The deployed members of each package are listed in `packages/<name>.check.json`.

## Rules note

Under the competition rules this material is for members of the same registered team only: "private sharing between participants is also not allowed". Prize winners must hand over their own training code and pass a code review. Nothing here uses the briefly public `dodler/wunder_connectome` repository or any other shared solution; only general public ideas were used, and the log records them.
