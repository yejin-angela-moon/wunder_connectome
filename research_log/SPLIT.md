# Locked evaluation split (2026-09-16)

- `datasets/train.parquet` is the only source of labels for fitting and model selection.
- The development holdout is **every tenth train row group**, zero-based (`group % 10 == 0`): 1,061 complete sequences. The other 9,546 groups are eligible for fitting. Never fit preprocessing, models, or ensembles on development labels.
- `datasets/valid.parquet` is the fixed external out-of-sample (OOS) set: all 1,873 complete sequences. Its `is_scored` mask is used only for official evaluation; candidate fitting and tuning stay on train.
- The supplied ONNX baseline has no released training provenance. It may have seen train development labels, so its score on that holdout is not used to select or compare candidates. Its masked official OOS score is the external reference.
- The pilot uses 128 fit and 32 development sequences, evenly spaced through the respective group-index lists. `pilot.py prepare` records their group indices and sequence IDs in `research/cache/pilot_groups.json`. The pilot is a compute-limited screen, not a claim about full-data performance.
- Group membership and the official OOS designation stay fixed throughout this project. Do not use `seq_ix` as an input feature or interpret it as time.

The ordered sequence-ID SHA-256 digests lock the two source files' group order (`int64` little-endian IDs):

| File | Bytes | Digest |
|---|---:|---|
| train | 29,240,219,011 | `11ba03e69dfdb79a1a78db84d2895fa98711b8cbb2e2d0bb94d66d93ce19701b` |
| valid | 5,615,509,128 | `d85f36905d7ebdf72ddd1169862a1c9d2d87bf832bf7a81f0aabe953f1c331fd` |

The released data has no timestamps, and the documentation says `seq_ix` is shuffled. A calendar-based validation *period* cannot be recovered from these files. This is a fixed sequence-level split, not a chronological backtest. Development scores use every step from 99 onward because train has no `is_scored` column; the OOS score uses the supplied mask.

On 2026-09-16, the user requested OOS scores for all three frozen pilot models against the supplied baseline. This is a one-time comparison of unchanged artifacts. Once these OOS scores are viewed, `valid.parquet` is no longer an untouched holdout for future model choices; later training or tuning must not use its labels or scores as optimization feedback.

On 2026-09-18, after the official metric changed from equal block averaging to Global WP, the same fixed OOS set was used to re-score the unchanged pilot, expanded, and selected artifacts plus the supplied baseline. No model fitting or variant selection used the new OOS values. See [RESULTS.md](RESULTS.md); future work must still treat this public OOS set as already inspected.

## Chronological walk-forward, when sequence order is available

`research/walkforward.py` implements expanding-window refits and next-window scoring for all three candidates **inside the 9,546 fitting sequences only**. It requires an independently verified chronological CSV with exactly one row for every 10,607 train sequence IDs: columns `seq_ix,rank`, where rank 0 is earliest and ranks are consecutive. Neither shuffled `seq_ix` nor Parquet row-group order supplies this chronology. The fixed 1,061 development sequences and `valid.parquet` remain outside every walk-forward fold.

The predeclared default uses the earliest 50% of fitting sequences for fold 1 training, then three consecutive equal-sized validation windows, refitting each model on all earlier fitting sequences for each fold. A prior walk-forward validation window enters a later fold's training history; the fixed development holdout never does. Default embargo is zero because the target horizon is undisclosed; `--embargo-groups` can exclude that many immediately preceding fitting sequences if a justified gap becomes known. Scores use all required train rows, not the hidden official scoring mask.

```bash
python research/walkforward.py selfcheck
python research/walkforward.py plan --order-file /path/to/verified_train_chronology.csv
python research/walkforward.py run --order-file /path/to/verified_train_chronology.csv
```

`run --fold 1` (or 2 or 3) executes one fold; completed artifacts and scores go under ignored `research/cache/walkforward/fold_01/` etc. A full run also writes the sequence-weighted aggregate to `summary.json`. Existing fold directories are not overwritten. Until a verified chronology is supplied, only `selfcheck` can run and no real chronological walk-forward score can be reported. Even with such a CSV, correctness of its claimed time order must be established outside this code.
