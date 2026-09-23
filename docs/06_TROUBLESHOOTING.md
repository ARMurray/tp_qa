# 6. Troubleshooting

Failures that have actually happened on this project, and what they meant.
Every entry here cost someone real time to diagnose the first time.

---

## Data / IO

### `Geoparquet metadata does not have a version`

DuckDB is trying to interpret Regrid's embedded GeoParquet metadata and
failing. Every script that touches parcel parquet must set:

```sql
INSTALL spatial; LOAD spatial; SET enable_geoparquet_conversion = false;
```

This is not optional. If you write a new script that reads the parcel store,
copy that line.

### Parcel lookups return nothing / empty map in the review app

`REGRID_STATE_GLOB` in `review_app/config.py` does not match your actual
local folder layout. It is set to `state={state}/*.parquet`, mirroring the
HPC convention, but was never verified against the real local mirror.

`parcels.py` prints the glob pattern it tried on failure — read that line and
compare it to what is actually on disk.

### `pyarrow` crashes reading the OD output directory

Do not `pd.read_parquet(directory)` on `od_features/`. Whole-directory schema
merge crashes. Also do not take "the newest part-file per partition" — that
silently drops rows from a resumable, multi-flush pipeline.

Use the union-then-dedup pattern in `02_feature_engineering.py` or
`06_build_stage2b_training.py`'s `read_od_plants_union()`: read every
`part-*.parquet`, sort by mtime, `drop_duplicates(subset="CWNS_ID", keep="last")`.

### 5.47× row inflation in 02's "Parcel features combined" step

Duplicate parcels in the source. See `diagnose_parcel_duplicates.py` — it was
written for exactly this, on the OH/MS/DE pilot.

### `TypeError: unhashable type: 'bytearray'` inside `OneHotEncoder`

Some value in a nominally-string feature column is a `bytearray`, coming out
of a live DuckDB query rather than from any already-written file. See
`diagnose_bytearray_columns.py` and `diagnose_scoreable_dtypes.py`.

### Columns named `STATE_CODE_x` / `LATITUDE_y` in a model

A merge collision — both frames carried the column, so pandas suffixed them
and the model was fitted on the wrong one. `inspect_stage2_columns.py` traces
it. `06_build_stage2b_training.py`'s `NAME_MATCH_GEO_COLS` comment documents
the general trap: never merge in a whole feature table when you need four
columns from it.

---

## Modelling

### A model's categorical space is mostly one high-cardinality column

The known instance: raw `owner` text survived into
`16_stage2b_training.parquet`, and `07`'s `build_preprocessor()` one-hot
encoded it into **226 levels — 74% of the deployed model's entire categorical
feature space**, almost all novel at inference time.

The fix was to run the frame through `02`'s `add_name_matching()`, which turns
`owner` into six derived features (`is_municipal`, `owner_water`, `sd_match`,
`place_match`, `county_match`, `any_geo_match`) and drops the raw column.

Run `inspect_model_features.py` to audit what each deployed model was actually
fitted on. The same trap applies to raw geography dummies.

### Stage 2b says object detection doesn't matter

It is not saying that. It is saying OD has no residual variance to explain
*on the pairs 06 builds* — reported parcel vs corrected parcel, which parcel
attributes separate trivially. See [01_ORIENTATION.md](01_ORIENTATION.md).

The re-ranker (`06b`/`07b`) measures the question you actually care about.
`diagnose_candidate_od.py` has the numbers.

### `check_od_freshness` passes but the detector was just retrained

`04_train_model.py` deploys `best.pt` with `shutil.copy`, which stamps the
copy with the current time. If someone changes that to `copy2`, or copies it
by hand with `scp -p` / `rsync -t`, the source mtime is preserved and a
brand-new model looks old — so the freshness check passes over detection
output produced by the *previous* weights. The check's own docstring warns
about this. `touch` the deployed file if you ever land in that state.

### A round of review produced no improvement in the re-ranker

Most likely `01e_run_od_candidates.py` was not re-run with `SCOPE="train"`.
`06b` silently drops any plant whose candidates have no OD output, so the
round's hard negatives never reached training and nothing errored.

Check: `06b`'s log prints how many candidates had OD output and across how
many plants. Compare to the plant count you expected.

### Feature tables only cover one state

02 ran as an array without `--shard`, or `02b_merge_feature_shards.py` never
ran. All 52 tasks write the same four filenames, so the last task to finish
wins. Check `05_plant_features.parquet`'s `STATE_CODE` values:

```python
import pandas as pd
print(pd.read_parquet("data/features/05_plant_features.parquet").STATE_CODE.value_counts())
```

One state means the merge is missing. Re-run `02b_merge_feature_shards.py`
— the shards are still under `data/feature_shards/`, so nothing is lost.

### `MISSING shard for N state(s)` from 02b

Those array tasks failed or never ran. The merge refuses rather than writing a
table that silently omits them. Check `logs/02_<jobid>_<taskid>.log` for the
named states, re-run just those (`sbatch --array=35 02_feature_engineering.slurm`),
then re-merge. Nothing was written, so the existing flat files are untouched.

### Training class counts are too small to train on

The OH-only pilot gave 47 Correct / 1 Incorrect for Stage 1 and 4 positives /
73,142 negatives for Stage 2. Widen the state list, and keep reviewing — the
review loop is what fixes this durably.

`list_training_states.py` shows what you currently have, per state.

---

## HPC

### Job fails immediately with a Python syntax error or ancient package versions

The compute nodes' default `python3` is **3.6**. `_common.sh` runs a
module-load loop to find a real one. If `module avail python` shows a version
whose name isn't in that loop, add it.

### `ERROR: venv not found at $ROOT/.venv`

`setup_env.sh` has not been run, or was run on a compute node. Run it on a
**login node**.

### `05_run_inference.py` runs out of memory

Known. Use `05_run_inference_array.slurm` (per-state array) followed by
`merge_05_shards.py`. `patch_05_for_array.py` is the record of what changed.

### 01b / 01c / 01e hit the time limit

They resume by default. Resubmit the identical command — any `CWNS_ID`
already written is skipped. They flush every 200 plants, so at most ~200
plants of work is lost.

Do **not** pass `NORESUME=1` unless you actually want a full reprocess.

### NAIP fetch failures spike

Probably rate-limiting from Planetary Computer. `NAIP_WORKERS = 32` tested
roughly linear to 47, but on one trial — the ceiling is not established. Lower
it and watch 01b's fetch-failure count. `diagnose_fetch_failures.py` reads the
`fetch_error` string written for every tile with `outcome='fetch_failed'`.

### Memory limits in `.slurm` files are wrong for your data

They are estimates from local runs. Check actual usage with `seff <jobid>`
after a run and adjust.

---

## The holdout

### `FileNotFoundError: Holdout manifest not found`

This is **intended behaviour**, not a bug. `exclude_holdout()` is fail-closed:
a training run that silently trains on its own evaluation set is worse than
one that stops.

Either the manifest genuinely hasn't been built (`09_build_holdout.slurm`,
once ever), or you are on a machine that doesn't have it. Pass
`allow_missing=True` / `--allow-no-holdout` **only** for a deliberate
pre-holdout run.

### Holdout numbers look suspiciously good

Suspect leakage. Check that whatever training script ran actually calls
`exclude_holdout()`, and that no `holdout_truth_round{N}.parquet` was merged
into anything other than `holdout_truth.parquet`.

`manifest_checksum()` exists so a stale or edited manifest on the review
machine is caught rather than silently used.

---

## The master locations file

### `No CWNS_Locations_YYYYMMDD layers in <path>`

`latest_master_layer()` found no layer matching the naming convention. Either
the file is wrong, or a layer was written with a different name. It raises
rather than falling back to an arbitrary layer, because training on last
month's labels because a name didn't match is a silent, expensive failure.

### `Corrected_X == Original_X` on rows that claim to be corrected

The old `pull_reviews.R` bug. It wrote the reported point as the correction
for every `candidate_correct` verdict. Any master layer written by that script
has this. See [04_REVIEW_LOOP.md](04_REVIEW_LOOP.md).

### `reviewed CWNS_ID(s) are not in the master layer at all`

`update_master_locations.py` reports these and does not add them. The master
is supposed to be the full CWNS universe, so a miss means an ID mismatch worth
looking at — probably a string/int coercion somewhere upstream.

### `TypeError: Invalid value for dtype 'str'`

A master layer where `Corrected_X`/`Corrected_Y` is entirely null reads back
from GPKG as a *string* column, and assigning a float into it raises rather
than upcasting. `update_master_locations.py` widens those columns explicitly
before writing. If you write new code that touches them, do the same.
