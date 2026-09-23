# 3. The correction pipeline

Everything here lives in `correction/scripts/` and runs on the HPC. Each
script has a `.slurm` wrapper of the same name.

**Read the script docstrings.** They are the design documents for this
pipeline and they record why choices were made, including ones that were
tried and reversed. This chapter gives you the shape; the docstrings give you
the reasoning.

---

## Numbered stages, in dependency order

```
build_training_bins.py     master Updates.gpkg -> training_locations.gpkg
        │                  (classes / corrections / unverified)
        ▼
01a_extract_parcels.py     k-ring parcel search + NLCD zonal stats
        │                  per state -> nlcd_{STATE}_k{K}.parquet
        ▼
01b_run_object_detection.py   OD at REPORTED locations -> od_features/
01c_run_od_corrected_locations.py   OD at CORRECTED locations -> od_features_corrected/
01d_nlcd_topup.py             NLCD for a specific parcel list (gap filler)
01e_run_od_candidates.py      OD at Stage 2a's TOP-K CANDIDATES
        │                     -> od_features_candidates{,_train}/
        ▼
02_feature_engineering.py     everything -> 05_/10_/14_/15_*.parquet
        │
        ├──▶ 03_train_stage1.py    Stage 1: is the reported location right?
        ├──▶ 04_train_stage2.py    Stage 2a: which candidate parcel is it?
        │
        ├──▶ 06_build_stage2b_training.py  ─▶ 07_train_stage2b.py
        └──▶ 06b_build_rerank_training.py  ─▶ 07b_train_rerank.py
                 │
                 ▼
05_run_inference.py        Stage 1 -> Stage 2a over the full universe
05b_rerank_candidates.py   applies the re-ranker to those candidates
        ▼
10_build_review_queue.py   assembles review_queue_round{N}.parquet
        ▼
                    [ human review, see 04_REVIEW_LOOP.md ]
        ▼
12_score_holdout.py        the clean read on the whole pipeline
```

---

## Full script inventory

### Core pipeline

| Script | Purpose |
|---|---|
| `build_training_bins.py` | Master `Updates.gpkg` → `training_locations.gpkg`. Splits into `classes` (Correct/Incorrect), `corrections` (known right answer), `unverified`. Resolves the newest dated layer automatically. |
| `01a_extract_parcels.py` | H3 k-ring search (k=18 ≈ 5.4 km) around each reported point; pulls candidate parcels and computes NLCD zonal stats. Array job, one task per state. |
| `01b_run_object_detection.py` | Runs `best.pt` on NAIP tiles covering each plant's **reported** parcel. Streams imagery from Planetary Computer. Resumable. |
| `01c_run_od_corrected_locations.py` | Same, for corrections-bin plants' **true** locations. Separate output root by design. |
| `01d_nlcd_topup.py` | NLCD stats for a specific parcel list, appended to 01a's output — for parcels that fell outside the original k-ring sweep. |
| `01e_run_od_candidates.py` | OD on Stage 2a's top-K candidate parcels. Keyed `(CWNS_ID, ll_uuid)`. **Required by the re-ranker.** |
| `04_train_model.py` (detection) | Copies `best.pt` to `correction/models/object_detection/` automatically on success — see below. |
| `dedupe_tiles_by_content.py` (detection) | Deduplicates the labelling inventory by image hash. Quarantines rather than deletes, and never drops a labelled tile whose annotation disagrees with its duplicate. |
| `02_feature_engineering.py` | Builds all feature tables. **Per-state array** — each task writes a shard with `--shard`. |
| `02b_merge_feature_shards.py` | Unions those shards into the flat files everything downstream reads. Validates all tables before writing any. |
| `03_train_stage1.py` | Stage 1 classifier. |
| `04_train_stage2.py` | Stage 2a candidate ranker. |
| `05_run_inference.py` | Stage 1 → Stage 2a over the full universe. Run as a per-state array (`05_run_inference_array.slurm`) — the national single job OOMs. |
| `merge_05_shards.py` | Concatenates the array's per-state output into the flat files `10` reads. |
| `05b_rerank_candidates.py` | Adds `rerank_score` alongside `stage2_prob_correct` so the two orderings can be compared. |
| `06_build_stage2b_training.py` | Contrastive table: reported parcel (label 0) vs corrected parcel (label 1). |
| `07_train_stage2b.py` | Trains Stage 2b. |
| `06b_build_rerank_training.py` | One row per candidate parcel; 1 for the true parcel, 0 for the ~19 competitors. **This is where review data pays off most.** |
| `07b_train_rerank.py` | Trains the re-ranker. |
| `09_build_holdout.py` | Samples the frozen evaluation holdout **once**. |
| `10_build_review_queue.py` | Assembles a reviewable batch. |
| `12_score_holdout.py` | Scores the holdout. The clean read. |

### Shared modules

| Script | Purpose |
|---|---|
| `config.py` | Every path and tunable. The only file to edit if the pipeline moves. |
| `holdout.py` | `exclude_holdout()` — fail-closed. Imported by 03/04/06/07/06b/07b and 12. |
| `model_utils.py` | Preprocessing pipeline, spatial CV folds, Youden-J threshold selection. |

### Diagnostics — run these before changing a modelling decision

| Script | The question it answers |
|---|---|
| `08_diagnose_candidate_coverage.py` | Do corrections fall inside the k-ring window at all? Gates every Stage 2b data decision. |
| `08b_analyze_ring_misses.py` | Would raising `K_RINGS` actually help? |
| `diagnose_candidate_od.py` | Does OD actually discriminate among top-20 candidates? (Yes — 46.4% vs 9.0%.) |
| `diagnose_stage1_attrition.py` | Where do plants drop out of Stage 1? |
| `diagnose_od_class_shift.py` | How much signal depends on OD, before swapping in new weights? |
| `check_od_freshness.py` | **Preflight for 02.** Fails if any OD partition predates the deployed `best.pt`. |
| `inspect_model_features.py` | What was each deployed model *actually* fitted on? |
| `list_training_states.py` | Which states are in the training universe, with counts. Run before submitting a big job. |
| `check_review_queue_scores.py` | Does the review queue actually carry both stage2a and stage2b scores? |

The remaining `diagnose_*` / `check_*` / `inspect_*` scripts are one-off
investigations of specific incidents. Their docstrings state which. They are
kept because the incidents recur.

### One-off patches — historical

`patch_01e_training_scope.py`, `patch_05_for_array.py`,
`*.bak-2026*`, `build_tasks_patch.diff`, `naip_fetch_patch.diff`. These were
applied already. They are a record of what changed and why, not something to
run. See [07_OPEN_ITEMS.md](07_OPEN_ITEMS.md) — cleaning these up is a
reasonable early task for a new owner.

---

## Argument style — a real trap

**01a and 01b are array jobs** and take space-separated states, one per task:

```bash
sbatch --array=0-1 --export=STATES="OH PA" 01a_extract_parcels.slurm
```

**02 is also an array now** (changed 2026-09-22 — a national single job took
too long), one state per task, followed by a merge:

```bash
JID=$(sbatch --parsable 02_feature_engineering.slurm)
sbatch --dependency=afterok:$JID 02b_merge_feature_shards.slurm
```

Getting the array/single distinction backwards fails in a confusing way rather
than an obvious one.

### Why the merge step exists

02 writes five outputs. Only one — `10_parcel_features_by_state/` — was
already keyed by state. The other four are fixed filenames, so running 02 as
an array without `--shard` has all 52 tasks overwrite the same four files and
the last to finish wins. Nothing errors; 03 and 04 train on one state's data.

`--shard` redirects the plant-keyed outputs to
`data/feature_shards/state=XX/`, and `02b_merge_feature_shards.py` unions
them back. Same pattern as `05_run_inference_array.slurm` +
`merge_05_shards.py`, which solved this first.

**Concatenation is valid here, and that was checked rather than assumed:**
`build_stage1_training` groups by `STATE_CODE`; `build_stage2_training` reads
each state's own `nlcd_{state}` file for its candidate pool; `add_name_matching`
is row-wise; nothing normalises, ranks or aggregates across plants. A plant's
candidates already came only from its own state, so sharding loses nothing.
If that ever stops being true, the merge stops being correct silently.

---

## Resumability

| Script | Resumable? |
|---|---|
| `01b`, `01c`, `01e` | **Yes**, by default. Resubmit the identical command; any `CWNS_ID` already written is skipped. Flushes every 200 plants, so at most ~200 plants of work is lost to a kill. `NORESUME=1` forces a full reprocess. |
| `01a`, `02` | No. Re-run from scratch. |
| Training scripts | No, but they are fast relative to the OD steps. |

---

## Tunables worth knowing about

All in `config.py`, all with long explanatory comments. The ones most likely
to come up:

| Constant | Value | Why it is what it is |
|---|---|---|
| `K_RINGS` | 18 | ≈5.4 km search **radius** (not diameter — earlier comments had this wrong). Mean correction distance is ~5.05 km, so the average correction lands near the window boundary. `08`/`08b` measure whether to change it. |
| `MAX_PARCEL_AREA_M2` | 2,000,000 | Excludes "whole town as one polygon" digitization artifacts. A judgment call, not a measured value — the docstring says how to set it empirically. |
| `MAX_TILES_PER_PLANT` | 150 | Last-resort cap. Raised from 60 after a real KY plant strung along a road legitimately needed 121 tiles. |
| `TARGET_RES_M` | 0.6 | **Must match `detection/config.py` exactly.** The OD model was trained at this resolution; changing it feeds the model images unlike anything it learned on. |
| `NAIP_WORKERS` | 32 | Tested roughly linear to 47 on the cluster. The local config's 4 was tuned around an EPA TLS-inspection proxy that doesn't apply here. |
| `TOP_K_SHOWN` | 5 (in `review_app/config.py`) | Candidates shown to the reviewer. Must match `10_build_review_queue.py`. |

---

## Known pipeline gaps

- **Training set size.** The OH-only pilot produced 47 Correct / 1 Incorrect
  for Stage 1 and 4 positives / 73,142 negatives for Stage 2. Neither is
  trainable. Check the class counts at the end of 02's log before submitting
  03/04 — widening the state list is what fixes this, and the review loop is
  what fixes it durably.
- **Memory ceilings** in the `.slurm` files are estimates from local runs.
  Check real usage with `seff <jobid>` and adjust.
- **`05_run_inference.py` OOMs as a national single job.** Use the array
  wrapper plus `merge_05_shards.py`.
- **Corrected parcels outside the k-ring** get dropped from Stage 2b training
  with a reported count, rather than triggering a targeted NLCD top-up.
  `01d_nlcd_topup.py` exists to fix this if the count turns out to matter.
