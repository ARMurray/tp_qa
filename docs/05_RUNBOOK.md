# 5. Runbook

Commands, in order, for the things you will actually do.

---

## A. One full cycle

A cycle is: train → infer → review → fold back → retrain → score. Once the
pipeline is set up, this is the loop you repeat.

### A1. Fold in the last round of review (local)

```bash
cd review_app
python -m sync.close_round --round N --dry-run
python -m sync.close_round --round N
```

Omit `--round` to close every reviewed round at once — the catch-up case,
for a machine with review history that has never folded any of it in.

Then upload what it tells you to:

| File | Destination on HPC |
|---|---|
| `data/outgoing/training_locations.gpkg` | `correction/data/training/` |
| `data/outgoing/candidate_recall_failures.parquet` | `correction/data/features/` |
| `data/outgoing/holdout_truth_round{N}.parquet` | merge **by hand** into `data/holdout/holdout_truth.parquet` |

Skip this step entirely on the very first cycle — there is no review yet.

### A2. Refresh features for anything new (HPC)

```bash
cd /work/GRDVULN/tp_qa/correction/scripts

# Only for plants NEW to the corrections bin this round:
sbatch --array=0-N --export=STATES="OH PA ..." 01a_extract_parcels.slurm
sbatch 01c_run_od_corrected_locations.slurm

# REQUIRED for the re-ranker. Do not skip.
sbatch --export=SCOPE="train" 01e_run_od_candidates.slurm

# Preflight: fails if any OD partition predates the deployed best.pt
python check_od_freshness.py

# ONE job, comma-separated states
sbatch --export=STATES="OH,PA,..." 02_feature_engineering.slurm
```

Check the class counts at the end of 02's log before going further. If Stage
1 has single-digit Incorrect rows or Stage 2 has single-digit positives,
nothing downstream is trainable and you need more states or more review.

### A3. Retrain (HPC)

```bash
sbatch 03_train_stage1.slurm
sbatch 04_train_stage2.slurm

sbatch 06_build_stage2b_training.slurm   # then:
sbatch 07_train_stage2b.slurm

sbatch 06b_build_rerank_training.slurm   # then:
sbatch 07b_train_rerank.slurm
```

The `06`/`07` and `06b`/`07b` pairs are ordered — the build must finish
before the train. The two *pairs* are independent of each other.

### A4. Run inference (HPC)

```bash
# Per-state array. The national single job OOMs.
sbatch --array=0-N --export=STATES="OH PA ..." 05_run_inference_array.slurm
python merge_05_shards.py

sbatch 05b_rerank_candidates.slurm
```

### A5. Build the next review queue (HPC)

```bash
sbatch 10_build_review_queue.slurm    # --round N+1
python check_review_queue_scores.py   # confirms stage2a + stage2b scores present
```

### A6. Score the holdout (HPC)

```bash
sbatch 12_score_holdout.slurm
```

This is the only clean read on whether the cycle actually improved anything.
Every plant in it was excluded from all four models. Compare round over
round.

### A7. Review (local)

```bash
# Copy down from HPC into review_app/data/incoming/:
#   review_queue_round{N+1}.parquet   (from data/review_queue/)
#   holdout_manifest.parquet          (from data/holdout/)

cd review_app
python -m backend.queue_loader --round N+1
uvicorn backend.app:app --reload --port 8000
# open http://localhost:8000
```

Then back to A1.

---

## B. First-time setup

### B1. HPC

```bash
bash /work/GRDVULN/tp_qa/correction/scripts/setup_env.sh   # ON A LOGIN NODE
```

Then place the data:

| What | Where |
|---|---|
| CWNS text exports | `data/cwns/` |
| `training_locations.gpkg` (or `Updates.gpkg`) | `data/training/` |
| `best.pt` from local `detection/` | `models/object_detection/` |
| OSM `Wastewater_Plants.gpkg` | `data/reference/` |

Confirm the shared external stores exist:
`/work/GRDVULN/data/{parcels,nlcd,Census}`.

### B2. Local review app

```bash
cd review_app
pip install -r requirements.txt
```

**Migrate the master to GeoPackage — once per machine that holds it.**

The master moved from `.gdb` to `.gpkg` on 2026-09-21 so the review loop can
write the file it reads. If this machine still has `Updates.gdb`:

```bash
python -m sync.migrate_master_to_gpkg --dry-run
python -m sync.migrate_master_to_gpkg
```

It copies the newest dated layer across unchanged — a container change, not
a data change — and asserts the row and column counts match afterwards.
`close_round.py` cannot create the master, only add layers to it, so this
must happen first or the first fold-back fails with "Master gpkg not found."

> Do **not** seed the master from any `Updates.gpkg` written by the retired
> `pull_reviews.R`. Its corrections point at the location they were meant to
> correct. The migration script refuses such a source, but it is worth
> knowing why.

Then check `config.py`:

- `REGRID_STATE_GLOB` against your actual Regrid folder layout — **do this
  first**, it is the most likely thing to be wrong
- `MASTER_GPKG` points at your `Updates.gpkg`
- `FACILITIES_PATH` points at `FACILITIES.txt`

Verify parcel lookups work before your first review session; an empty parcel
lookup makes the app look broken in a confusing way.

### B3. Freeze the holdout — once, ever

```bash
sbatch 09_build_holdout.slurm
```

Do this **before** the first full inference run and never again. Everything
downstream anti-joins against the manifest it writes. Re-running it after
training has happened invalidates every comparison the project has produced.

---

## C. Smaller tasks

### Rebuild training bins without a full round

```bash
cd review_app
python -m sync.close_round --round N --skip-tiles
```

Or directly:

```bash
python correction/scripts/build_training_bins.py \
    --master "<path to Updates.gpkg>" \
    --out training_locations.gpkg
```

No `--layer` needed — it resolves the newest dated layer. Pass `--layer` to
pin an older one deliberately.

### Test the pipeline without the HPC

```bash
python correction/scripts/build_test_bundle.py --state OH
```

Assembles a self-contained copy — every script plus a geographically
concentrated subset of every input — into `correction/testing/tpqa_test/`,
mirroring the HPC layout. Lets you find schema/type/join bugs in seconds
instead of one `sbatch` per hypothesis.

> **It is a fixture, not a training set.** Any model trained on ~25 plants is
> meaningless. A fixture-trained `.joblib` must never be copied back to HPC.

### Correct a mis-clicked verdict

```bash
sqlite3 review_app/data/app.db "UPDATE plants SET reviewed = 0 WHERE cwns_id = '...'"
```

There is no admin UI for this.

### Retrain the object detection model (local)

```bash
cd detection
python pipeline/01_sample_sites.py --source ...
python pipeline/02_extract_tiles.py
# label in Label Studio
python pipeline/03_prepare_dataset.py
python pipeline/04_train_model.py
```

Then copy the new `best.pt` to HPC `models/object_detection/` and run
`check_od_freshness.py` before the next `02_feature_engineering` — stale OD
partitions produced by the *old* weights will otherwise be silently mixed in.

Note that `extract_review_tiles.py` (step 4 of `close_round`) has already been
feeding the labeling inventory every round, so there should be new material
waiting.

---

## D. Order-of-operations traps

These are the ones that fail silently rather than loudly.

| Trap | Consequence |
|---|---|
| Skipping `01e` before `06b` | Re-ranker silently gets no hard negatives from this round |
| Running `02` with stale OD partitions | Features mix old and new model outputs. `check_od_freshness.py` catches it — run it |
| Space vs comma separated `STATES` | 01a/01b are arrays (space); 02 is single (comma) |
| Re-running `09_build_holdout.py` | Destroys every round-over-round comparison, undetectably |
| Feeding `holdout_truth_round{N}.parquet` into training | Same |
| Rebuilding bins from a master that failed to update | Trains on last round's labels. `close_round.py` gates step 3 on step 2 for this reason |
