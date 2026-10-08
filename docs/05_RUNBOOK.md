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

Step 4 (NAIP tiles) runs with `detection/.venv`'s Python and only for round
N's plants, tiling just the grid cells that touch each parcel. New tiles show
up in `label_app.R` after a restart.

**Then commit and push immediately** — the master and `app.db` are binary and
git cannot merge them:

```bash
git add correction/data/training/Updates.gpkg review_app/data/app.db review_app/data/outgoing/
git commit -m "round N closeout" && git push
```

No file transfer is needed: the HPC gets `Updates.gpkg` through `git pull`,
`00_build_training_bins` builds the training labels from it there, and 10's
wrapper copies the review logs out of the repo. The one manual step left is
`holdout_truth_round{N}.parquet` (only when holdout plants were reviewed):
merge it **by hand** into `data/holdout/holdout_truth.parquet` on the HPC.

Skip this step entirely on the very first cycle — there is no review yet.

### A2–A6. Retrain, infer, score (HPC) — the order that works

Updated 2026-10-02. **Submit each step only after the previous one has
finished** (`sacct -j <JOBID> -X --format=JobID%20,State,ExitCode`). The login
shell is tcsh, so `$(...)` chaining does not work, and `--dependency=afterok`
has repeatedly failed with "Job dependency problem" — check and submit by hand.

```bash
cd /work/GRDVULN/tp_qa && git pull          # brings the master, labels, code
cd correction/scripts

# 1. Training labels from the master (Updates.gpkg is tracked in git, so the
#    pull already brought the newest round)
sbatch 00_build_training_bins.slurm

# 2. ONLY if the detector (best.pt) changed since the last run: re-run
#    detection at every reported location. NORESUME=1 is essential -- resume
#    keys on CWNS_ID and cannot tell which model wrote a row.
sbatch --array=0-50%6 --export=FULLUNIVERSE=1,NORESUME=1 01b_run_object_detection.slurm
sbatch check_od_freshness.slurm                 # must report no stale partitions

# 3. Features -- full universe, every time. 02b must succeed: if it fails it
#    writes NOTHING and 03/04 will silently train on the previous tables.
sbatch --export=FULL_UNIVERSE=1 02_feature_engineering.slurm
sbatch 02b_merge_feature_shards.slurm           # after all 51 tasks COMPLETED 0:0

# 4. Stage 1 and Stage 2a
sbatch 03_train_stage1.slurm
sbatch 04_train_stage2.slurm

# 5. Inference: 48 states (INFERENCE_STATES -- no AK/HI/PR, no NAIP).
#    TX (index 40) needs 128G.
sbatch --array=0-39,41-47%8 05_run_inference_array.slurm
sbatch --array=40 --mem=128G 05_run_inference_array.slurm
sbatch merge_05_shards.slurm

# 6. Re-ranker. 01e reads 05's new candidates, so it comes AFTER 05.
#    Add NORESUME=1 if the detector changed.
sbatch --export=SCOPE="train" 01e_run_od_candidates.slurm
sbatch 06b_build_rerank_training.slurm
sbatch 07b_train_rerank.slurm                   # refuses a stale 17_ table

# 7. Holdout candidates, re-rank, score
sbatch 01e_run_od_candidates.slurm              # holdout scope (NORESUME=1 after a detector change)
sbatch 05b_rerank_candidates.slurm
sbatch 12_score_holdout.slurm

# 8. Logs back through git
sbatch --export=TRAINING=1 collect_logs.slurm
#    then, from a login node: git add correction/diagnostics/ && git commit && git push
```

Before step 5 on a fresh setup, `sbatch preflight_inference.slurm` checks that
02's tables, 01a's candidates and 01b's detections cover the full universe.

**What to check:** 02b ends `=== merge complete ===` with four `wrote` lines;
02 and 05 print `[population]` lines; 03 and 04 include `name_match_*`
columns; 05 prints the OSM-confirmed count; 07b's model is newer than 04's.
Step 5 is `05_run_inference_array.slurm` — the national
`05_run_inference.slurm` now refuses `--array`, and `merge_05_shards` refuses
shards older than the models. `TRAINING=1` collect_logs does not pick up the
01e / 05b logs; collect those with `PATTERN=`.

Re-running one failed 02 state: array indices follow `DEFAULT_STATES` in
`_common.sh` (51 states, DC excluded — **OH is 34**). Re-run with
`sbatch --array=34 --export=FULL_UNIVERSE=1 02_feature_engineering.slurm`,
then the full 02b. For 05 the list is `INFERENCE_STATES` (48; OH is 32).

### A5. Build the next review queue (HPC) — four steps

The queue picks plants, detection runs on their candidates, the re-ranker
re-ranks them with those detections, and the queue is rebuilt **with the same
plants** (`KEEP=1` — the re-rank changes the margins the uncertain slice is
chosen on, so a plain rebuild picks different plants).

```bash
sbatch --export=ROUND=5 10_build_review_queue.slurm
sbatch --export=SCOPE="queue",ROUND=5,NORESUME=1 01e_run_od_candidates.slurm
sbatch 05b_rerank_candidates.slurm                # ONLY after 01e shows COMPLETED -- 10 now refuses otherwise
sbatch --export=ROUND=5,KEEP=1 10_build_review_queue.slurm
sbatch --export=PATTERN="10_*",LATEST=1 collect_logs.slurm
cd /work/GRDVULN/tp_qa
cp correction/data/review_queue/review_queue_round5.parquet review_app/data/incoming/
git add correction/diagnostics/logs/ review_app/data/incoming/review_queue_round5.parquet
git commit -m "round 5 queue" && git push
```

From round 6, add `AUDIT=80` to the first 10 call (not the `KEEP=1` one):
80 plants drawn at random from 13's `moved` plants, before the
uncertain/random split, so 13 must be current (10 refuses a 13 output older
than 05b). See `docs/RUN_20261008_round6.md`.

10's wrapper copies prior rounds' review logs from the repo itself, so earlier
plants (including `needs_info`) are excluded automatically. Check its log for
`--keep-selection: reusing the 150 plant(s)` and, in the provenance block,
`written BEFORE the deployed best.pt: 0`.

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

### Get cluster state to a machine that cannot see the cluster

The HPC and whatever machine is doing the analysis share no filesystem, and the
login console is not always somewhere things can be run interactively. Git works
in both directions, so two scripts write text into
`correction/diagnostics/`, which is **deliberately not gitignored** (unlike
`logs/` and `data/`).

**State snapshot** — what exists on disk, what the deployed detector is, parcel
coverage, detection inventory, feature tables, log tails:

```bash
sbatch collect_diagnostics.slurm
```

**Whole job logs** — when the question is what a run *printed*, not what exists:

```bash
sbatch --export=TRAINING=1 collect_logs.slurm
```

`--training` takes the newest log for each of `03`, `04`, `06`, `06b`, `07`,
`07b` and `12` — per pattern, so `12`'s single log is not crowded out by two runs
of `04`. One call captures a round's whole modelling picture. Comparing those
across rounds is what exposed the spatial-CV fold imbalance; no single run made
it visible.

For anything else, name a pattern. Note `PATTERN` **replaces** the training
preset rather than adding to it, and `CLEAR=1` wipes earlier copies — so two
batches means two calls, with `CLEAR` on the first only:

```bash
sbatch --export=PATTERN="08_*",LATEST=1 collect_logs.slurm
```

Then, **from a login node** — the job cannot push for you:

```bash
cd /work/GRDVULN/tp_qa
git add correction/diagnostics/
git commit -m "diagnostics"
git push
```

Progress-bar output is collapsed to its final frame and anything over 512 KB is
truncated from the middle, keeping head and tail — inputs are at the top, results
at the bottom. Every truncation says so inside the copy. Originals in
`correction/logs/` are never touched.

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
# label in detection/OWM_Imagery_Labeler/label_app.R (not Label Studio)
python pipeline/03_prepare_dataset.py
python pipeline/04_train_model.py
```

**Back up the deployed model first** (`copy correction\models\object_detection\best.pt best_previous.pt`): `04_train_model.py` **deploys `best.pt` itself** on success, whether or not it is better. Commit the new one only if it is clearly better; otherwise `git checkout -- correction/models/object_detection/best.pt`.

It deploys copying it to
`correction/models/object_detection/best.pt` — no manual step. Nothing is
lost: every run's weights stay under `detection/models/runs/`, which is the
archive; the deployed copy is just a pointer to whichever one is current.

Commit and push that file so the HPC picks it up, then re-run `01b`, `01c`
and `01e` and check `check_od_freshness.py` before the next
`02_feature_engineering`. A new detector makes every existing detection
output stale, and mixing two models' output in one feature table is silent.

> The deploy uses `shutil.copy`, not `copy2`, on purpose. `copy2` preserves
> the source mtime, which would make a brand-new model look old to
> `check_od_freshness.py` — the same trap its docstring warns about for
> `scp -p` and `rsync -t`.

Note that `extract_review_tiles.py` (step 4 of `close_round`) has already been
feeding the labeling inventory every round, so there should be new material
waiting.

**What gets tiled for labelling (2026-10-08).** Only two kinds of tile, from
reviewed plants (not `needs_info`): tiles on a verified true location's
parcel(s), including parcels ticked "also part of this plant"; and tiles on a
candidate parcel where the detector fired but the reviewer said it is not the
plant (false positive). Rejected candidates where nothing fired are not tiled:
there are already ~900 confirmed-empty labels. This is the default of
`extract_review_tiles.py` and so of `close_round`; `--all-candidates` and
`label_priorities.py` are opt-in only. Each run takes only the **latest reviewed round**
(`--round N` for another, `--all-rounds` for every round).

To clear an inventory built under older rules (keeps every labelled tile,
including old Label Studio names and confirmed-empty labels):

```bash
cd detection/pipeline
python purge_unlabelled_tiles.py            # dry run: counts by source/role
python purge_unlabelled_tiles.py --apply    # deletes PNG + NDWI, trims tile_metadata.csv (backed up)
cd ../../review_app
..\detection\.venv\Scripts\python.exe -m analysis.extract_review_tiles --dry-run
..\detection\.venv\Scripts\python.exe -m analysis.extract_review_tiles
```

then restart `label_app.R`.

---

## D. Order-of-operations traps

These are the ones that fail silently rather than loudly.

| Trap | Consequence |
|---|---|
| Skipping `01e` before `06b` | Re-ranker silently gets no hard negatives from this round |
| Skipping `02b` after the `02` array | Feature tables stay as the last run left them — 03/04 train on stale or single-state data |
| Running `02` as an array without `--shard` | All 51 tasks write the same four filenames; last to finish wins |
| Continuing after a failed `02b` | 02b writes nothing on failure, so 03/04 train on the PREVIOUS tables without error (happened 2026-09-29). Check for `merge complete` |
| New `best.pt`, then 01b/01e without `NORESUME=1` | Resume skips every plant/parcel already written, so the old detector's output stays in place |
| Rebuilding the review queue after 05b without `KEEP=1` | Different plants get picked -- ones 01e never examined |
| Running `02` with stale OD partitions | Features mix old and new model outputs. `check_od_freshness.py` catches it — run it |
| Space vs comma separated `STATES` | Array wrappers (01a, 01b, 02, 05) take space-separated lists; `--export` splits on commas, so pass comma lists positionally |
| Re-running `09_build_holdout.py` | Destroys every round-over-round comparison, undetectably |
| Feeding `holdout_truth_round{N}.parquet` into training | Same |
| Rebuilding bins from a master that failed to update | Trains on last round's labels. `close_round.py` gates step 3 on step 2 for this reason |
