# 7. Open items

Grouped by whether it is a decision, a gap, or a cleanup. Sorted roughly by
consequence within each group.

---

## Settled — recorded so they don't get reopened

### `needs_info` is a real outcome, not a backlog

88 of the first 300 reviewed plants came back `needs_info`. That is **not** a
gap to close. It means exactly what it says: the reviewer could not determine
the truth, so the plant remains an unknown.

Those rows stay `Verified = "No"` in the master and carry no label into
training. `update_master_locations.py` skips them explicitly rather than
guessing, and `10_build_review_queue.py` excludes every previously-reviewed
plant — including `needs_info` ones — from later rounds, so they do not
silently resurface in the next random draw.

If you ever *want* to re-queue some of them (better imagery, a better
candidate pool), that is a deliberate choice to make explicitly, not a
default to restore.

---

### Training metrics are measured at 1:50, deployment is ~1:10,500

Stage 2's log reports `roc_auc: 0.99`, `specificity: 0.96`. Those are computed
on the negative-downsampled set (`--neg-ratio 50`). The raw table is 3.7M
candidate rows against 353 positives — about 10,500 candidate parcels per
plant. A 3.6% false-positive rate at that scale is roughly **380 false parcels
per plant**.

So the Stage 2 numbers are not a statement about deployment, and never were.
`12_score_holdout.py` is, because it scores rank-1 accuracy per plant. The
threshold drifting 0.318 → 0.409 → 0.439 across three rounds is the same fact
showing up a second way: Youden's J is tracking a prior set by `--neg-ratio`,
not a model getting better.

Do **not** "fix" this by raising `--neg-ratio` toward reality. 1:10,500 is not
trainable; the top-K candidate step and the re-ranker are the parts of the
design that deal with it. `04`'s log now prints the deployment ratio next to
the training ratio, so the distinction is visible in place rather than needing
to be rediscovered.

### Spatial CV folds were rebuilt (2026-09-23) — old CV numbers aren't comparable

`spatial_cluster_folds` was `KMeans(n_clusters=5)` with leave-one-cluster-out.
Plants aren't uniformly distributed, so k=5 on US plant locations reliably made
one huge cluster and one tiny one. Measured across three Stage 2 runs a month
apart, cluster sizes were ~59 / 14 / 12 / 10 / 4 % of rows **every time** — the
same shape, merely permuted in index. In the 2026-09-23 run, fold 1 trained on
40% of the data and fold 3 computed its ROC AUC from **10 positives**, with
`RandomizedSearchCV` weighting all five folds equally.

It now clusters into `n_splits * 6` blocks and packs whole blocks into
size-balanced folds, clustering on distinct locations so row counts cannot drag
the centroids. On synthetic data shaped like the real thing, the old code
produced two folds with *zero* positives; the new one gives five even folds of
67–75 positives each.

Two consequences worth knowing:

- **`stage2_cv_comparison.parquet`, and any spatial CV score from before this
  date, cannot be compared against one from after it.** Different folds,
  different numbers. The `12_score_holdout` series is unaffected and stays
  comparable.
- Spatial separation is now *weaker per fold* — 30 small blocks means a held-out
  block's nearest neighbours may sit in the training set, where 5 big regions
  guaranteed they did not. That is the standard spatial-block-CV trade-off,
  taken deliberately to get folds whose scores mean something.
  `blocks_per_fold` tunes it.

### Stage 2's reported metrics used to leak across the plant boundary

The final fit used `train_test_split(stratify=y)` on rows. Each plant
contributes 1 + `NEG_RATIO` rows sharing every plant-level feature — discharge,
population, census, name matching — differing only in parcel attributes. So the
same plant sat on both sides of the split; measured on the real shape, **all 395
plants did**. Now `GroupShuffleSplit` on `CWNS_ID`.

Expect the reported numbers to come down. That drop is the fix working, not a
regression — the real task is ranking candidates for a plant never seen before,
and that is now what the split measures.

### 11% of Stage 2 plants have no positive row at all

`Plants: 395` against `Positive labels: 353`: 42 plants' corrected parcel is not
among their candidates, so they contribute only negatives and cannot be learned
from. That is a candidate-**recall** ceiling — no amount of model tuning
recovers a plant whose answer was never offered.
`candidate_recall_failures.parquet` says whether they are k-ring misses or
parcel-store gaps.

Worth reading before investing in model architecture: if recall is the binding
constraint, a better classifier cannot reach those plants at all.

## Decisions that need a human

### Weighting `confirmed_proposal` vs `independent`

The app captures `confirmation_type` — whether the reviewer reached the answer
themselves or agreed with what the model proposed. Agreeing with the model is
weaker evidence than finding it independently, and using both at equal weight
risks a confirmation-bias feedback loop where the model is trained to agree
with itself.

The field is captured. Nothing uses it yet. Deliberately deferred.

### K shown to the reviewer

`TOP_K_SHOWN = 5`. Larger K means more hard negatives per review and a better
`candidate_recall` estimate, but slower reviews. Revisit now that per-review
timing is actually known.

Note this interacts with the re-ranker: `06b` builds negatives from the
**top-20** pool regardless of what was shown, so raising K improves the
recall estimate and reviewer context, not the negative count.

### Holdout size

250–300 plants gives recall@1 standard error around ±3pp. Tighter needs a
bigger holdout, which costs training data. Re-examine only if the holdout is
ever rebuilt — which should be approximately never.

### `K_RINGS`

Currently 18 (≈5.4 km radius). Mean correction distance is ~5.05 km, so the
average correction lands near the window boundary. `08` puts the ceiling at
**91%** of corrections having their true parcel inside the ring at all.

`candidate_recall_failures.parquet` — now produced locally by
`update_master_locations.py` — accumulates reviewer-confirmed cases where the
truth was outside the pool. That is the empirical input to this decision.
`08b_analyze_ring_misses.py` answers whether raising it would help.

### `MAX_PARCEL_AREA_M2`

2 km² is explicitly a judgment call, not a measured value. The docstring
says how to set it properly: from the area distribution of parcels matched to
already-verified-**Correct** plants. Worth doing once there is more than one
state's worth of those.

---

## Gaps

### `Duplicate_Parcel_Flag` is ignored

It was computed without restricting the OSM/parcel intersect to
treatment-plant-type parcels, so it is not a reliable signal. Recomputing it
correctly might produce a useful feature. Currently dropped.

### Corrected parcels outside the k-ring are dropped from Stage 2b

`06` reports the count and drops them rather than triggering a targeted NLCD
top-up. `01d_nlcd_topup.py` exists to fix this. Deliberately not wired up
until the count is known to matter — check `06`'s log.

### `14_/15_*.parquet` path-collision risk

Training table output paths are not scope-tagged, so a full-universe pilot and
a nationwide retrain can overwrite each other's tables. Fine while these are
one-off events; needs a structural fix before they become routine interleaved
operations.

### Reported-parcel exclusion and candidate-competition resolution

These existed in the original R pipeline and have no home in the Python one.
They belong somewhere between Stage 2a/2b scoring and final output assembly.
`check_competition_losses.py` is the diagnostic that was written around this.

### `pull_round.sh` has never worked

Template with placeholder host and paths. Either fix it for the real access
method or delete it so it stops looking like a working tool.

### The master is backed up by git, on a personal account

Resolved in part (2026-09-21): `correction/data/training/Updates.gpkg` is
tracked, so it has a backup, version history, and cross-machine sync.

What remains: the repository is under a personal GitHub account, which moves
the account-deprovisioning risk rather than removing it. A repo under an EPA
organization, or a copy somewhere the team controls, would close it properly.
See [08_HANDOFF.md](08_HANDOFF.md).

---

## Cleanups

### Retire the one-off patch scripts

`patch_01e_training_scope.py`, `patch_05_for_array.py`,
`01e_run_od_candidates.py.bak-20260908-124200`,
`05_run_inference.py.bak-20260908-083146`, `build_tasks_patch.diff`,
`naip_fetch_patch.diff`, `PATCH_holdout_antijoins.md`.

All already applied. Git history holds the record. They currently read like
things you might need to run.

### Prune the `diagnose_*` scripts

About 15 of them, most written for a single incident. Several are still
genuinely useful (`check_od_freshness.py`, `diagnose_candidate_od.py`,
`inspect_model_features.py`, `08`/`08b`). The rest could move to an
`archive/` subdirectory so the useful ones are findable.

### Reconcile the stale documentation

`detection/PROJECT_ANCHOR.md` describes the May 2026 all-R pipeline —
`app.R`, `Inspection.gdb`, `build_results.R`. `REVIEW_LOOP_PLAN.md` Phase 4
describes extending `app.R`, which never happened (the app was built in
Python). `correction/README_HPC.md`'s run order references a `slurm/`
subdirectory that doesn't exist.

Either update them or mark them `HISTORICAL` at the top. The
[docs/README.md](README.md) status table is a stopgap.

### The committed logs and artifacts

1,731 `.txt`, 485 `.log`, 128 `.png` files are committed to the repo. Some
are real inputs; much is run output. Worth a pass with fresh eyes on what
belongs in git.

### `review_log_candidates_round{N}.parquet` feeds nothing

Deliberate — see [04_REVIEW_LOOP.md](04_REVIEW_LOOP.md). Keep it as a record
of what was on screen, or delete the export. Do not wire it into training;
`06b` reconstructs a strictly richer version.

---

## What I would do first, taking this over

1. **Get the master somewhere the team controls.** It is in git now, which
   is a real backup, but the repo is on a personal account — so the
   deprovisioning risk has moved rather than gone.
2. **Run one full cycle end to end** on a small state list, using
   [05_RUNBOOK.md](05_RUNBOOK.md), to confirm your access and environment
   work before you need them to.
3. **Confirm the re-ranker actually improved** after the next round —
   `12_score_holdout.py`, compared round over round. That validates the whole
   feedback loop, and it is the first cycle where review-derived hard
   negatives flow through cleanly.
4. Then cleanups, in whatever order annoys you most.
