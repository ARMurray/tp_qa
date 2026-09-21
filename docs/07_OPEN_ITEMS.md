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

### No automated backup of the master

See [08_HANDOFF.md](08_HANDOFF.md). This is the most urgent item in this
document.

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

1. **Back up the master.** Nothing else matters if `Updates.gpkg` is lost.
2. **Run one full cycle end to end** on a small state list, using
   [05_RUNBOOK.md](05_RUNBOOK.md), to confirm your access and environment
   work before you need them to.
3. **Confirm the re-ranker actually improved** after the next round —
   `12_score_holdout.py`, compared round over round. That validates the whole
   feedback loop, and it is the first cycle where review-derived hard
   negatives flow through cleanly.
4. Then cleanups, in whatever order annoys you most.
