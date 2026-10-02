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

**Measured 2026-09-24** by `08_diagnose_candidate_coverage.py` over all 452
corrections:

| cause | plants | share |
|---|---|---|
| usable | 413 | **91.4%** |
| (a) outside the k=18 search window | 32 | 7.1% |
| (b) no parcel at the corrected point — Regrid gap | 6 | 1.3% |
| (d) lost in `02` PART 2's join | 1 | 0.2% |

So the ceiling is 91.4%, and the three causes have very different prices. (d) is
one plant and already written to `missing_parcels_topup.csv` for `01d`. (b) is a
data wall. Only (a) carries a real decision, and `08`'s inline suggestion —
"k=238 would capture 95%", at ~175× the candidate pool — is a heuristic its own
author disowns: a 948-ring miss is ~295 km, a plant recorded in the wrong
county, which no radius fixes. Run `08b_analyze_ring_misses.py`, which splits
near misses from records errors, before touching `K_RINGS`.

Note: `candidate_recall_failures.parquet` is a *different* population — it counts
`truth_outside_candidates` review verdicts, i.e. recall failures against the five
candidates the app showed. Useful, but not the cause split.

Worth settling before investing in model architecture: if recall is the binding
constraint, a better classifier cannot reach those plants at all. Deciding that
needs a valid `12_score_holdout` run — see the note below on why the existing
one is unusable.

### The 2026-09-09 `12_score_holdout` log is unusable

It reports `With NO candidates: 50` and `0% candidate recall`, with `nan`
everywhere downstream. That is not a finding about the model. The log's own
command line reads:

```
Running: python /work/GRDVULN/correction/scripts/12_score_holdout.py
```

— missing the `tp_qa` path segment, so it ran from a different tree and found no
inference output. The same typo was still sitting in
`check_01a_01b_complete.sh` until 2026-09-23. The current `12` wrapper sources
the right `_common.sh`, so that log simply predates the fix.

**Superseded:** valid holdout scores now exist (2026-09-24 and 2026-09-30) — see
[RESUME_20261002.md](RESUME_20261002.md). Candidate recall (Stage 2a top-20,
69%) is the binding constraint; the re-ranker gains +7 plants @1 over Stage 2a.

### The project's success measure is precision of automated moves (2026-09-24)

The owner's target: be more than 90% sure that a moved location is right.
recall@k (what 12 reports) is not that. The relevant read is precision of
"move to the re-ranker's #1" at a score cutoff, measured on plants the models
have **not** trained on — i.e. each new review round, not rounds already
folded into training. Round 4 (2026-10-01): 15/16 = 93.8% at score ≥ 0.95,
95% CI 72–99%. Not yet provable; more out-of-sample review at the top of the
score range is what closes the interval.

### Population floor: plants serving > 1,000 people only (2026-09-25)

`config.MIN_POP_SERVED`, applied in 02, 05, 12 and preflight. All 8
zero-population plants in round 3 came back `needs_info`; `needs_info` fell
from 32% (round 3) to 17% (round 4) after the filter. CWNS includes planned
facilities that do not exist yet. The holdout is filtered at scoring time,
never resampled. `--min-pop 0` disables it.

### OSM-tagged reported parcels pass Stage 1 by rule (2026-10-02)

`config.STAGE1_OSM_PASS`. 99.7% of OSM-tagged reported parcels are labelled
Correct and the reviewer treats them as correct, so 05 routes them past Stage
1 (`stage1_route = osm_confirmed`). They stay as Stage 1 TRAINING rows; only
`osm_ww` is dropped as a Stage 1 feature, because it dominated (importance
0.34) and starved owner evidence. The threshold is tuned on untagged plants.
`osm_ww` stays in Stage 2 and the re-ranker. **A rule, not a model finding**
— if labels were partly made by trusting OSM, the 99.7% is partly circular;
12's OSM block and each round's random slice are the independent check.

### Owner-vs-facility-name features: rarity-weighted distinctive tokens (2026-09-25)

`name_match.py`. Chosen by `name_match_benchmark.py` over fuzzy matching and
two embedding models (which rescued similar amounts but broke ~30% of the
pools Stage 2a had right). The name identifies the OWNER, not the parcel, so
pool context (`name_pool_*`, `name_top20_*`) carries "the city owns six
parcels here". Do not swap in embeddings without re-running the benchmark.

### Stage 1 one parcel per plant: lowest ll_uuid (2026-09-29)

A reported point can fall in several overlapping parcels (one plant: 29). 02,
05 and 01b all keep the lowest `ll_uuid`, so parcel and detection features
describe the same parcel. Changing the rule in one place without the others
is a silent mismatch.

## Decisions that need a human

### The move rule and its cutoff

Needed before any corrected output exists: move only if Stage 1 flags the
reported location AND the re-ranker's #1 clears a cutoff. The cutoff has to be
set from out-of-sample precision with a confidence interval, and is the
owner's call. Candidates so far: 0.95 (93.8%, n=16).

### A Stage 1 guardrail for utility-owned reported parcels

Held, deliberately (2026-10-02). Utility-owned reported parcels are 99% Correct
in the labels, and 6 of round 4's 14 false flags were this pattern; the 2
genuine exceptions were large regional utilities owning many facilities.
First see whether the retrained Stage 1 (OSM rule + `utility_owner`) learns it.
If not, add "never flag unless Stage 1 < ~0.03".

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

Currently 18 (≈5.6 km radius, measured). The median correction is 0.91 km,
well inside the window; the mean (3.9 km) is dragged by a long tail.

**`08`'s 91.4% ceiling is overstated** (found 2026-09-24). Its own log says 53
of 452 corrections lie beyond their k=18 ring, but `classify()` labels only 32
as outside the window: it checks whether the corrected parcel is in
`10_parcel_features` before checking the ring, and 01a's correction seeding
(plus neighbouring plants' windows) puts many such parcels into the parcel
features even though 02 never offers them as that plant's candidates. The
realistic ceiling is nearer 87%. `08b` (run 2026-09-24) analysed only the 32,
so its recovery curve understates near misses. **Fix `classify()` to test
candidate-ring membership of the parcel's own `h3_index_9` and re-run 08 + 08b
before any K_RINGS decision.** Decision so far: keep k=18 — k=25 would roughly
double the pool for ~6 recovered plants, and every extra candidate is another
distractor for the precision target.

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

### No corrected-output product yet

Nothing writes the final corrected-locations file. Proposed:
`13_build_corrected_output.py`, one row per CWNS plant with a decision of
`verified` (human, from the master — always wins), `moved` (model, above the
cutoff), `kept` (incl. `osm_confirmed`), or `not assessed` (population ≤ 1,000
or missing; AK/HI/PR, no NAIP). ~9,400 of 16,430 treatment plants are in scope.

### The re-ranker trains on in-sample Stage 2a scores

06b takes `stage2_prob_correct` / `stage2a_rank` from a Stage 2a model fitted
on the same plants, and those are the re-ranker's top two features. On
training plants Stage 2a looks far better than it is (top-20 recall 84–86%
vs 69% on the holdout), so the re-ranker over-trusts it. Fix: out-of-fold
Stage 2a predictions for the training plants. 07b's internal recall@1
(84.6%) vs the holdout's 66.7% is the size of the symptom.

### 01e --from-queue examines only the 5 shown candidates

05b then re-ranks the full top-20 and can promote an unexamined candidate into
the shown five: round 4 had 154 of 660 shown candidates "not run" and 63% of
plants on Stage 2a fallback. Make the queue scope cover each queued plant's
top-20 (~3,000 parcels).

### Detector validation split re-draws whenever plants are added

`detection/pipeline/03_prepare_dataset.py` shuffles the plant list with a seed,
so adding any plant changes most of the val set; old-vs-new detector metrics
are not comparable. A hash-of-CWNS_ID split fixes it (each plant keeps its side
forever). Deferred by the owner until the dataset stabilises.

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
