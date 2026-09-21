# 4. The review loop

This is the part that compounds. Everything else is machinery around it.

---

## What a reviewer actually does

The app (`review_app/`) shows one plant at a time on NAIP imagery, with the
reported location and up to 5 ranked candidate parcels drawn on a Leaflet
map, plus context (owner, LBCS codes, acreage, population served, discharge
flags). The reviewer decides.

Two task types, assigned by `10_build_review_queue.py`:

- **`candidate_pick`** — Stage 1 flagged the reported location as probably
  wrong, and Stage 2a produced candidates. Pick one, or say the reported
  point was right after all, or say the truth isn't shown.
- **`confirm_reported`** — no candidates exist (Stage 1 passed it, or nothing
  survived candidate generation). Confirm or reject the reported point.

## The four verdicts

| Verdict | Meaning | What it becomes |
|---|---|---|
| `reported_correct` | CWNS was right | `classes`: **Correct** |
| `candidate_correct` | one of the shown parcels is it | `classes`: **Incorrect** + a `corrections` row |
| `truth_outside_candidates` | it's neither the reported point nor anything shown; reviewer clicked the map | `classes`: **Incorrect** + a `corrections` row + a recall-failure record |
| `needs_info` | can't tell right now | nothing — stays unverified |

`backend/models.py` enforces which fields each verdict requires.
`candidate_correct` needs `selected_ll_uuid` + `candidate_rank`;
`truth_outside_candidates` needs `truth_latitude`/`truth_longitude`;
both confirmations need `confirmation_type` (`independent` vs
`confirmed_proposal` — whether the reviewer reached the answer themselves or
agreed with what the model proposed).

`truth_rank` distinguishes "found but mis-ranked" from "not in the pool at
all". It must never silently default to the same value in both cases.

---

## Where the corrected coordinate comes from

This is the subtlety that matters most, and it is easy to get wrong.

**The app never captures a coordinate for a `candidate_correct` verdict.** It
captures a *parcel id* (`selected_ll_uuid`). The `plants.latitude` /
`plants.longitude` columns in `app.db` are the **reported** location, loaded
from the queue by `queue_loader.py` — they are not the answer.

So the corrected coordinate is derived:

| Verdict | Corrected_X/Y source |
|---|---|
| `candidate_correct` | `ST_PointOnSurface` of the selected parcel, looked up live against the local Regrid mirror |
| `truth_outside_candidates` | the reviewer's clicked point, already exact |
| `reported_correct` | none — the reported point stands |

**`ST_PointOnSurface`, not `ST_Centroid`.** A polygon's geometric centroid can
fall *outside* the polygon for concave, L-shaped, or multipart parcels, which
Regrid has plenty of. `PointOnSurface` guarantees a point inside, at the cost
of being less "central" for a convex shape. A training coordinate outside its
own parcel is a correctness bug; one that is merely off-center is not.

> A previous R prototype (`pull_reviews.R`, since replaced) wrote
> `Corrected_X = longitude` — the *reported* point — for every
> `candidate_correct` verdict. That records "this was wrong, and here is the
> right answer" where the right answer is the wrong answer. If you ever see
> `Corrected_X == Original_X` on a row that claims to be corrected, this is
> the bug that produced it.

---

## Closing a round

One command:

```bash
cd review_app
python -m sync.close_round --round N --dry-run   # see what it will do
python -m sync.close_round --round N
```

It runs four steps and then prints exactly what to upload and what to run on
the HPC.

| Step | Script | Produces |
|---|---|---|
| 1 | `push_review_log.py` | Verdict parquet (audit trail), and routes holdout verdicts to a separate file |
| 2 | `update_master_locations.py` | **A new dated layer in the master** + recall-failure diagnostics |
| 3 | `build_training_bins.py` | `training_locations.gpkg` — the file that actually crosses to HPC |
| 4 | `extract_review_tiles.py` | NAIP tiles into `detection/data/tiles/` |

Every step is idempotent. Re-running is safe; step 2 writes a `_vN`-suffixed
layer rather than overwriting a same-day one.

Useful flags: `--skip-tiles` (step 4 fetches imagery and is slow — skip it
when re-running to fix something downstream), `--skip-bins`, `--all`
(re-export everything rather than only what's new).

### The master layer convention

`CWNS_Locations_YYYYMMDD`, with `_vN` for same-day re-runs. `YYYYMMDD` sorts
lexicographically in date order, which is what makes "the newest layer" a
reliable question to ask. `build_training_bins.py` resolves it automatically
via `config.latest_master_layer()`, which sorts by **parsed date** and raises
if nothing matches, rather than falling back to whatever the driver listed
first.

Set `config.MASTER_LAYER` to a specific name to freeze training labels at a
known version — reproducing an old run, or bisecting a regression.

---

## How a verdict reaches each model

All four models read `training_locations.gpkg`. There is one derivation path.

```
verdict in app.db
   └─ update_master_locations.py
        └─ Updates.gpkg :: CWNS_Locations_YYYYMMDD
             └─ build_training_bins.py
                  └─ training_locations.gpkg
                       ├─ classes      ─▶ Stage 1  (03)
                       ├─ corrections  ─▶ Stage 2a (04)
                       ├─ corrections  ─▶ Stage 2b (06 ─▶ 07)
                       └─ corrections  ─▶ re-ranker (06b ─▶ 07b)
```

### The re-ranker is where review pays off most

`06b_build_rerank_training.py` resolves each correction's `Corrected_X/Y` to
its containing parcel, labels that parcel **1**, and labels the other ~19
Stage 2a candidates **0**.

So **one `candidate_correct` verdict becomes one positive and ~19
distribution-matched hard negatives**, automatically, with no extra plumbing.
Those negatives are the valuable part: every one of them is a parcel Stage 2a
ranked highly, so they teach the model to discriminate in exactly the
near-tie situation it faces at deployment.

`truth_outside_candidates` verdicts are valuable too. 06b keeps those plants
with 20 negatives and no positive, flagged via `pool_has_positive`. They are
real — they teach what a near-miss looks like, and dropping them would train
the model on an easier world than the one it is deployed into.

### One thing you must not forget

> **`06b` only sees a plant's candidates if `01e_run_od_candidates.py` has run
> OD on them.** It silently drops any plant whose candidates have no OD
> output. Skipping `01e` is the way a round's hard negatives quietly fail to
> reach the re-ranker, with no error anywhere.

`close_round.py` prints this reminder. It is step 2 of the HPC sequence in
[05_RUNBOOK.md](05_RUNBOOK.md).

### Why `review_log_candidates_round{N}.parquet` feeds nothing

`push_review_log.py` exports the candidate-level rows the reviewer saw. That
file is **deliberately not** wired into training.

The app shows `TOP_K_SHOWN = 5` candidates. `06b` reconstructs the full
**top-20** pool from `stage2_candidates.parquet`. The reconstruction is
strictly richer, so feeding the export as well would add nothing and create a
second path for the same information to reach training — the exact thing the
2026-09-21 consolidation removed. The export is kept as a record of what was
on screen when the reviewer decided.

---

## Holdout handling in the review loop

Holdout plants **are** reviewed — the app shows an orange banner — but their
verdicts route differently at every step:

- `push_review_log.py` writes them to `holdout_truth_round{N}.parquet`, never
  to the training export.
- `update_master_locations.py` excludes them from the master by default
  (`--include-holdout` overrides; read that docstring first).
- The merge into `holdout_truth.parquet` on HPC is manual.

The reviewer is not asked to review them any differently. That is the point —
a holdout verdict has to be produced the same way as any other to be a valid
evaluation.

---

## Known limitations of the app

- **No undo.** `POST /api/verdict` returns 409 on a second submission rather
  than overwriting. To correct a mis-click:
  `UPDATE plants SET reviewed = 0 WHERE cwns_id = '...'` directly in SQLite.
- **No out-of-order review.** `GET /api/plants/{cwns_id}` exists as a raw
  endpoint; there is no UI for it.
- **`pull_round.sh` is a template** with placeholder host and paths, never
  confirmed working.
