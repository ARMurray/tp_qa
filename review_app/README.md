# tp_qa Review App

Local review app for treatment plant location corrections. Runs entirely on
your machine -- no HPC connection needed while reviewing. Parcel geometry is
looked up live via DuckDB against your local Regrid parquet mirror.

## Setup (one time)

```
pip install -r requirements.txt
```

Check `config.py`'s `REGRID_STATE_GLOB` matches your local Regrid folder
layout before your first review session -- see the comment there. This is
the one thing I couldn't verify without seeing your actual folder structure.

## Each review round

1. **Pull the round's queue down from HPC.** Either run `sync/pull_round.sh N`
   (after fixing the host/path placeholders in it for your actual HPC access
   method), or just manually copy these two files into `data/incoming/`:
   - `review_queue_round{N}.parquet` (from `10_build_review_queue.py`'s output)
   - `holdout_manifest.parquet` (from `data/holdout/` on HPC)

2. **Load the queue into the local database:**
   ```
   python -m backend.queue_loader --round N
   ```
   Refuses to re-run for a round already loaded, to protect in-progress
   reviews -- pass `--force` only if you actually want to wipe and reload.

3. **Start the app:**
   ```
   uvicorn backend.app:app --reload --port 8000
   ```
   Open http://localhost:8000

4. **Review.** Enter your name once (saved locally). The queue serves
   holdout plants first, then uncertain/random in whatever order they were
   loaded. Each plant is one of two task types:
   - **candidate_pick** -- pick one of up to 5 ranked candidates, say the
     reported location was actually right, or say the truth isn't shown at
     all (click the map to mark it).
   - **confirm_reported** -- no candidates exist for this plant (either Stage 1
     passed it as correct, or nothing survived candidate generation). Confirm
     or reject the reported point directly; the "truth isn't shown" path is
     still available if you can tell it's wrong from the imagery.

   A holdout plant shows an orange banner -- its verdict routes to
   `holdout_truth`, not the training feed. This matters for your workflow but
   the app doesn't ask you to review any differently.

5. **Export verdicts back out**, any time, as often as you like (safe to
   re-run -- only exports what's new since the last export unless you pass
   `--all`):
   ```
   python -m sync.push_review_log --round N
   ```
   Writes two files to `data/outgoing/`:
   - `review_log_round{N}.parquet` -- non-holdout verdicts. Input to the
     not-yet-built `11_ingest_review_log.py`.
   - `holdout_truth_round{N}.parquet` -- holdout verdicts. Needs manual
     merging into HPC's `holdout_truth.parquet` -- **do not** feed this into
     training.

6. **Sync `data/outgoing/*.parquet` back to HPC** (reverse of step 1 -- no
   script for this yet, same manual/scp process).

## What's NOT built yet

- `10_build_review_queue.py`'s round 2+ behavior for holdout: after round 1,
  holdout plants should stop appearing in the queue entirely (per
  REVIEW_LOOP_PLAN.md Phase 4 #3). The queue builder already handles this
  (`--round` > 1 skips the holdout slice) -- nothing to do here, just noting
  it so it's not mistaken for a gap in the app.
- `11_ingest_review_log.py` on HPC. This app's export is designed to be a
  reasonable input to it, not a guess at its exact final schema.
- Any way to review a specific plant out of the queue's own order, beyond
  `GET /api/plants/{cwns_id}` existing as a raw endpoint. No UI for it yet.
- Undo / re-review. `POST /api/verdict` refuses a second submission for an
  already-reviewed plant (409) rather than overwriting -- there's no admin
  path to correct a mis-click yet. If that happens, it needs a direct SQLite
  edit for now: `UPDATE plants SET reviewed = 0 WHERE cwns_id = '...'`.

## Project layout

```
review_app/
├── config.py              Local paths -- Regrid mirror, app.db, sync dirs
├── backend/
│   ├── app.py             FastAPI entrypoint
│   ├── db.py               SQLite schema + queries
│   ├── models.py            Verdict payload validation
│   ├── parcels.py            Live DuckDB lookups against local Regrid
│   ├── queue_loader.py        Imports a round's queue parquet -> SQLite
│   └── routes/
│       ├── plants.py           GET next plant / status / by-id
│       └── verdicts.py          POST a verdict
├── frontend/
│   ├── index.html
│   └── static/{js,css}/     Leaflet map + review flow, vanilla JS, no build step
├── data/
│   ├── incoming/            Synced FROM HPC
│   ├── app.db                Local SQLite -- your live review state
│   └── outgoing/              Exported for syncing back TO HPC
└── sync/
    ├── pull_round.sh          Template -- adjust host/path for your HPC access
    └── push_review_log.py      app.db -> outgoing/*.parquet
```
