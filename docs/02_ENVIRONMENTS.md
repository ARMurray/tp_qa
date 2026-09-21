# 2. Environments, paths, and access

Two machines. They share nothing automatically — every transfer between them
is manual.

---

## The local workstation (Windows)

Runs `detection/` and `review_app/`. This is also where the **master
locations file** lives, and it is the only copy.

### Python environments

| Purpose | Environment | Notes |
|---|---|---|
| `review_app/` | `pip install -r review_app/requirements.txt` | FastAPI, DuckDB, pandas, geopandas, pyogrio |
| `detection/` | `detection/.venv` (gitignored) | Adds `ultralytics` (YOLOv8) and `torch` |

`geopandas` + `pyogrio` are needed only by
`sync/update_master_locations.py` and `build_training_bins.py`, not to run
the review app itself.

### Paths — all configured in `review_app/config.py`

| What | Path |
|---|---|
| Regrid parcel mirror | `C:\Users\AMURRA02\OneDrive - Environmental Protection Agency (EPA)\Data\Regrid\Parquet_Storage` |
| Layout within it | `state={state}/*.parquet` (`REGRID_STATE_GLOB`) |
| CWNS facility names | `...\Github\Sewersheds\Data\FACILITIES.txt` |
| **Master locations** | `...\Github\Location_Correction\data\Updates.gpkg` |
| Review database | `review_app/data/app.db` |
| Synced in / out | `review_app/data/incoming/`, `review_app/data/outgoing/` |

> **`REGRID_STATE_GLOB` is the first thing to check** if parcel lookups come
> back empty. `config.py` carries a long comment about this — it was never
> verified against the real local folder layout by the person who wrote it.

> **`Updates.gpkg` is not in this repo and not in any backup this project
> controls.** It is the accumulated output of months of manual verification.
> See [08_HANDOFF.md](08_HANDOFF.md) — protecting this file is the single
> highest-value handoff action.

### Local storage note

`detection/data/` and `correction/data/` are both `.gitignore`d. The repo
holds code and documentation only. Cloning it gets you nothing runnable until
you also have the data.

---

## The HPC cluster

Runs `correction/` only.

| | |
|---|---|
| Root | `/work/GRDVULN/tp_qa/correction` |
| Scheduler | SLURM |
| Account | `grdvuln` |
| Shared data | `/work/GRDVULN/data/{parcels,nlcd,Census}` |
| Venv | `$ROOT/.venv`, built once by `scripts/setup_env.sh` on a **login node** |
| Host alias in scripts | `atmos3` (in `sync/pull_round.sh` — a placeholder, confirm it) |

`TODO(handoff)`: the access method — VPN, SSH host/alias, key vs password,
whether MobaXterm is required or just convention — is not recorded anywhere
in this repo. See [08_HANDOFF.md](08_HANDOFF.md).

### Environment setup, first time

```bash
bash /work/GRDVULN/tp_qa/correction/scripts/setup_env.sh
```

Run on a **login node**, once. Every SLURM wrapper sources the venv this
creates rather than building its own.

Two things this script exists to work around, both of which cost real time
to discover:

- The cluster's default `python3` on compute nodes is **3.6**, far too old —
  it resolves 2020-era wheels and chokes on modern syntax. `setup_env.sh`
  and `_common.sh` both run a module-load loop (`python/3.11` → `3.10` →
  `3.9` → bare names) to find a real Python first.
- Building a venv inside each job wastes allocation minutes and can silently
  resolve against that stale system Python.

If `module avail python` shows a version whose name isn't in the loop, add it.

### Every job sources `_common.sh`

```bash
source /work/GRDVULN/tp_qa/correction/scripts/_common.sh
```

It loads the module, activates the venv, asserts Python ≥ 3.9, prints node /
job / CPU info into the log, and sets `pipefail` so a failing step fails the
job instead of writing partial output that looks successful.

### Directory layout under `$ROOT`

```
scripts/          all .py + .slurm + _common.sh + setup_env.sh (flat, no slurm/ subdir)
logs/             every SLURM .log
.venv/            the shared environment
models/           trained stage1_*/stage2_*/rerank_* models
  object_detection/   best.pt, uploaded from local detection/
data/
  cwns/                 CWNS text exports
  training/             Updates.gpkg (uploaded), training_locations.gpkg (built)
  nlcd_features/        01a output
  od_features/          01b output — OD at REPORTED locations
  od_features_corrected/    01c output — OD at CORRECTED locations
  od_features_candidates/       01e output — holdout candidates
  od_features_candidates_train/  01e output — training candidates
  features/             02 output, plus diagnostics
  inference/            05 output: stage2_candidates.parquet etc.
  holdout/              manifest + truth + scores
  review_queue/         10 output
  reference/            OSM gpkg
```

The `od_features_corrected` / `od_features_candidates*` split is deliberate:
each root is "this plant's OD result at **one specific** location." Keeping
them separate means no reader can accidentally union reported-location and
corrected-location rows for the same `CWNS_ID`. And
`od_features_candidates_train` is separate from `od_features_candidates`
specifically so the re-ranker cannot train on the holdout's OD rows.

---

## Moving data between them

There is no automation. Both directions are manual file transfer.

### HPC → local (starting a review round)

| File | From | To |
|---|---|---|
| `review_queue_round{N}.parquet` | `data/review_queue/` | `review_app/data/incoming/` |
| `holdout_manifest.parquet` | `data/holdout/` | `review_app/data/incoming/` |

`review_app/sync/pull_round.sh N` is a **template** with placeholder host and
paths. It has never been confirmed working. Either fix it for your access
method or copy the two files by hand.

### Local → HPC (closing a review round)

| File | From | To |
|---|---|---|
| `training_locations.gpkg` | `review_app/data/outgoing/` | `correction/data/training/` |
| `candidate_recall_failures.parquet` | `review_app/data/outgoing/` | `correction/data/features/` |
| `holdout_truth_round{N}.parquet` | `review_app/data/outgoing/` | merge **by hand** into `data/holdout/holdout_truth.parquet` |

`python -m sync.close_round --round N` prints this list at the end of every
run, so you do not have to remember it.

> The holdout truth file is merged by hand deliberately. It is evaluation
> data; an automatic append that went wrong would corrupt the only clean read
> on the pipeline, and the corruption would be invisible.

### What does **not** get uploaded

The master `Updates.gpkg` stays local. Only the narrow derived file
(`training_locations.gpkg`) crosses. That was a deliberate change on
2026-09-21 — it is smaller, it is the actual contract the pipeline consumes,
and it means the HPC never holds a second copy of the master that could drift.

`build_training_bins.py` still runs on the HPC if you want it to
(`00_build_training_bins.slurm`), which requires uploading the master. Kept
as a fallback, not the normal path.
