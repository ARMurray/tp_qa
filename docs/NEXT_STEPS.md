# Resume here — snapshot, 2026-09-21

Point-in-time checklist for picking up the round 1–2 fold-back on the HPC.
**Once these steps are done this file is history** — [05_RUNBOOK.md](05_RUNBOOK.md)
is the durable version, and it stays correct for every future round.

---

## Where things stand

Rounds 1 and 2 are reviewed (300 plants), folded into the master, and pushed.

| | |
|---|---|
| Master | `correction/data/training/Updates.gpkg`, layer `CWNS_Locations_20260921`, tracked in git |
| `Verified = Yes` | **2,690** (was 2,494) |
| Corrections bin | **452** (was 316) — 113 from `candidate_correct`, 28 from `truth_outside_candidates` |
| Training bins | already built: 2,690 classes / 452 corrections / 29,606 unverified |
| Branch | `review-loop-consolidation`, **not yet merged to `master`** |

`build_training_bins` already ran as step 3 of `close_round`, so
`review_app/data/outgoing/training_locations.gpkg` exists on the work machine.
Nothing left to do locally.

---

## Step 0 — decide how the HPC gets the code

**First find out whether the HPC copy is a git clone.** It changes everything
downstream, and the repo's own history suggests scripts were originally copied
across by hand rather than cloned.

```bash
cd /work/GRDVULN/tp_qa
git rev-parse --is-inside-work-tree 2>/dev/null && git branch --show-current
```

- Prints `true` and a branch name → **it's a clone.** Follow 0A.
- Prints nothing, or an error → **it's not.** Follow 0B.

### 0A — it IS a clone

Check for local edits before switching, because a branch change will refuse or
clobber:

```bash
cd /work/GRDVULN/tp_qa
git status
```

If anything under `scripts/` is modified, decide what to do with it first —
`git stash` to park it, or commit it. Then:

```bash
git fetch origin
git checkout review-loop-consolidation
git pull
```

If `git checkout` complains that local changes would be overwritten:

```bash
git stash push -m "hpc local edits before branch switch"
git checkout review-loop-consolidation
git pull
git stash pop        # then reconcile by hand
```

**Your data is safe.** `correction/data/` is gitignored apart from the master
file, so switching branches does not touch `od_features/`, `features/`,
`nlcd_features/`, `holdout/`, `inference/`, the venv, or any trained model.

**And the clone makes this much simpler.** The master comes down with the pull,
so you do not need to upload anything — skip to 1A.

### 0B — it is NOT a clone

Two options.

**Make it one** (worth doing once; every future round becomes a `git pull`):

```bash
cd /work/GRDVULN/tp_qa
git init
git remote add origin https://github.com/ARMurray/tp_qa.git
git fetch origin
git checkout -b review-loop-consolidation origin/review-loop-consolidation
```

`git checkout` will refuse if untracked files would be overwritten by tracked
ones. That is the scripts you uploaded by hand. If the refusal lists only files
identical to what is in the repo, `git checkout -f` is safe. If it lists files
you have edited on the cluster and not pushed, **stop and copy those off
first** — that is real work that exists nowhere else.

**Or keep copying by hand.** Upload the changed scripts from
`correction/scripts/` to `$ROOT/scripts/`. The ones that changed in this branch:

```
config.py                    build_training_bins.py
00_build_training_bins.slurm 10_build_review_queue.py
build_test_bundle.py         setup_env.sh
```

And **delete** these two on the HPC — they were removed in this branch and will
otherwise sit there looking runnable:

```bash
rm /work/GRDVULN/tp_qa/correction/scripts/11_ingest_review_log.py
rm /work/GRDVULN/tp_qa/correction/scripts/11_ingest_review_log.slurm
```

---

## Step 1 — get the new labels onto the HPC

### 1A — if the HPC is a clone

The master arrived with the pull, so rebuild the training bins on the cluster:

```bash
cd /work/GRDVULN/tp_qa/correction/scripts
sbatch 00_build_training_bins.slurm
```

Then copy the recall diagnostics into place — also already on disk from the
pull, no upload needed:

```bash
cp /work/GRDVULN/tp_qa/review_app/data/outgoing/candidate_recall_failures.parquet \
   /work/GRDVULN/tp_qa/correction/data/features/
```

Check `logs/00_bins_*.log` before continuing. It should say:

```
Resolved newest master layer: CWNS_Locations_20260921
  classes     : 2690 rows (2238 Correct / 452 Incorrect)
  corrections : 452 rows
  unverified  : 29606 rows
```

**If the corrections count is 316, you are on the old master** — the pull did
not bring the new one, or an older layer got resolved. Do not continue.

### 1B — if the HPC is not a clone

Upload both from the work machine:

| From | To |
|---|---|
| `review_app/data/outgoing/training_locations.gpkg` | `correction/data/training/` |
| `review_app/data/outgoing/candidate_recall_failures.parquet` | `correction/data/features/` |

No `sbatch 00_build_training_bins.slurm` needed — you are uploading the
already-built result.

---

## Step 2 — the HPC run order

```bash
cd /work/GRDVULN/tp_qa/correction/scripts
```

**Which states?** Run this first; its output is the list `02` needs:

```bash
python list_training_states.py
```

### 2.1 — detection at the new corrected locations

136 corrections are new to the bin, and none of them have detection output at
their true location yet.

```bash
sbatch 01c_run_od_corrected_locations.slurm
```

### 2.2 — detection on the candidate parcels — DO NOT SKIP

```bash
sbatch --export=SCOPE="train" 01e_run_od_candidates.slurm
```

This is the one that quietly costs you the round. `06b_build_rerank_training.py`
drops any plant whose candidates have no detection output, with no error — so
skipping this means every one of the round's ~19-per-plant hard negatives never
reaches the re-ranker, and the only symptom is that the re-ranker doesn't
improve.

2.1 and 2.2 are independent; both can be queued at once. Both resume if they
hit a time limit — just resubmit the identical command.

### 2.3 — preflight, then features

```bash
python check_od_freshness.py
sbatch --export=STATES="OH,PA,..." 02_feature_engineering.slurm
```

`02` takes a **comma-separated** list and is a single job. `01a`/`01b`/`01c`/`01e`
are array jobs taking **space-separated** states. Getting that backwards fails
confusingly.

Check the class counts at the end of `02`'s log before training anything.

### 2.4 — retrain

```bash
sbatch 03_train_stage1.slurm
sbatch 04_train_stage2.slurm

sbatch 06_build_stage2b_training.slurm     # then, after it finishes:
sbatch 07_train_stage2b.slurm

sbatch 06b_build_rerank_training.slurm     # then, after it finishes:
sbatch 07b_train_rerank.slurm
```

The two build→train pairs are independent of each other. Within a pair the
build must finish first.

### 2.5 — the honest measurement

```bash
sbatch 12_score_holdout.slurm
```

Compare against the previous round. This is the only number that says whether
any of this worked — everything else is measured on data the models could have
seen.

---

## What to watch for

**`06b`'s log** prints how many candidates had detection output and across how
many plants. If that plant count is far below what you expect, `01e` did not
cover them and the hard negatives are missing.

**`01a` does not need re-running.** It searches k-rings around *reported*
points, which have not changed. The reviewed plants were already in the
universe — they came out of an inference run.

**No `holdout_truth_round2.parquet` exists**, which means all 4 reviewed
holdout plants were in round 1. Nothing to merge this round. Worth confirming
round 1's was ever merged into `data/holdout/holdout_truth.parquet`.

---

## If something looks wrong

| Symptom | Cause |
|---|---|
| `corrections : 316` instead of 452 | Old master. Check `git log` on `correction/data/training/Updates.gpkg`. |
| `No CWNS_Locations_YYYYMMDD layers` | Wrong file at `MASTER_GPKG`, or the migration was never run on that machine. |
| `Holdout manifest not found` | Intended — `exclude_holdout()` is fail-closed. Either the manifest is genuinely missing or you are on the wrong machine. |
| `ERROR: venv not found` | `setup_env.sh` was never run, or was run on a compute node instead of a login node. |
| Python syntax errors in a job | The 3.6 system python. `_common.sh` should prevent this; check its module-load loop found something. |
| `05_run_inference` out of memory | Known. Use `05_run_inference_array.slurm` then `merge_05_shards.py`. |

Fuller list in [06_TROUBLESHOOTING.md](06_TROUBLESHOOTING.md).

---

## Afterwards

Merge the branch once you are satisfied it all runs:

```bash
git checkout master
git merge --ff-only review-loop-consolidation
git push
```

Then this file can be deleted.
