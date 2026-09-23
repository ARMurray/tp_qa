# Resume here — snapshot, 2026-09-22

Point-in-time checklist. **Once these steps are done this file is history** —
[05_RUNBOOK.md](05_RUNBOOK.md) is the durable version and stays correct for
every future round.

*(The 2026-09-21 version of this file covered the round 1–2 fold-back, which
is done: the master carries 2,690 verified locations and 452 corrections.)*

---

## ⚠ READ BEFORE YOU PULL

Pulling on the work machine **will delete your NAIP training tiles.**

`detection/dataset_filtered/` — 2,010 files, 1.17 GB — was committed by
accident and has been untracked. Git applies a removal to any clone that
pulls it, and `.gitignore` does not protect a file that *was* tracked. Move
it aside first:

```powershell
Move-Item detection\dataset_filtered detection\dataset_filtered_keep
git pull
Move-Item detection\dataset_filtered_keep detection\dataset_filtered
```

It is ignored after that, so it stays put from then on. Regenerating it with
`03_prepare_dataset.py` also works, but moving is instant.

---

## Where things stand

You merged to `master`, trained a new detector, and got as far as feature
engineering. Since then, on the repo side:

| | |
|---|---|
| `02` is now a per-state array | writes shards; `02b` merges them |
| `best.pt` deploys itself | `04_train_model.py` copies it to `correction/models/object_detection/` |
| Tile dedup exists | `detection/pipeline/dedupe_tiles_by_content.py` |
| NAIP tiles untracked | repo stopped growing; history not yet rewritten |

None of the new pipeline code has run against real data yet. That is what
tomorrow is for.

---

## 1. Deduplicate the labelling inventory

```bash
python detection/pipeline/dedupe_tiles_by_content.py
```

Report only. `--apply` moves duplicates to `tiles/_duplicates/` and writes a
manifest; `--delete` opts into real deletion. Start with the report.

### What I measured, and what it means

Across the 1,004 labelled tiles in `dataset_filtered`:

| | |
|---|---|
| Redundant | 35 of 1,004 (**3.5%**) |
| Cross-plant duplicates | **0** |
| Same plant, different parcel uuid | **17 of 17 groups** |
| byte / pixel / perceptual hash | all agree exactly |

**Your hypothesis was not the cause here.** It is not multiple plants sharing
an area — it is one plant having several Regrid parcel records for the same
ground. `08000000031` has four parcel uuids over the same site, each tiled
separately into the same nine images: 27 redundant tiles from one plant.

Also measured: no train/val leakage, and all 17 duplicate groups carry
identical annotations, so nothing needs reconciling by hand.

**Caveat.** That is the *labelled* subset, and `dataset_filtered` may already
have been filtered once. Your ~5,000 unlabelled tiles were never measured, and
cross-plant duplication is still plausible there. The script reports the
same-plant / cross-plant split rather than assuming my numbers hold — read
that line in the output before `--apply`.

Afterwards, run `reconstruct_tile_metadata.py`: dedup does not rewrite
`tile_metadata.csv`, so removed tiles leave rows pointing at absent files.

### One number worth sitting with

**872 of your 1,004 label files are zero bytes** — 87% of the labelling you
have done is on tiles containing nothing. Empty labels are real and necessary
training signal, but at that ratio, dedup is not your main lever. Selection
is. Which brings us to:

---

## 2. Filtering the labelling pool by detection hits — your idea, and I think you're right

You proposed using detection hits to decide which tiles enter the labelling
pool after each review round, instead of taking every candidate. That is
hard-negative mining, and it is the right instinct. Here is the shape of it,
plus the one thing that blocks it today.

### The four quadrants

For every candidate parcel of every reviewed plant, two facts are known: did
the detector fire there, and did the reviewer say it was the plant.

| | Reviewer: **is** the plant | Reviewer: **not** the plant |
|---|---|---|
| **Detector fired** | true positive — some value | **FALSE POSITIVE — the gold** |
| **Detector silent** | **FALSE NEGATIVE — the gold** | true negative — teaches least |

The false positives are exactly what you said you value: tiles that should be
empty where the detector sees infrastructure anyway. The false negatives are
the mirror — real plants it walked past.

The bottom-right cell is the bulk, and it is where the 5,000 came from. Those
are parcels the detector ignored and the reviewer rejected: the model already
agrees with you. Labelling them confirms what it already knows, and it is
where most of those 872 empty labels went.

### Proposed rule

Take every FP and FN, take TPs, and **cap** true negatives at a sample rather
than all of them. YOLO does need background examples — but you already have
872, and thousands more have sharply diminishing returns.

If the quadrants split anything like I would expect, this cuts the per-round
pool by most of its volume while *improving* what is in it.

### What blocks it today

**The detection results never reach the review app.** I checked:

- `10_build_review_queue.py` has zero `od_` references
- `app.db`'s `candidates` table has no OD columns — `stage2a_score`,
  `stage2b_score`, `distance_m`, LBCS fields, but nothing about detections

The data exists. `01e_run_od_candidates.py` writes
`od_features_candidates_train/` keyed by `(CWNS_ID, ll_uuid)` with `od_ran`,
`od_has_detection`, `od_n_objects` and the per-class counts. It just never
crosses to the machine where tile selection happens.

So this is a three-hop plumbing change, not a one-script change:

1. `10_build_review_queue.py` joins 01e's candidate OD output onto the
   candidate rows and carries `od_has_detection` / `od_n_objects` into the
   queue parquet
2. `queue_loader.py` adds them to `CAND_COLS` and the `candidates` schema
3. `extract_review_tiles.py` gains the quadrant filter

Step 1 has a scheduling wrinkle worth knowing before committing to this:
`01e` currently runs with `SCOPE="train"` *after* a review round, to feed the
re-ranker. For the queue to carry OD hits, candidate detection has to have run
**before** the queue is built — which is `SCOPE="holdout"` or a full run at
inference time. Worth checking whether the timing already works out for the
plants that end up in a queue, or whether 01e needs to run earlier in the
cycle.

**Nothing here is built.** Say the word and I will, but I would rather you
look at the quadrant table first and tell me whether the keep/drop split
matches how you actually want to spend labelling time.

---

## 3. Run the feature-engineering array

Not yet exercised against real data.

```bash
cd /work/GRDVULN/tp_qa/correction/scripts

JID=$(sbatch --parsable 02_feature_engineering.slurm)
sbatch --dependency=afterok:$JID 02b_merge_feature_shards.slurm
```

Each task writes a shard under `data/feature_shards/state=XX/`; `02b` unions
them into the flat files 03/04/05/06/06b/10 read by name. `afterok` means the
merge runs only if every task succeeded — which is when merging is safe, since
a merge over a partial array produces a well-formed file that silently omits
states.

**Check `02b`'s log before training anything.** It prints row counts per
table. The failure it is built to catch:

```
MISSING shard for N state(s): [...]
```

That means those array tasks failed. It refuses and writes **nothing**, so the
existing flat files are untouched. Re-run just those tasks
(`sbatch --array=35 02_feature_engineering.slurm`), then re-merge.

Memory dropped 200G → 64G, since one state is a fraction of the national
parcel set. If CA or TX gets OOM-killed, raise it — check `seff <jobid>`
rather than guessing.

Then the usual:

```bash
sbatch 03_train_stage1.slurm
sbatch 04_train_stage2.slurm
sbatch 06_build_stage2b_training.slurm    # then 07_train_stage2b.slurm
sbatch 06b_build_rerank_training.slurm    # then 07b_train_rerank.slurm
sbatch 12_score_holdout.slurm
```

---

## 4. The new detector

`04_train_model.py` now deploys `best.pt` itself on success — no manual copy.
Commit and push that file so the HPC picks it up.

**A new detector makes every existing detection output stale.** Re-run `01b`,
`01c` and `01e`, then `check_od_freshness.py`, *before* `02`. Mixing two
models' output in one feature table is silent.

> The deploy uses `shutil.copy`, not `copy2`, deliberately. `copy2` preserves
> the source mtime, which would make a brand-new model look old to
> `check_od_freshness.py` — the same trap its docstring warns about for
> `scp -p`. If you ever copy a model by hand, `touch` it afterwards.

---

## Still open

**The 1.17 GB is still in history.** Untracking stopped the growth; `.git`
stays at 1.2 GB and every fresh clone pays for it — which matters most for
whoever picks this up after you. Fixing it needs `git filter-repo` and a
force-push: everyone re-clones, and any existing clone (including the HPC)
gets reset rather than pulled. Five commits deep is much cheaper than fifty.

**`correction/temp/` holds seven tracked files** — six `07b_rerank_*.log` and
`outputs.gpkg` — from before that directory was ignored. Harmless, but they
will never update. Untrack them when convenient.

**The master is on a personal GitHub account.** Still the handoff gap. See
[08_HANDOFF.md](08_HANDOFF.md).
