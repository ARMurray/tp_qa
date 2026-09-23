# Resume here — snapshot, 2026-09-23

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
| Tile dedup + purge | `detection/pipeline/dedupe_tiles_by_content.py` |
| Targeted tile selection | `extract_review_tiles.py` picks by verdict + detections |
| Detection stats in review | candidate cards show objects / max confidence |
| NAIP tiles untracked | repo stopped growing; history not yet rewritten |

The pieces are verified individually against real data wherever that was
possible locally. **None of it has run end-to-end on the HPC or through a
real review round.** That is what the next session is for.

---

## 1. Reset the labelling inventory — ✅ DONE 2026-09-23

```
6,814 tiles -> 1,003
   35 exact duplicates
5,776 never labelled
```

A follow-up `--near-duplicates` pass over the survivors found **zero** near
matches too. So duplication was never the problem — 35 files out of 6,814,
all inside the labelled subset. Volume was, and 5,776 tiles had been fetched,
stored and never looked at.

The impression that the inventory was full of duplicates came from two things
that are not duplication: one plant (`08000000031`) contributing 27 of the 35,
which clusters while labelling and feels like many more, and thousands of
empty fields that look alike without being the same tile. That is the
argument for targeted selection over better dedup.

Remaining: run `reconstruct_tile_metadata.py`, since ~5,800 rows of
`tile_metadata.csv` now point at files that are gone.

<details>
<summary>How it was run</summary>

Keep everything labelled, collapse labelled duplicates to one copy, and
remove every tile that was never labelled — the new baseline, now that
future tiles arrive by targeted selection rather than bulk extraction.

```bash
python detection/pipeline/dedupe_tiles_by_content.py --purge-unlabeled
```

Report only. Add `--apply` to act, which **quarantines to
`tiles/_duplicates/`** and writes a manifest of every file moved; `--delete`
opts into real deletion. Start with the report and read the counts.

Verified against the real 1,004-tile set with 600 labels: 565 survive, all
labelled, zero duplicate content, 439 quarantined.

Duplicate keepers are chosen **deterministically** (lexicographically first),
not randomly — the labels are identical so it makes no difference which, and
two runs agreeing is worth more. If two copies were ever labelled
*differently*, that group is refused rather than guessed at.

</details>

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

## 2. Targeted tile selection — built, needs a real round to exercise

Tile selection after a review round no longer takes every candidate parcel.
It takes two things:

- **the verified true location**, however the reviewer established it — the
  selected candidate, the reported parcel, or a clicked lat/lon
- **parcels where the detector fired but the reviewer said it was not the
  plant** — confirmed false positives

Everything else is dropped. A candidate the detector ignored and the reviewer
rejected is one where the model already agrees with you, and labelling it
confirms what it knows. That category was the bulk of the old behaviour, and
it is where 872 of the 1,004 existing labels went.

`needs_info` plants are excluded entirely: with no verdict there is no ground
truth, so a detection on one of their candidates cannot be called a false
positive. It might be the plant.

**Clicked points are now tiled**, which the old code skipped for want of a
parcel — a synthetic square centred on the click, buffered after reprojection
so the half-width means the same thing at every latitude. A verified true
location is worth labelling whether or not Regrid has a polygon for it.

Run against the real `app.db` (300 reviewed plants, rounds 1–2):

```
212 verified true locations (28 from a clicked point)
  0 detections on a parcel that was not the answer
212 sites to tile (was 1,735)
```

That zero is correct: rounds 1–2 were queued before detection results could
be attached, so no false positives are identifiable in them. The script says
so rather than silently reporting zero.

`--all-candidates` restores the old behaviour if you ever need it.

### No per-session budget — settled 2026-09-23

An earlier draft of this file proposed capping each round at ~150 tiles with
a priority allocation across the four quadrants. **Decided against.** The
point is fewer tiles with more impact, not a fixed number of them; if a round
yields 700 tiles and they are all signal, that is a good round. Selection
already does the work that mattered.

Recording it so it does not get reopened as an obvious improvement.

### Detection stats now show in the review app

While deciding, each candidate card shows what the detector found on that
parcel: `3 objects · max 87% · clarifier`.

Three states, not two, and the distinction is deliberate:

| Shown | Means |
|---|---|
| `3 objects · max 87% · clarifier` | detector examined it and found infrastructure |
| `nothing found` | detector examined it and found nothing — this is evidence |
| `not run` | detector never looked — this is **not** evidence |

Collapsing the last two would quietly argue against a parcel that was simply
never examined.

### What has to happen for the false-positive half to work

**`01e` must run before `10_build_review_queue`.** The queue is what carries
detection results to the local machine; if 01e has not run, the candidates
arrive with null detection columns and only verified true locations get
tiled. The script warns when that happens.

This needs no reordering — 01e and 10 both read `data/inference/`, so 01e
slots in between `merge_05_shards` and `10`:

```bash
sbatch --export=SCOPE="train" 01e_run_od_candidates.slurm
# then, after it finishes:
sbatch 10_build_review_queue.slurm
```

### Still unexercised

None of this has run end-to-end on a real round. Specifically untested:
`attach_candidate_od()` against real 01e output, the review UI rendering, and
the synthetic-square tiling of a clicked point actually fetching imagery.
The pieces are verified individually against real data where that was
possible locally.

---

## 3. Run the feature-engineering array

> **Do the detector retrain (section 4) FIRST.** The class list changed on
> 2026-09-23, so the deployed detector is superseded and every detection
> output is stale. Running `02` before re-running `01b` / `01c` / `01e` means
> redoing `02`, all four model trainings, and inference.

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

## 4. Retrain the detector on all six classes — DO THIS BEFORE SECTION 3

`KEEP_CLASSES` in `04_train_model.py` restricted training to
`aeration_basin`, `clarifier`, `digester`. It is now `[]` — all six.

Annotation counts when that changed, across 1,004 label files (132 with
boxes, 872 deliberate empties):

| class | boxes | tiles | was trained |
|---|---|---|---|
| clarifier | 264 | 73 | ✓ |
| **oxidation_pond** | **138** | **67** | ✗ |
| aeration_basin | 105 | 59 | ✓ |
| digester | 82 | 29 | ✓ |
| **chlorine_contact** | **29** | **22** | ✗ |
| **drying_bed** | **10** | **6** | ✗ |

`oxidation_pond` is the one that mattered — more instances than digester,
more tiles than aeration_basin, excluded the whole time, and the dominant
infrastructure at small plants, which is exactly where the correction
pipeline performs worst.

**Watch `drying_bed`.** Six tiles is below where a YOLO class learns
anything, so expect unreliable detections at first. That matters because the
correction models consume `od_has_drying_bed` / `od_n_drying_bed` as
features. It is in so new labels count immediately and because targeted tile
selection now accumulates examples where they occur — but if `07` / `07b`
show it carrying weight before the count is near 25+ tiles, that weight is
noise.

`correction/scripts/config.py`'s `CLASSES` already listed all six and always
has; it fixes the `od_*` feature schema independently of what the detector was
trained on. Three of those columns have simply been permanently False.
Nothing changes there.

```bash
cd detection
python pipeline/03_prepare_dataset.py    # only if you have labelled since the last run
python pipeline/04_train_model.py
```

Empty `KEEP_CLASSES` trains straight from `dataset.yaml` with no filtered
copy, which also stops the ~1.2 GB duplication into `dataset_filtered/`.

`04` deploys `best.pt` itself on success — no manual copy. Commit and push
that file so the HPC picks it up.

**Then re-run `01b`, `01c`, `01e` and `check_od_freshness.py` before `02`.**
A new detector makes every existing detection output stale, and mixing two
models' output in one feature table is silent. This does mean losing the
`01b`/`01c`/`01e` runs done with the 3-class model — unavoidable, and far
cheaper now than after `02` plus four trainings.

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
