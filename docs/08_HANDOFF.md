# 8. Handoff

Written 2026-09-21, in anticipation of the original author leaving.

This chapter covers what is **not** recoverable from the code: access,
irreplaceable data, and judgment that only existed in one person's head.

---

## Read this first: the irreplaceable thing

**`Updates.gpkg` — the master locations file.**

```
C:\Users\AMURRA02\OneDrive - Environmental Protection Agency (EPA)
    \Github\Location_Correction\data\Updates.gpkg
```

It is the accumulated result of months of manual verification. Every verified
location, every correction, every round of review is in it.

- It is **not in this repo** (`correction/data/` is gitignored).
- It is **not on the HPC** — only the derived `training_locations.gpkg` is.
- It lives in one OneDrive folder under one departing person's account.

Everything else in this project can be rebuilt from code. This cannot. If the
account is deprovisioned before the file is moved, the project restarts from
zero.

### Action, before anything else

1. Copy it somewhere the team controls and the departing account does not.
2. Copy it a second time somewhere else.
3. Establish whose it is going forward and write that down.
4. `TODO(handoff)`: record the agreed permanent location here.

The same reasoning applies, less urgently, to:

| File | Why it matters |
|---|---|
| `data/holdout/holdout_manifest.parquet` + `holdout_truth.parquet` | Frozen. Regenerating them invalidates every round-over-round comparison ever made. |
| `models/object_detection/best.pt` | Retrainable, but only from the labeled tile inventory. |
| `detection/data/tiles/` + Label Studio annotations | Months of manual labeling. |
| `review_app/data/app.db` | Live review state. Exported verdicts are in `outgoing/`, but in-progress work is only here. |
| Local Regrid parquet mirror | Large; licensed. Confirm the team's license covers a successor. |

---

## Access you will need

`TODO(handoff)` — none of this is recorded anywhere in the repo. The
departing author should fill this in.

| Thing | What to record |
|---|---|
| HPC login | Host, VPN requirement, key vs password, account name. Scripts reference alias `atmos3` and SLURM account `grdvuln` — confirm both. |
| `/work/GRDVULN/` | Who grants access to this allocation? Is there a quota? |
| Regrid data | License terms, who holds the subscription, how the local mirror is refreshed. |
| CWNS source data | Where the text exports come from, and how often they are reissued. |
| Label Studio | Where it runs, credentials, whether the project export is backed up. |
| OneDrive folders | `Location_Correction`, `Sewersheds`, `Regrid` — will these survive the departure? |

---

## Things that are true but not obvious

Collected from the codebase and from the shape of the decisions in it. These
are the ones most likely to be re-learned the hard way.

**The review loop is the product, not the model.** Models get retrained and
replaced. The accumulated verified-location dataset is what the project is
actually building, and it compounds. If you are ever choosing between "improve
the model" and "keep reviewing," review.

**There is exactly one path from a verdict to training data.** A second one
existed and was deleted on 2026-09-21 (`11_ingest_review_log.py`). Two paths
deriving the same labels from the same verdicts is how they silently diverge.
Resist adding a second.

**The holdout is the only honest signal.** Everything else is measured on data
the models could have seen. `12_score_holdout.py` round over round is the only
statement about whether the project is improving that will survive scrutiny.

**Object detection matters more than Stage 2b's feature importances suggest.**
This has confused people before. See [01_ORIENTATION.md](01_ORIENTATION.md) —
it is an artifact of what 06 trains on, and the re-ranker measures the real
question.

**Docstrings are the design documents.** This codebase records *why*,
including approaches tried and reversed, with dates. Before changing
something that looks odd, read the docstring — there is usually a reason, and
it is usually written down. That habit is worth continuing.

**Almost every failure mode in this pipeline is silent.** Stale labels, a
missed `01e` run, a holdout leak, a merge suffix, a correction equal to its
original. Very little of it raises. This is why `exclude_holdout()` is
fail-closed, why `latest_master_layer()` raises rather than guessing, and why
`close_round.py` gates each step on the previous one. Keep new code in that
style.

---

## The 2026-09-21 consolidation

The most recent structural change, in case the reasoning matters later.

**Before:** two paths turned review verdicts into training labels. The master
(`Updates.gdb`) was read but never written by the review loop; a parallel
`review_derived_locations.gpkg` was built on the HPC by
`11_ingest_review_log.py` and unioned in at training time with "Updates wins
on conflict."

**Problems:** the master drifted further from reality every round; the
conflict rule became meaningless once review rows were in both; and the
corrected-coordinate lookup ran against two different parcel stores (local
mirror and HPC), so the same verdict could yield two different points.

**After:**

| Change | |
|---|---|
| Master moved `.gdb` → `.gpkg` | so the review loop can write the file it reads |
| `update_master_locations.py` | new — folds verdicts into the master as a dated layer |
| `close_round.py` | new — one command for the whole local round-close |
| `11_ingest_review_log.py` + `.slurm` | **deleted** |
| `build_training_bins.py --review-gpkg` | **removed** |
| `MASTER_GDB`/`MASTER_LAYER` | → `MASTER_GPKG` + `latest_master_layer()` |
| `candidate_recall_failures.parquet` | moved from 11 to the local round-close |

**Also fixed in the process** (bugs in the R prototype `pull_reviews.R` that
this replaced): corrections were being written as the *reported* coordinate;
`truth_outside_candidates` verdicts were dropped entirely; geometry was
rebuilt from the reported point, regressing previously corrected rows; and
the layer-date sort was a no-op because the format string didn't match the
layer names.

**Not changed:** the hard-negative path. `06b_build_rerank_training.py`
already derived hard negatives from the master's corrections layer, with a
richer top-20 pool than the app ever showed. It needed no modification — only
the discipline of running `01e` before it.

---

## A 30-day orientation plan for a successor

**Week 1 — don't change anything.**
Read [01_ORIENTATION.md](01_ORIENTATION.md) and
[04_REVIEW_LOOP.md](04_REVIEW_LOOP.md). Get HPC access working. Back up the
master. Run `build_test_bundle.py --state OH` and step through the pipeline
locally against the fixture.

**Week 2 — review a round.**
Actually use the app. Review 50 plants yourself. Nothing else will teach you
as quickly what the model is good and bad at, or why `needs_info` is a third
of the outcomes.

**Week 3 — close a round and retrain.**
Run `close_round.py`, upload, run the HPC sequence in
[05_RUNBOOK.md](05_RUNBOOK.md) including `01e`, and score the holdout. This
is the full loop; once you have done it once you own the project.

**Week 4 — pick one open item.**
[07_OPEN_ITEMS.md](07_OPEN_ITEMS.md). Read the "Settled" section at the top
first, so you don't spend the week reopening something that was already
decided.

---

## Questions for the departing author

Fill these in before you go. They are the ones a successor cannot answer from
the code.

1. Who is the stakeholder for this work, and what do they expect and when?
2. Is there a target state list or national deadline?
3. What accuracy would count as "good enough to publish"?
4. Which states have been reviewed so far, and was that order deliberate?
5. Are there facility types or regions known to be systematically hard?
6. Has any of this been presented or published? Where?
7. Who else has ever run any part of this?
8. Is there anything you would have done differently that isn't in
   [07_OPEN_ITEMS.md](07_OPEN_ITEMS.md)?
