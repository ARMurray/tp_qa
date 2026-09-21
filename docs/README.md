# tp_qa — Project Manual

Start here. This manual exists so that someone who has never seen this
project can pick it up, run a full cycle, and keep it moving.

**Written as a handoff document, 2026-09-21.** Read [08_HANDOFF.md](08_HANDOFF.md)
first if you are the person taking this over.

---

## What this project does, in one paragraph

The EPA's Clean Watersheds Needs Survey (CWNS) collects self-reported
locations for every wastewater treatment plant in the country. Many of those
reported coordinates are wrong — they point at a city hall, a mailing
address, a centroid of nothing. This project finds the wrong ones and
corrects them, using machine learning over parcel data, land cover, and
object detection on aerial imagery. A human review loop checks the model's
work, and every review permanently improves the training data for the next
run.

---

## Reading order

| # | Document | What it covers |
|---|---|---|
| 1 | [01_ORIENTATION.md](01_ORIENTATION.md) | The three subsystems, how data flows between them, the vocabulary |
| 2 | [02_ENVIRONMENTS.md](02_ENVIRONMENTS.md) | Machines, Python environments, every path, access you will need |
| 3 | [03_CORRECTION_PIPELINE.md](03_CORRECTION_PIPELINE.md) | The ML pipeline stage by stage, and the full script inventory |
| 4 | [04_REVIEW_LOOP.md](04_REVIEW_LOOP.md) | The review app, and how a verdict becomes training data |
| 5 | [05_RUNBOOK.md](05_RUNBOOK.md) | **The operational runbook.** Start-to-finish commands for a full cycle |
| 6 | [06_TROUBLESHOOTING.md](06_TROUBLESHOOTING.md) | Failures that have actually happened, and what they meant |
| 7 | [07_OPEN_ITEMS.md](07_OPEN_ITEMS.md) | Known gaps, pending decisions, what to do next |
| 8 | [08_HANDOFF.md](08_HANDOFF.md) | What only lived in one person's head |

If you need to **run something today**, go straight to
[05_RUNBOOK.md](05_RUNBOOK.md).

---

## The single most important thing to understand

Everything the human review loop produces reaches the models through **one
file**:

```
Updates.gpkg  (master, local)
    └─ newest CWNS_Locations_YYYYMMDD layer
         └─ build_training_bins.py
              └─ training_locations.gpkg  (classes / corrections / unverified)
                   └─ every model: Stage 1, Stage 2a, Stage 2b, re-ranker
```

There is exactly one derivation path. A second one existed
(`11_ingest_review_log.py`) and was deleted on 2026-09-21 because two paths
producing the same labels from the same verdicts is how they silently
diverge. If you find yourself adding a second way for review data to reach
training, that is the thing this design is specifically trying to prevent.

---

## Documentation status — read this before trusting anything

This repo accumulated documentation over ~5 months of fast iteration. Some
is current, some describes a design that was later replaced. Sorted by how
much you should trust it:

| Document | Status | Notes |
|---|---|---|
| **Script docstrings** | **Authoritative** | Unusually thorough, and kept current. When a docstring and a document disagree, believe the docstring. |
| `docs/` (this manual) | Current as of 2026-09-21 | |
| `TPQA_MASTER_REFERENCE.md` | Mostly current | Sections 1–6, 9 are good. §7 "No Inference Pipeline Exists" is **stale** — `05_run_inference.py` exists. Counts throughout are from the OH/MS/DE pilot and are now low. |
| `REVIEW_LOOP_PLAN.md` | Historical | The *plan*, largely delivered. Phase 4 says "extend `app.R`" — the app was built in Python/FastAPI instead. Phase 5's fold-back design was superseded on 2026-09-21. |
| `correction/README_HPC.md` | Partly stale | Run order references `slurm/` subdirectory; the files are flat in `scripts/`. Use [05_RUNBOOK.md](05_RUNBOOK.md) instead. |
| `review_app/README.md` | Current | Updated 2026-09-21. |
| `detection/PROJECT_ANCHOR.md` | **Largely stale** | "as of May 2026". Describes the earlier all-R pipeline, `Inspection.gdb`, `app.R`, `build_results.R`. Useful for *history and rationale*, not for how anything runs now. |
| `detection/docs/Archive/` | Historical only | Archived deliberately. Do not follow instructions in these. |

**Rule of thumb:** a Python docstring in this repo is a design document. The
authors wrote down *why*, not just *what*, including decisions that were
tried and reversed. Read them before changing anything.

---

## Conventions used throughout this manual

- `local` means the Windows workstation. `HPC` means the cluster.
- Paths under `/work/GRDVULN/tp_qa/` are on the cluster.
- A **round** is one batch of human review (currently ~150 plants).
- A **cycle** is inference → review → fold back → retrain.
- `TODO(handoff)` marks something the original author needs to fill in, or
  that the successor must establish for themselves. These are real gaps, not
  placeholders left by accident.
