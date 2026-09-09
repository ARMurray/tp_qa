# Treatment Plant QA (tp_qa) — Master Reference

This document replaces `HPC_NOTES.md`, `MODEL_NOTES.md`, `RESULTS_NOTES.md`,
`OD_PROJECT_DOCUMENTATION.md`, `TPQA_PROJECT_STATUS_HPC.md`, `SESSION_LOG_2026-08-24.md`,
and `REVIEW_LOOP_PLAN.md` as the single current-state reference. Where those documents
described the R-era pipeline or intermediate designs, this version reflects what's
actually running today, and marks historical material as historical rather than
silently dropping it — some of it (hard-negative mining ratios, the Baden Borough
case, class-balancing lessons) is still-live design reasoning, not just old numbers.

**Last folded together:** 2026-08-25, after the HPC migration, holdout construction,
anti-join patching, the Stage 2b provenance-leak check, and the OH/MS/DE full-universe
pilot.

---

## 1. The Two Pipelines, and How They Connect

This project is two separate codebases that meet at exactly one artifact: a trained
YOLO object-detection model, `best.pt`.

```
  detection/pipeline/  (local, Windows, GPU-bound)
        │
        │  produces
        ▼
     best.pt  ──────────────────────►  correction/  (HPC, Linux, volume-bound)
                  uploaded manually        consumed by 01b and 01c
```

**`detection/pipeline/`** trains the model that finds treatment-plant infrastructure
(aeration basins, clarifiers, digesters) in NAIP aerial imagery. This runs locally —
small dataset, GPU-bound, iterative, annotation-in-the-loop. It has its own `config.py`
and its own path conventions, entirely separate from the correction pipeline's.

**`correction/`** is everything that runs on HPC: extracting candidate parcels,
running the trained OD model against real plant locations at national scale, and
training/serving the three-stage Random Forest architecture (Stage 1 → Stage 2a →
Stage 2b) that actually decides whether a reported coordinate is right and, if not,
what the correct one is. OD is one input into this — not the whole answer. The
original design rationale (from the now-superseded `OD_PROJECT_DOCUMENTATION.md`):

> An existing random-forest model corrects coordinates using text/locational fields
> alone; approximately 40% of reported plant coordinates are wrong, and this module
> adds a signal based on actually detecting treatment infrastructure in aerial imagery.

The three-step workflow that document sketched — (1) is the reported location likely
correct, (2) if not, narrow candidate parcels within a search radius, (3) run targeted
detection against only those candidates — **is** the current Stage 1 / Stage 2a / Stage
2b architecture. It shipped as designed.

**The handoff is manual and one-directional.** `04_train_model.py` writes
`weights/best.pt` locally; a person copies it to HPC's model directory. Nothing
automates this, and nothing on HPC re-triggers if the local model changes. If OD
detection quality is ever suspected as a cause of a correction-pipeline problem, the
first question is always "which `best.pt` is actually deployed" — `01b`/`01c` auto-select
the most recently modified `.pt` file, not a pinned version.

**Currently deployed OD model:** 3 classes — `aeration_basin`, `clarifier`, `digester`
(class IDs 0/1/2). `chlorine_contact`, `drying_bed`, and `oxidation_pond` were annotated
but are excluded from this training round (`KEEP_CLASSES` filter in `04_train_model.py`,
applied to a copy of the dataset — no annotation work is lost). `oxidation_pond`
specifically needs a larger tile size than the current 200m geometry supports and is
deferred to a separate future model, not force-fit into this one.

**Tile geometry / CRS contract** (must match between the two pipelines exactly, since
correction/'s `01a`-`01d` and detection/'s `02_extract_tiles.py` both do pixel↔lonlat
math against it):

| Constant | Value |
|---|---|
| `TILE_SIZE_M` | 200 |
| `OVERLAP_PCT` | 0.33 |
| `STRIDE_M` | ~134 |
| `TARGET_RES_M` | 0.6 |
| `IMAGE_PX` | 333 |
| `MODEL_IMGSZ` | 352 (333 rounded up to YOLO's stride-32 requirement; detected boxes still come back in 333px space) |
| `EXPORT_CRS` | 4326 (WGS84) |
| `PROJECTED_CRS` | 5070 (Albers CONUS, meters) |

---

## 2. Current File Tree

```
tp_qa/
│
├── correction/                              HPC: /work/GRDVULN/correction/scripts/
│   │
│   ├── PIPELINE — run in this order
│   │   ├── 01a_extract_parcels.py(+.slurm)       candidate parcels + NLCD zonal stats, per state
│   │   ├── 01b_run_object_detection.py(+.slurm)  OD on REPORTED location, using best.pt
│   │   ├── 01c_run_od_corrected_locations.py(+.slurm)
│   │   │                                          OD on CORRECTED location — Stage 2b's positive-
│   │   │                                          side contrastive pair. Reuses 01b's fetch/NMS/
│   │   │                                          parcel-cap machinery via importlib.
│   │   ├── 01d_nlcd_topup.py(+.slurm)             small top-up for parcels 01a's k=18 ring missed
│   │   ├── 02_feature_engineering.py(+.slurm)     plant + parcel features; Stage 1/2 training tables
│   │   ├── 03_train_stage1.py(+.slurm)            Stage 1 RF — parcel-correctness classifier
│   │   ├── 04_train_stage2.py(+.slurm)            Stage 2a RF — candidate ranking, no OD
│   │   ├── 06_build_stage2b_training.py(+.slurm)  builds Stage 2b's OD-aware contrastive training set
│   │   ├── 07_train_stage2b.py(+.slurm)           Stage 2b RF — final candidate re-rank, OD-aware
│   │   ├── 08_diagnose_candidate_coverage.py(+.slurm)
│   │   │                                          k=18 ring's real radius + candidate-pool ceiling
│   │   ├── 08b_analyze_ring_misses.py(+.slurm)    near/far/wrong-county split — is raising K_RINGS worth it
│   │   └── 09_build_holdout.py(+.slurm)           samples + freezes the eval holdout
│   │
│   ├── SHARED / SUPPORT
│   │   ├── config.py                    single source of truth: paths, K_RINGS, etc.
│   │   ├── holdout.py                   exclude_holdout() — fail-closed anti-join (03/04/06/07)
│   │   ├── model_utils.py               shared preprocessing/CV/threshold logic (03 + 04)
│   │   ├── build_training_bins.py       builds training_locations.gpkg (classes/corrections layers)
│   │   ├── 00_build_training_bins.slurm
│   │   ├── setup_env.sh                 one-time venv build (run once on login node)
│   │   └── _common.sh                   shared SLURM preamble, sourced by every wrapper
│   │
│   └── DIAGNOSTICS / ONE-OFF CHECKS      (investigative tools, not a repeatable pipeline stage)
│       ├── check_01a_01b_complete.sh(+.slurm)
│       ├── diagnose_01b_output.py(+.slurm)
│       ├── diagnose_fetch_failures.py(+.slurm)
│       ├── diagnose_parcel_duplicates.py(+.slurm)
│       ├── diagnose_stage1_attrition.py(+.slurm)
│       ├── list_training_states.py(+.slurm)
│       ├── inspect_deployed_02.py
│       └── inspect_stage2_columns.py
│
└── detection/
    └── pipeline/                        local/Windows — never runs on HPC
        ├── 01_sample_sites.py           live — round-2 training sample (plants + TRI negatives)
        ├── 02_extract_tiles.py          live — NAIP tile extraction
        ├── 03_prepare_dataset.py        live — LS label matching, train/val split, dataset.yaml
        ├── 04_train_model.py            live — YOLOv8s training → weights/best.pt
        │                                   ★ sole handoff point into correction/
        ├── 05_run_inference.py          DEPRECATED — standalone local demo, no RF involved
        └── 06_build_map.py              DEPRECATED — viewer for the old HPC OD-only state pipeline
                                          (07_run_state_pipeline_hpc.py / config_hpc.py / 
                                          00_build_full_plant_list_hpc.py / state_fips_hpc.py — 
                                          all superseded, folded into correction/'s current design)
```

Reference files kept alongside code: `CWNSDatabaseDictionaryJanuary2025.xlsx` (CWNS
field dictionary, used throughout `01a`–`09`).

---

## 3. Correction Pipeline Architecture

### Stage 1 — Location Confidence Classifier
Binary: is the reported plant location on the correct parcel? Random Forest.
Below-threshold plants get flagged for Stage 2.

### Stage 2a — Candidate Parcel Ranker
For flagged plants, searches parcels within `K_RINGS` (currently 18) H3 res-9 rings of
the reported point and scores each by likelihood of being the true location. No OD
signal — parcel/LBCS/ownership features only.

### Stage 2b — OD-Aware Final Re-Rank
Takes Stage 2a's output and re-ranks using OD detection features, sourced from `01b`
(reported-location OD) and `01c` (corrected-location OD, for the positive side of the
contrastive training pair).

**2026-08-25 finding, current model:** the OD-leak concern that motivated checking
this (see §6) came back clean — `od_ran`/`od_has_detection` don't appear near the top
of permutation importance. But OD features overall aren't contributing much either
(`od_max_conf_aeration_basin` led the OD features at ~0.005 importance, versus
`parcel_bbox_max_dim_m` at 0.134). Current working conclusion: Stage 2b is mostly
re-deriving parcel geometry that Stage 2a already had, not yet earning its OD-aware
premise. The review loop (§4) — specifically distribution-matched hard negatives,
which `06_build_stage2b_training.py`'s training-only-plant construction structurally
cannot produce — is the intended fix, not a retrain with the current data.

### Candidate search geometry
- H3 resolution 9, ~174m edge, ~330m per ring step.
- `K_RINGS = 18` measures to roughly 5.4km radius empirically (not the ~10km the
  original comment claimed — see `08_diagnose_candidate_coverage.py`'s Q1).
- Coverage ceiling: candidate pool contains the true parcel for ~84.2% of corrections
  (`08`'s Q2). The rest split into near-miss (radius problem, `08b`'s near category),
  far-miss (same-metro, expensive to fix), and wrong-county/state (no radius fixes this
  — confirmed case from the R era: CWNS 36007190002, 101 rings / ~33km off).
- 43% of reported points land on no parcel at all nationally — this is why Stage 2b's
  negatives have to come from ranked candidates, not from improving parcel coverage.

### Historical design lessons still worth carrying forward
From the R-era models (`MODEL_NOTES.md`, now superseded — the actual thresholds/AUCs
below are stale, but the *reasons* are not):

- **Class weights don't work for this imbalance; per-tree balanced sampling does.**
  With ~168 positives vs thousands of negatives, class weights adjust the loss but not
  the split criterion — Gini impurity at each node is still dominated by the majority
  class, so the model still learns majority-class splits regardless of weight
  direction. `sample.fraction = c(0.5, 0.5)`-style equal per-tree sampling was what
  actually fixed it (confirmed: threshold jumped from 0.057 to 0.298 once this was
  fixed, reflecting a model that finally made meaningful positive predictions instead
  of never predicting "correct" at all).
- **Hard negative mining matters.** Most parcels are easy negatives (single-family
  homes) the model learns instantly; the useful signal comes from forcing it to
  distinguish real treatment-plant parcels from other government/utility/water-adjacent
  parcels that superficially resemble them.
- **Reported-parcel exclusion and candidate competition resolution are both required
  at inference, not just nice-to-haves.** A plant cannot be "corrected" to its own
  reported parcel, and when two plants' candidate lists overlap on the same parcel, it
  must go to whichever plant scores it higher — resolved only at final-assignment time,
  keeping the full candidate pool intact for QA. Neither of these exists yet in the new
  inference pipeline (see §7) and both need to be built in.
- **Baden Borough, PA (CWNS 42005015001) — the case that motivated `has_ww_keyword`.**
  A plant spanning two parcels, reported location on no parcel. Stage 2 ranked a park
  (`osm_ww=TRUE`, wrong parcel) 1st and a *different* treatment facility across the
  river 2nd, while the actual correct second parcel — 1.68 acres, real wastewater LBCS
  signals — ranked 8th, purely because `log_gisacre` penalized its small size and
  nothing outweighed that. This is directly relevant to the current Stage 2b result
  above: `parcel_bbox_max_dim_m` (a close cousin of `log_gisacre`) dominating again is
  the same failure mode recurring, not a new one.
- **Geography dummies are a trap.** Passing `subdivision`/`place`/`county` as raw
  categorical columns exploded feature count into ~1,037 near-useless dummies (nearly
  all novel at inference, coerced to NA). The boolean aggregates (`any_geo_match`,
  `sd_match`, `place_match`, `county_match`) capture the same signal without the
  explosion — current Python feature engineering uses the aggregate approach, not raw
  geography columns, for this reason.

---

## 4. The Review Loop (Phases 1–7, per `REVIEW_LOOP_PLAN.md`)

The forward plan for closing the loop between model output and improved training data.
**Full phase detail lives in `REVIEW_LOOP_PLAN.md`'s content, folded in here at a
summary level** — consult it directly for the phase-by-phase build order if this
summary isn't enough:

1. Frozen holdout (**done** — see §5), anti-joins applied to `03`/`04`/`06`/`07`
   (**done**).
2. Pilot inference on 2–3 states end-to-end. **This step was never built for the new
   architecture** — see the gap in §7. Currently in progress: OH/MS/DE full-universe
   `01a`→`01b`→`02` is done; the actual scoring/ranking/assembly code is not.
3. `12_score_holdout.py` — score the frozen holdout once models are holdout-clean.
   Not yet built. 350 of 425 holdout plants have derivable truth and can score
   immediately; the 75 unlabeled-bin plants need one-time review first.
4. `10_build_review_queue.py` — assemble ranked candidates into a reviewable queue.
   Not yet built. Also not yet routing holdout plants away from the training feed —
   important once pilot inference produces output, since every pilot state contains
   holdout plants.
5. `app.R` rewrite. The old `app.R` and its four upstream R scripts
   (`run_inference_stage1/2/combine.R`, `build_results.R`) are gone — deliberately
   thrown away, not part of the new design. Nothing has replaced them yet.
6. `11_ingest_review_log.py` — fold review verdicts back into training data.
   Distribution-matched negatives are the point of this step; anything reviewed before
   it exists (e.g. against an old `Version_N` R-pipeline output, if that's ever done
   as a stopgap) must be tagged and excluded from ingestion, since those negatives
   match the R pipeline's candidate distribution — exactly the train/serve mismatch
   this loop exists to fix.
7. Retrain, re-score, repeat. `round_manifest.json` records provenance per round
   (holdout checksum, model versions, review-log range).

---

## 5. The Holdout

Built by `09_build_holdout.py`, frozen once created — plants in it never enter
training, in any round. `holdout.py`'s `exclude_holdout()` is fail-closed: a missing
manifest raises rather than silently training on the holdout.

**Current committed manifest** (2026-08-25, checksum `6d300a633b34...`):
- 425 plants: 300 correct, 50 corrections, 75 unlabeled.
- 350 plants have derivable truth and score immediately; 75 (unlabeled bin) need a
  one-time review before the deployment-distribution error rate is real.
- Corrections slice standard error ≈ ±7.1pp at n=50 (worst case); unlabeled slice
  national error-rate SE ≈ ±5.8pp at n=75.
- Growth path: `--append-cohort`, never `--force-resample` once training has run
  against it (contamination is not reversible after that point).

**Two bugs found and fixed in `09_build_holdout.py` this session** (already applied to
the live copy — noted here for provenance, not as an open item):
1. Per-bin RNG seed used Python's salted `hash()`, making `--seed` non-reproducible
   across processes. Fixed with a stable `hashlib.sha256`-based seed.
2. The truth-file union with prior contents wasn't filtered back to current manifest
   membership, so `--force-resample` left behind a superseded draw's rows as live
   truth — confirmed once (425-plant manifest paired with a 522-row truth file, 172 of
   them plants already back in training). Fixed: truth is now filtered to manifest
   membership immediately before every write, under both `--force-resample` and
   `--append-cohort`.

---

## 6. Stage 2b Provenance-Leak Check (resolved, 2026-08-25)

Standing question from the HPC migration: did Stage 2b's strong AUC (0.977 spatial CV)
reflect real signal, or an `01b`/`01c` provenance artifact — i.e., was the model
learning "OD was run for this row" (`od_ran`) as a proxy for the label, rather than
learning from what OD actually detected? No prior run had reached permutation
importance to check, because every run before the holdout/anti-join patch crashed
first.

**Result: clean.** `od_ran` and `od_has_detection` don't appear near the top of
permutation importance at all. See §3 for the follow-up finding (OD features
contribute little either way, for a different reason).

---

## 7. Current Gap: No Inference Pipeline Exists

Nothing on HPC currently chains Stage 1 → Stage 2a → OD-on-top-K → Stage 2b → assembled
output. This is the largest unbuilt piece of the project and the actual blocker on
Phase 2 of the review loop. Needed:

- Stage 1 scoring over the full-universe `05_plant_features.parquet` /
  `10_parcel_features.parquet` (these two now build correctly for a given state
  scope — see §8).
- Stage 2a ranking of candidates for flagged plants.
- A decision on whether Stage 2b belongs in this round at all, given §3's finding —
  current lean is Stage 1 → Stage 2a → top-K, skip OD-on-candidates and Stage 2b for
  the first pilot round, since building the (unbuilt, expensive) OD-on-arbitrary-top-K
  step to feed a model that isn't clearly earning its premise yet is likely wasted
  effort until the review loop supplies better Stage 2b training data.
- Reported-parcel exclusion and candidate-competition resolution (§3) — real
  algorithmic requirements from the R era that have no equivalent yet in the new code.
- Assembly into whatever `app.R`'s rewrite ends up expecting (not yet designed).

**Pilot states in progress:** OH (full universe, in-training-heavy — good for finding
schema drift against banked debugging), MS (thin training presence — domain-shift
probe), DE (thin training presence, geographically tiny — cheap probe, also validated
the `--full-universe` flag itself, which had never been exercised before this pilot).

---

## 8. OH/MS/DE Full-Universe Pilot — Status and Fixes Applied

`01a`→`01b`→`02` full-universe extraction is complete and correct for OH/MS/DE as of
2026-08-25. Two real bugs were found and fixed in `02_feature_engineering.py` along
the way; both are already applied to the live copy.

1. **`fetch_parcel_attrs` dedup scare — false alarm, but worth the record.** A 5.47x
   row inflation (26.65M parcel-feature rows against 4.87M candidate parcels from
   `01a`) was initially suspected to be duplicate `ll_uuid` rows in the raw parcel
   store. Directly checked: OH's `PARCEL_BASE` store has zero duplicates
   (total rows == distinct `ll_uuid`, ratio 1.000). Root cause was elsewhere.

2. **Real bug: the per-state parcel-features cache directory was never scoped to the
   current run.** `10_parcel_features_by_state/` is a deliberate persistent cache
   (each state's PART 2 output written once, reused across runs to bound memory) —
   correct design. But the union step at the end of PART 2 globbed every `*.parquet`
   file in that directory, not just the states the current run asked for. The
   nationwide training-scope run from 2026-08-21 had left 51 state files sitting
   there; the 3-state OH/MS/DE full-universe run unioned all of them in, producing a
   26.65M-row table when the correct answer was 4.87M. Fixed: the union now builds
   its file list from `state_list` explicitly, warns if a requested state has no
   cached file, and notes when the cache holds more states than the current run used.
   Confirmed fixed: re-run produced exactly 4,873,248 rows, matching `01a`'s
   three-state total.

3. **A related, still-open risk for future retrains:** `14_stage1_training.parquet`
   and `15_stage2_training.parquet` share the same fixed output path regardless of
   scope, and are NOT resume/cache-protected the way the parcel-features cache is —
   every `02` run unconditionally overwrites them with whatever scope it was given.
   Running `02` for a 3-state full-universe pilot silently shrinks these two files
   down to that pilot's labeled subset (e.g. 8 corrections instead of the nationwide
   316), which would silently corrupt a `03`/`04`/`06`/`07` retrain if run against them
   without first re-running `02` nationwide (no `--states`, no `--full-universe`) to
   restore them. **Not yet fixed at the code level** — current mitigation is
   procedural discipline (always restore nationwide `14`/`15` before retraining), not
   a structural fix. A durable fix would give the parcel-features table and the
   training tables separate, non-colliding output paths by scope, the same way the
   per-state cache already does.

**Resume behavior, three scripts, three different guarantees:**
- `01a`: whole-state only. Existing output means the whole state is skipped —
  `--full-universe` on an already-training-scoped state needs `NORESUME=1` paired
  with it, or it silently produces nothing.
- `01b`: real per-plant resume. Safe to re-run with wider scope; only new plants
  get processed.
- `02`: no resume at all. Every state in `--states` is fully recomputed every run;
  the per-state cache files exist only to bound memory during the union, not to skip
  work.

**Known, accepted tail issue:** ~5–15% NAIP fetch failures per state, virtually all
the same signature — a tile whose bbox straddles two adjacent NAIP quads, and the
retry's re-resolved item still doesn't cover it. Real cross-quad mosaicking would
close this fully; deliberately not built as low-value for the current scope. Not
rate-limiting related (confirmed via `diagnose_fetch_failures.py`'s per-error-message
breakdown) unless retry-storm messages appear in the failure text directly.

**Multi-state array submissions can trigger sustained Planetary Computer
rate-limiting** when run concurrently (confirmed: OH+MS+DE at 32 workers each = up to
96 concurrent workers, one batch went from 3.4min to 19min mid-run). Retries absorbed
it without data loss, but it cost real wall-clock. Recommend serializing rather than
array-parallelizing future multi-state full-universe pulls, or using `%N` array
throttling.

---

## 9. HPC Environment — Durable Facts

(Filtered from `HPC_NOTES.md`; folder-structure and R-script-naming content there is
fully superseded by the tree in §2 and dropped here. Cluster-level facts and gotchas
that apply regardless of language are kept.)

- **Cluster:** atmos HPC, RHEL 8.10, SLURM. Partitions: `debug` (interactive/testing),
  `compute` (batch). Account: `grdvuln`. 128 nodes × 2 processors × 32 cores, 256GB
  RAM/node (4,096 cores total). Request up to ~180GB in practice to leave headroom.
- **`_common.sh`** is the current shared preamble (module load, venv activate, sanity
  checks) sourced by every correction-pipeline `.slurm` wrapper — this replaces the
  old per-script SLURM template pattern from the R era.
- **DuckDB file locking:** exclusive write locks, one writer at a time. Read-only
  connections still hold a lock that blocks a writer from opening. A stale `.wal` file
  alongside a `.duckdb` file means a previous connection didn't close cleanly — safe
  to delete if nothing is actively writing. The current Python pipeline mostly reads
  parquet directly via DuckDB's `read_parquet()` rather than maintaining a persistent
  `.duckdb` file, which sidesteps most of this — but it's relevant if any script ever
  moves to a persistent DB file.
- **DuckDB spatial extension** must be installed+loaded fresh each session
  (`INSTALL spatial; LOAD spatial;`), plus
  `SET enable_geoparquet_conversion = false;` — without the latter, reading parquet
  with geometry metadata fails with "Geoparquet metadata does not have a version."
  This pattern is already in every correction-pipeline script that touches DuckDB.
- **H3 (res 9):** ~0.1 km² per cell, ~174m edge, ~330m per ring step. k=18 → current
  standard, measures to ~5.4km radius empirically (see §3's coverage note — the
  original "~10km" figure in comments looks like it was describing the diameter, not
  the radius).
- **XGBoost blocked on this cluster:** Intel oneAPI compiler interprets `-O3` as
  invalid (`-03`), and no pre-compiled binary exists for RHEL 8.10. Deferred
  indefinitely; Random Forest is the working algorithm for all three stages.
- **Census GDB path capitalization:** `/work/GRDVULN/data/Census/` (capital C).
- **OSM path:** `/work/GRDVULN/data/osm/Wastewater_Plants.gpkg`, geometry column
  named `geom`, not the geopandas default `geometry` — must be specified explicitly
  wherever this file is read.
- **Regrid parcel store:** `/work/GRDVULN/data/parcels/state=XX/{geoid or county}.parquet`
  (correction/'s `01a`/`01b` key by `state=XX/*.parquet` broadly; detection/'s local
  pipeline keys by `state=XX/{geoid}.parquet` specifically — same store, different
  read patterns per pipeline's needs).

---

## 10. Open Questions / Things to Resolve Next

- Should Stage 2b be in the first pilot inference round at all, given §3's finding?
  Current lean: no — ship Stage 1 → Stage 2a → top-K first, revisit Stage 2b once the
  review loop can supply distribution-matched negatives.
- `14_stage1_training.parquet`/`15_stage2_training.parquet` path-collision risk (§8.3)
  — needs a structural fix (scope-tagged output paths) before full-universe pilots and
  nationwide retrains become routine, interleaved operations rather than one-off events.
- `app.R`'s replacement is undesigned. Nothing currently specifies what the new review
  UI needs from the inference pipeline's output — this should probably be decided
  before, not after, the inference assembly step is built, so the output format is
  designed for its actual consumer.
- Reported-parcel exclusion and candidate-competition resolution (§3) have no home yet
  in the new pipeline — need to land somewhere between Stage 2a/2b scoring and
  whatever assembles final output.
