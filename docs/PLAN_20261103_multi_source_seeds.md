# Plan — 2026-11-03: feed every plant location into the correction models

**Draft for the owner's review. Built on master (decided 2026-11-03). Nothing here is built yet.**
Status checklist at the bottom.

## Why

Today the pipeline starts from one point per plant, the CWNS reported location:
Stage 1 judges that point's parcel, and Stage 2 searches the k=18 ring (about 5.6 km)
around it. We now have up to five plant-role points per plant, from CWNS and ICIS-NPDES
(see `build_plant_candidates.py` and the QA dashboard). Against the reviewed truth
(837 corrected plants, 2026-11-03):

| Point within 500 m of truth | Share |
|---|---|
| CWNS point | 46% |
| Any plant-role point | 85% |
| Consensus point | 70% |

Sometimes the CWNS point is right, sometimes the ICIS facility point, and when all are
wrong there is often one near the truth. The reviewers did not use the ICIS
pins to find the corrections (owner, 2026-11-03), so this is not circular. The truth set is
still the reviewed plants, which lean toward doubted ones; new out-of-sample review rounds
are the real test of the end-to-end result.

## Terms

- **Seed**: one source point for one plant. Sources: `cwns_point`, `cwns_addr`,
  `icis_fac_frs`, `icis_fac_addr`, `icis_feat_INF`. Rollup: CWNS coordinates, CWNS
  geocode, ICIS-NPDES.
- **Origin**: where a seed ultimately comes from (CWNS, ICIS facility, ICIS feature).
  Support is counted in origins, not points.
- **Site**: parcels merged by `site_geometry.py` (within 50 m, holding detected objects
  or sharing an owner).

## Flow

1. **Seeds.** Build `plant_seeds`: one row per plant and source, with `seed_id`, lat, lon,
   origin, and `copy_of` (the seed it duplicates within 1 m, else null). Copies are kept
   and tagged but do not count as independent support. Many ICIS facility points equal
   the CWNS point exactly.
2. **Dedupe first.** Map each seed to its parcel (nearest parcel within a small snap
   distance; otherwise keep it as point-only evidence). Collapse seeds on the same parcel
   into one row that carries the list of sources. Then merge parcels into sites. A site
   goes through Stage 1 once.
3. **Stage 1** scores each distinct site. It is retrained on seed sites: for corrected
   plants the truth site is positive and the other seed sites are negative. The number of
   independent origins on the site is a feature.
4. **Decision after Stage 1.**
   - Any site at or above the threshold: take the highest-scoring one. Near-ties break on
     origin support, then on the CWNS point. Record the runner-up as `alt_location`.
   - Two sites both pass, far apart, with close scores: flag for review, do not choose.
     This is also where one CWNS ID may cover two real facilities.
   - Nothing passes: go to Stage 2.
5. **Stage 2** gets the union of the search windows around every seed, not only the seed
   parcels. Seed sites that failed Stage 1 stay in the pool with their source flags:
   failing Stage 1 means unconfirmed, not ruled out.
6. **Threshold calibration.** Calibrate on the plant-level outcome (of the plants where we
   accept a Stage 1 site, how many are right), not per parcel. Taking the best of several
   looks raises the chance a wrong parcel passes. The project target is more than 90%
   precision of automated moves (07_OPEN_ITEMS).

## Provenance in the output

Source must follow the models through to `13_build_corrected_output.py`.

| Column | Meaning |
|---|---|
| `loc_sources` | sources whose seed lies in the final site or within 250 m of it, e.g. `cwns_point+icis_fac_frs` |
| `led_by` | the source whose search window first surfaced the winning parcel (Stage 2 moves) |
| `dist_to_final_<source>_m` | one column per source, so the 250 m rule can change without a rebuild |
| `copy_of` flags | sources that coincide exactly with another and so are not independent |
| `alt_location` | runner-up site when more than one passed |
| `status` | existing values, plus how a Stage 1 acceptance was reached (which sources were on the accepted site) |

Carried per candidate through 01a, 01e, 02, 04, 05, 05b and 10: the distance to each
source and the independent-origin count.

## What changes where

| Script | Change |
|---|---|
| new `build_plant_seeds.py` | writes `plant_seeds` from the candidate tables |
| `01a`, `01e` | search around every seed; candidate carries seed distances and origin count |
| `02` | Stage 1 training and inference tables keyed by site, not plant |
| `03` | retrain on seed sites; recalibrate the threshold at plant level |
| `04`, `05`, `05b` | seed features; pool is the union of windows |
| `10`, `13` | provenance columns, `alt_location`, multi-pass flag |
| `12` | holdout scoring reports per-source recall |

## Working on master, safely

Decided 2026-11-03: no long-lived branch. The HPC checkout and `config.ROOT` are shared
by everything, so a second tree there would write over master's artifacts, and a diverging
branch would collide with edits made for ongoing review rounds. Instead, every change
lands on master as a small, separately runnable step.

- **Seed mode defaults to `multi`** (owner's decision, 2026-11-03: use all possible
  locations). `SEED_MODE` in `config.py` is `multi` or `cwns_only`, overridable with the
  `TPQA_SEED_MODE` environment variable. `cwns_only` is kept only so the old behavior can be
  reproduced as the baseline for holdout comparisons. A script that has been converted runs
  multi-seed by default; one that has not is unchanged.
- **New outputs get new names.** Multi-seed tables, models and output files are written
  under their own folders or suffixes (for example `*_multi`). Nothing existing is
  overwritten. Retrained models are saved beside the current ones, not over them.
- **Small, separately runnable steps.** Each change is committed on its own so a bad one
  is a single revert, and the build order below keeps the pipeline runnable at each step.
- **Compare against the old pipeline, but do not gate on it.** Run `12_score_holdout` for both
  modes on the frozen holdout and record the difference. The current single-seed models and
  outputs stay in place as the baseline because new outputs are written beside them.
- Retraining changes every score. Scores from before and after are not comparable except
  through the `12_score_holdout` series.

## Open decisions

- Snap distance for a seed that falls on no parcel.
- Near-tie margin and the "far apart" distance for the multi-pass flag.
- The 250 m rule for `loc_sources`.
- Whether `icis_feat_INO` and the exo/iu points enter as seeds. This plan uses plant-role
  points only.
- Whether a copy of the CWNS point counts toward origin support (this plan: no).

## Status

Built 2026-11-03 and tested on synthetic parcel stores; **not yet run on the HPC**. How to run it:
[RUNBOOK_multi_source.md](RUNBOOK_multi_source.md).

- [x] Plan written; decided to build on master, not a branch
- [x] `plant_seeds` table (`build_plant_seeds.py`, 58,130 seeds for 17,538 plants)
- [x] Recall-ceiling check (`check_seed_recall.py`, cell level, 837 corrected plants): truth inside the
      window 95.8% today (state union of CWNS-point disks) vs 99.4% with every seed. A plant's own
      window grows a median 1.07x, mean 1.2x, 12.6% of plants above 1.5x a single disk. Five plants stay
      outside (CWNS point 9 km or more off). Cell level only: it ignores parcel gaps, so it is not
      comparable to 08's 91.4%
- [x] `SEED_MODE` setting in `config.py`, default `multi`, `TPQA_SEED_MODE` override; mode-suffixed output directories
- [x] Seed -> parcel -> site (`seed_sites.py`, `01f_build_seed_sites.py`). Parcels merge on owner within 50 m, and (added after the
      OH run) on seeds within 100 m of each other whoever owns them
- [x] 01a seed window centers; 01e `--seed-sites` detection
- [x] Stage 1 on sites: 02 table, 03 grouped splits and plant-level accept threshold, label-leak guard
- [x] Stage 2 window and per-candidate seed features (02, `seed_sites.add_seed_features`)
- [x] 05 site decision, near-tie and multi-pass rules, union pool with nothing excluded
- [x] 13 provenance columns; 12 end-to-end outcomes; 02b/merge_05 keys
- [ ] Run on the HPC, then holdout comparison against the single-seed pipeline
- [ ] Review app: show each candidate's sources on its card (not started)
