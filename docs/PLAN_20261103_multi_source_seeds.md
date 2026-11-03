# Plan — 2026-11-03: feed every plant location into the correction models

**Draft for the owner's review. Branch: `multi-source-seeds`. Nothing here is built yet.**
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
wrong there is often one near the truth. Caveat: reviewers may have used the ICIS pins
to find the answer, so these numbers may be optimistic. New out-of-sample review rounds
are the real test.

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

## Branch and run isolation

- Work on `multi-source-seeds`. Master keeps the current single-seed pipeline as the
  baseline and for any review round that is already running. Merge only after a holdout
  comparison.
- The branch name is deliberately not `claude/*`: the Stop hook in
  `.claude/hooks/auto-push.sh` commits everything and pushes on those branches.
- `config.ROOT` is a fixed HPC path, so a branch checked out there would write over
  master's artifacts. Add a run tag or data-root override (for example a
  `TPQA_DATA_ROOT` environment variable) before the first HPC run, and keep the frozen
  holdout shared.
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

- [x] Branch created
- [x] Plan written
- [ ] `plant_seeds` table
- [ ] Recall-ceiling check on the HPC (today 91.4% of corrections fall inside the CWNS-centered window)
- [ ] Run-isolation override in `config.py`
- [ ] 01a / 01e multi-seed windows
- [ ] Stage 1 retrain and plant-level calibration
- [ ] Stage 2 pool and features
- [ ] Output provenance columns
- [ ] Holdout comparison against master
