# Plan — 2026-10-23: anchor points in tp_qa

**Draft for the owner's review.** Nothing here is built yet. Status checklist
at the bottom.

## Why this belongs in tp_qa

Sewersheds v2 (ARMurray/sewersheds_v2) assigns H3 hexagons to treatment
plants. Besides the plant's own location it needs, per plant, a set of
**anchor points**: points that show where the plant or its collection system
is, each with a reliability score. Judging *what is physically at a point*
is what tp_qa already does, and its review loop is the one place location
verdicts are made. So all point-location judgement lives here, and v2 only
consumes a frozen snapshot (decided 2026-10-23, sewersheds v2 planning).

Split of responsibility:

| Evidence | Owner |
|---|---|
| Point anchors: plant locations, outfalls, CSO outfalls, overflow events, collection-system facility points, pretreatment users, geocoded addresses | **tp_qa** (this plan) |
| Area anchors: CWNS collection system → census place / county subdivision matching | sewersheds_v2 (name matching, not a point problem) |
| How anchors are used to build sewersheds | sewersheds_v2 |

## Vocabulary

- **Anchor**: a point linked to one CWNS treatment plant.
- **Family**:
  - `plant_site` means the anchor claims to be *at the plant*.
  - `collection` means it claims to be *somewhere in the plant's collection system*.
- **Location reliability**: is the point where it says it is.
- **Link reliability**: does the point belong to this plant. These are two separate
  questions and are scored separately: a correctly located CSO outfall can still be
  linked to the wrong plant.
- **Independent**: an anchor whose coordinate is not a copy of another anchor's.
  CWNS often copied ICIS, the FRS best pick is the ICIS point, and outfalls with
  `LATLONG_TYPE = 'Facility'` carry the facility coordinate. Copies are kept,
  but they are tagged and don't count toward consensus.

## What exists today

Counts from the sewersheds_v2 database on 2026-10-23. These are rows **before**
linking to CWNS plants, so the number of plants covered will be lower.

| Source (v2 table) | Family | Rows with coordinates | Notes |
|---|---|---|---|
| CWNS reported location (`staged.location`) | plant_site | ~17,500 plants | already tp_qa's starting point |
| ICIS facility / geocode, FRS (`raw_icis`, `raw_frs`) | plant_site | to count | mostly not independent of CWNS |
| Outfalls, POTW permits (`raw_outfalls.npdes_outfalls_layer`) | plant_site (effluent outfalls sit near the plant) | 42,036 | 8,630 are `LATLONG_TYPE = 'Facility'`, i.e. copies |
| CSO outfalls (`raw_cso.ALL_CSO_DOWNLOADS`, `PF_LAT/LON`) | collection | 8,236 | 792 permits |
| Sewer overflow / bypass events (`raw_sso.sewer_overflow_bypass_report_events`) | collection | 5,451 | manholes 1,181, pump stations 1,069, basement backups 520, broken pipes 362, CSO outfalls 303, other or untyped 2,016 |
| CWNS standalone collection-system facilities (`staged.facility_type` `Collection:*`, not also a plant) | collection | 4,933 | linked to plants through the CWNS discharge table; often a town-hall or centroid point (to measure) |
| Plant addresses for the EPA geocoder (`derived.plant_address`) | plant_site *or* collection | 15,961 addresses | flags: PO box, civic building, no street number, intersection; geocoder results not back yet |
| Pretreatment industrial users | collection | pending (owner obtaining) | addresses expected |
| Influent structures, eRule collection systems | either | to confirm | `raw_sso.collection_system_permits` has population but no coordinates |

**A wrong plant location is often a good collection anchor.** A reported point
that tp_qa moves off a town hall, or an address flagged as a civic building, is
not the plant, but it almost certainly sits inside the sewershed. So tp_qa's
rejected reported points become `collection` anchors and are not discarded.

## Output: the anchor snapshot

One frozen file per snapshot, `anchor_points_<snapshot_id>.parquet`, plus a
manifest recording the tp_qa commit, master layer name, deployed `best.pt` hash
and input hashes. v2 reads only this file.

| Column | Meaning |
|---|---|
| `anchor_id` | stable hash of `source_dataset` + `source_key` |
| `cwns_id` | 11-digit text |
| `anchor_family`, `anchor_type` | e.g. `collection` / `sso_manhole` |
| `source_dataset`, `source_key` | e.g. `raw_cso`, `NPDESID_PFID` |
| `lon`, `lat` | WGS84 |
| `coord_provenance` | `LATLONG_TYPE`, geocoder match level, rounded-coordinate flag, centroid flag |
| `copy_of` | `anchor_id` of the anchor this coordinate duplicates (within 1 m), else NULL |
| `link_method` | `permit_id`, `cwns_discharge`, `name_match` or `address` |
| `link_score`, `location_score`, `reliability` | each in [0, 1]; `reliability` combines the two |
| `score_components` | struct: the individual checks, for auditing |
| `review_status` | `unreviewed`, `confirmed`, `rejected` or `needs_info` |

Plus one summary row per plant, `anchor_plant_summary`, with:
- counts by family and type, and how many are independent;
- the geometric median of the independent anchors;
- robust spread and outlier count;
- convex-hull area and the maximum distance from the plant.

This table is also the input to explorations E2 and E3.

## Scoring

**Plant-site anchors** are judged as tp_qa judges any location: parcel,
land cover, detections, and agreement with the verified master.

**Collection anchors** can't be judged by "is there a plant here?", so they
get their own checks:

1. **Provenance.** A copy of the facility coordinate, rounded coordinates, or a
   ZIP or city centroid: all lower location reliability.
2. **Physical plausibility, by type:**
   - manholes, basement backups and broken pipes should be in developed land
     near roads and buildings;
   - CSO outfalls should sit on or near an NHD flowline or water body;
   - pump stations should be near a road.
3. **Consensus.** The point's distance from the geometric median of the plant's
   *independent* anchors, scaled by a robust spread (median absolute deviation).
   This is not a fixed radius around the plant, because a collection system's
   extent grows with plant size.
4. **Link.**
   - A shared NPDES permit with the plant scores highest.
   - Next comes a CWNS discharge-table link.
   - Name or address matches score lowest.
   - A collection permit that serves several plants splits its link score.

**Calibration truth.** Sourced sewershed polygons in sewersheds_v2 give truth
for collection anchors automatically:

| Where the anchor falls | What it means |
|---|---|
| Inside its own plant's polygon | right |
| Inside another plant's polygon | mislinked |
| Outside every polygon, in a county with complete coverage | mislocated |

Use only High-quality `sewershed_polygon` sources. Where a sourced polygon is
the whole sewershed (`boundary_scope = whole`), an anchor of an upstream plant
also counts as right.

**Leakage rule.** v2 evaluates its model on the same sourced polygons.
So anchor scoring must exclude v2's holdout states and blocks, read from one
holdout file that **v2 owns and tp_qa imports**, fail-closed. This is the same
pattern as `holdout.exclude_holdout()`. tp_qa's own holdout still applies to
anything that feeds the plant-location models.

## Explorations to run before choosing the scoring model

These come from the owner and are recorded so they get done. No decisions yet.

- **E1. Types of points.** For each anchor type and provenance:
  - share right, mislinked or mislocated against the sourced polygons;
  - median distance to the polygon edge when outside it;
  - how many plants it covers, by population band.

  Output: which types carry signal, and a per-type prior reliability.
- **E2. Number of points.** How often plants have 0, 1, 2–4, 5–19 or 20+ independent
  anchors, by population band and state. Then: does consensus get more accurate as
  the count grows (median-to-truth distance against n)?
- **E3. Spatial spread versus number of points.** Spread measured as standard
  distance, hull area, and the maximum distance from the median, against n and
  against sourced sewershed area and reach. Spread is biased low when there are
  few points, so model the expected spread as a function of n before using it as
  a sewershed-extent prior for v2.
- **E4. Addresses.**
  - Geocode plant, collection-system facility, CSO facility and pretreatment-user
    addresses with the EPA geocoder (never external services).
  - Score them by geocoder match level and the `plant_address` flags.
  - Test each address-derived anchor as both a `plant_site` and a `collection`
    anchor: a civic-building address is a poor plant site but may be a good
    collection anchor.
  - Measure how often a plant's address geocode lands on the verified site.

## How it fits the existing pipeline

New scripts, numbered after the current pipeline so nothing existing moves:

| Script | Runs on | Purpose |
|---|---|---|
| `20_collect_anchors.py` | local | Reads the sewersheds_v2 Parquet mirror (read-only) and the master. Links each anchor to a `cwns_id`, deduplicates, records provenance and `copy_of`. |
| `21_score_anchors.py` | local or HPC | Provenance, plausibility, consensus and link scores; the combined `reliability` |
| `22_anchor_diagnostics.py` | local | E1–E4 against the sourced polygons, written to `correction/diagnostics/anchors_*.txt` |
| `23_export_anchor_snapshot.py` | local | Frozen snapshot plus manifest for v2 |

**Phase 2, optional: plant-site anchors inside tp_qa's own models.**
Candidate recall is the binding constraint: Stage 2a top-20 is 69%, and the
realistic k-ring ceiling is about 87% (07_OPEN_ITEMS). Seeding the candidate pool
with parcels around independent ICIS, FRS, outfall and address points could
recover plants recorded in the wrong place, which no k-ring radius reaches. Adding
"distance to nearest independent anchor" and "agreeing-anchor count" as Stage 2a
and re-ranker features is the second half.

This needs a full retrain and a `12_score_holdout` and `calibrate_move_rule.py`
comparison before it is kept. It is not on the critical path for v2.

**One derivation path.** If anchors are ever reviewed by a person, verdicts
go into **one** master, either a layer in `Updates.gpkg` or its own tracked
file, and `20` reads them from there. This is a single path, not a second route
into training (see docs/README.md, "The single most important thing").

## Decisions for the owner

| # | Question | Recommendation |
|---|---|---|
| A1 | Where raw ECHO / CWNS are read from | The sewersheds_v2 Parquet mirror, read-only, with input hashes in the manifest. A second copy of ECHO in tp_qa would drift. |
| A2 | Which plants get anchors | Every plant v2 models, ignoring `MIN_POP_SERVED`, including AK/HI/PR. Collection scoring doesn't need NAIP. |
| A3 | Human review of anchors | Only a calibration sample per type plus disputed links. The sourced polygons provide most of the truth. |
| A4 | Phase 2 (anchors in Stage 2a) | After the anchor snapshot ships, as its own retrain cycle |
| A5 | Pretreatment-user data | Format and fields when the owner has it. Addresses go through E4. |
| A6 | Snapshot cadence | One snapshot per v2 release, the same snapshot that v2 freezes for plant locations |

## Status

- [ ] A1–A6 decided
- [ ] P0 inventory: per-type counts *after* linking to CWNS plants, duplicate and copy rates
- [ ] P1 `20_collect_anchors.py`: first anchor table, provenance rules only
- [ ] P2 `22_anchor_diagnostics.py`: E1–E4 (needs v2 holdout file and `boundary_scope`)
- [ ] P3 `21_score_anchors.py` + `23_export_anchor_snapshot.py`: first snapshot for v2
- [ ] P4 (optional) plant-site anchors in Stage 2a candidates and features, retrain, compare
