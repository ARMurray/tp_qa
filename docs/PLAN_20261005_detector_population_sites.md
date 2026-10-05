# Plan — 2026-10-05: new detector, population > 100, split-parcel sites, output layers

Agreed with the project owner on 2026-10-05. Status per item at the bottom.

## Order

1. **Detector first** (work PC): stable validation split, targeted labelling,
   retrain. A new `best.pt` makes ALL detection output stale (01b, 01c, 01e).
2. **Code changes in parallel** (none depend on the new detector): output
   geometry, site assembly, output layers, viewer layers, review-app
   multi-parcel pick, population floor.
3. **One full HPC cycle** with the new detector and the floor at 100:
   01b/01c/01e `NORESUME=1`, 00 → 02 → 02b → 03 → 04 → 05 → 01e train →
   06b → 07b → 01e holdout/all → 05b → 12 → 13.
4. **Evaluate by population band**, then round 6 (audit sample of moved
   plants + uncertain slice, small plants included).

## Decisions

| Topic | Decision |
|---|---|
| Population floor | `MIN_POP_SERVED` 1,000 → **100**; plants with no population row stay out |
| Small plants in training | **yes** — train and infer on them (`pop_served` is already a feature) |
| Small-plant uncertainty | slice every evaluation by band (100–1,000 vs > 1,000); `confidence_tier` in the output; own cutoff if precision differs (07_OPEN_ITEMS) |
| Moved coordinate, detector fired | **mean centre of the detected objects inside the site**, objects with confidence ≥ **0.4** only |
| Moved coordinate, no detection | parcel **centroid if it falls inside** the polygon, else `shapely.polylabel` (pole of inaccessibility); #1 parcel |
| Verified-corrected plants | keep the reviewer's point |
| Site gap | neighbouring parcels within **50 m** of the site (roads split plants) |
| Site membership | a top-20 candidate (or the reported parcel) joins if within 50 m **and** (a ≥ 0.4 detection lies inside it **or** same normalised owner as #1 with re-rank ≥ 0.5); max **4** parcels |
| Reported parcel part of the site | plant is **kept** (`kept_site`), not moved |
| Output layers | `plants` (points + `coord_method`, `site_parcels`), `sites` (final parcel outline(s)), `detections` (one point per object at the final location: class, confidence, CWNS_ID, ll_uuid, source) |

## Why sites matter beyond the map

- **Scoring:** if the plant is parcels A+B and #1 is B, calibration counts a
  `wrong_candidate` — precision is understated.
- **Training:** 06b labels B a negative when A is the truth.
- **False moves:** a reported point on A "moves" a few hundred metres to B.

Fix in two steps: (a) rule-based site assembly in 13 now; (b) review app gets
"also part of this plant", the master stores the extra parcel ids, 06b stops
using them as negatives, calibration counts any site parcel as right.

## Detector labelling priorities (script to list tiles)

- confirmed plant parcels where nothing fired (misses);
- candidates where it fired that were not the plant (false positives);
- drying beds — the class has never fired on a candidate;
- small plants (lagoons, package plants), coming in with the floor change.

## Status

- [ ] stable (hash-based) detector validation split
- [ ] labelling priority list
- [x] 13: coordinate rule, site assembly, `kept_site`, three layers (`site_geometry.py`; tested on synthetic data, not yet run on the HPC)
- [x] viewer: sites + detections layers (tested on synthetic data)
- [ ] review app: multi-parcel pick → master column → 06b + calibration
- [ ] population floor 100 + band slicing in 12 / calibrate / 13 `confidence_tier`
- [ ] full HPC cycle

## Open question (raised 2026-10-05)

When detections sit on two parcels of a split site, their mean centre can
fall in the road between them. 13 flags it (`coord_in_site = False`, counted
in the log). Options: accept it (it is the true middle of the plant), or snap
such points to the nearest point inside the site.
