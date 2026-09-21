# 1. Orientation

## The problem

CWNS facilities report their own coordinates. A large fraction are wrong in
ways that matter for downstream analysis — sited on the operator's office, a
municipal building, a ZIP centroid, or simply mistyped. There is no
authoritative national dataset of true treatment plant locations to join
against, so the correct location has to be *inferred* and then *verified*.

Two things make inference tractable:

1. **Treatment plants sit on distinctive parcels.** Large, municipally
   owned, near water, specific land cover. Regrid's national parcel data
   plus NLCD land cover carries most of that signal.
2. **Treatment plants look distinctive from the air.** Circular clarifiers,
   rectangular aeration basins, oxidation ponds. A YOLO model trained on
   NAIP imagery can see them.

## The three subsystems

```
┌─────────────────────────────────────────────────────────────────┐
│  detection/          LOCAL                                      │
│  Computer vision. Trains a YOLOv8 model to find wastewater      │
│  infrastructure in NAIP aerial imagery.                         │
│  Output: models/object_detection/best.pt                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ best.pt uploaded
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  correction/         HPC  (/work/GRDVULN/tp_qa/correction)       │
│  The ML pipeline. Consumes best.pt, parcel data, NLCD, and the  │
│  verified-location training labels. Trains four models and runs │
│  national inference.                                            │
│  Output: a ranked list of candidate parcels per suspect plant   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ review_queue_round{N}.parquet
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  review_app/         LOCAL                                      │
│  Human review of model output. A reviewer sees the reported     │
│  location and the top candidates on NAIP imagery and decides.   │
│  Output: verdicts → the master locations file → new training    │
│          labels, AND new NAIP tiles for detection/ retraining   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            └──── feeds back into BOTH other subsystems
```

Only `correction/` runs on the HPC. `detection/` and `review_app/` both run
locally, which is why `review_app/analysis/extract_review_tiles.py` can write
directly into `detection/data/tiles/` — they share a filesystem.

## Why the loop is the point

Each review round does three jobs at once:

1. **Measures** the model — is it picking the right parcels, and at what rank?
2. **Grows the training set** — every verdict becomes a permanent label.
3. **Grows the detection training set** — every parcel the reviewer saw gets
   its NAIP tiles pulled into the labeling inventory.

The second is the one that compounds. The project started with a few dozen
manually corrected locations; each round of review adds ~200 more verified
rows to the master. That is the asset being built, and it outlives any
particular model.

## The four models

| Model | Script | Question it answers |
|---|---|---|
| **Stage 1** | `03_train_stage1.py` | Is this plant's *reported* location correct? |
| **Stage 2a** | `04_train_stage2.py` | Given it's wrong, which nearby parcel is the real one? |
| **Stage 2b** | `07_train_stage2b.py` | OD-aware: reported parcel vs. corrected parcel |
| **Re-ranker** | `07b_train_rerank.py` | Among Stage 2a's top-20, which is really it? |

Stage 2b and the re-ranker both exist because of a subtle training-data
problem worth understanding before you touch either:

> Stage 2b trains on (reported parcel, corrected parcel) pairs. Those two are
> usually nothing alike — a residential lot versus a 40-acre municipal parcel
> on a river. Parcel attributes separate them trivially, so object detection
> has no residual variance left to explain, and its measured importance
> collapses to near zero. That is a fact about the *training pairs*, not about
> object detection.
>
> The re-ranker fixes this. It trains on Stage 2a's top-20 candidates, where
> every candidate is already a plausible municipal parcel of roughly the right
> size and land cover. There the parcel features are near-tied and OD is the
> only feature family that can look at the actual infrastructure.
>
> Measured on the holdout (`diagnose_candidate_od.py`, 2026-09-08): among
> top-20 candidates, true parcels fire OD at **46.4%** and competitors at
> **9.0%** — a 5× ratio. Ranking by OD confidence alone picks the true parcel
> 35.7% of the time against a 5% random baseline.

This is the single best example of the kind of reasoning recorded in the
docstrings. `06b_build_rerank_training.py`'s header has the full argument.

## Vocabulary

| Term | Meaning |
|---|---|
| **CWNS_ID** | The unique key for a facility. String, always — never let it become an int. |
| **Reported location** | What CWNS says. `Original_X` / `Original_Y`. |
| **Corrected location** | The verified truth. `Corrected_X` / `Corrected_Y`. |
| **The master** | `Updates.gpkg` — the accumulated human-verified record. Local. |
| **Training bins** | `training_locations.gpkg`: `classes`, `corrections`, `unverified` layers. |
| **Correct / Incorrect** | Whether the *reported* point was right. Stage 1's label. |
| **Corrections bin** | Incorrect plants where we also know the *right* answer. Stage 2's positives. |
| **Candidate** | A parcel from the k-ring search that might be the true location. |
| **ll_uuid** | Regrid's parcel identifier. |
| **The holdout** | A frozen evaluation sample, never trained on. See below. |
| **Round** | One batch of human review. |
| **k-ring** | H3 hexagon search radius around the reported point. `K_RINGS = 18` ≈ 5.4 km. |

## The holdout — the rule that must never be broken

`09_build_holdout.py` samples an evaluation set **once** and freezes it.
Plants in it never enter training, in any round, in any form.

`correction/scripts/holdout.py` enforces this and is deliberately
**fail-closed**: if the manifest file is missing, it raises rather than
returning the data untouched. A training run that silently trains on its own
evaluation set is worse than one that stops.

Every training script (03, 04, 06, 07, 06b, 07b) calls `exclude_holdout()`.
So does the review export (holdout verdicts route to a separate file) and the
master update (holdout rows are excluded by default). If you add a new
training script, it must call it too.

If the holdout is ever compromised, every round-over-round comparison the
project has produced becomes meaningless and there is no way to detect it
after the fact. This is the highest-consequence invariant in the codebase.

## Data sources

| Source | What | Where |
|---|---|---|
| CWNS text exports | `PHYSICAL_LOCATION.txt`, `FACILITY_TYPES.txt`, `DISCHARGES.csv`, `POPULATION_WASTEWATER.txt` | `correction/data/cwns/` |
| Regrid parcels | National parcel polygons + attributes, partitioned `state=XX/*.parquet` | HPC: `/work/GRDVULN/data/parcels/`<br>Local mirror: see [02_ENVIRONMENTS.md](02_ENVIRONMENTS.md) |
| NLCD | Annual land cover raster, 2023 | `/work/GRDVULN/data/nlcd/` |
| NAIP imagery | Streamed from Microsoft Planetary Computer (free, no key) | `STAC_URL` in `config.py` |
| Census geography | `tlgdb_2022_a_us_substategeo.gdb` | `/work/GRDVULN/data/Census/` |
| OSM wastewater points | `Wastewater_Plants.gpkg` | `correction/data/reference/` |

**Note on NAIP:** the pipeline originally read local MrSID county mosaics.
USDA's hosted imagery service was retired and the Box-distributed `.sid`
files don't fit on cluster disk, so `01b` was rewritten to stream windowed
COG reads from Planetary Computer. There is no `osgeo.gdal` / MrSID
dependency on the HPC side at all. The local `detection/` side still has
its own NAIP fetch (`detection/pipeline/naip_fetch.py`), also via PC.
