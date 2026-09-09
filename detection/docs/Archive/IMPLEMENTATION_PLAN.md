# Implementation Plan — NAIP Object Detection Pipeline

The pipeline is fully Python, six numbered scripts in `pipeline/`, all reading
shared constants from `config.py`. Each script bootstraps the repo root onto
`sys.path` and does `import config as C`. Run them in order, from the repo root:

```powershell
python pipeline/01_sample_sites.py      # -> data/samples/training_sample_round2.gpkg
python pipeline/02_extract_tiles.py     # -> data/tiles/... + data/tile_metadata.csv
# ... annotate new tiles in Label Studio, export YOLO to annotation/ls_export/ ...
python pipeline/03_prepare_dataset.py   # -> dataset/ + dataset.yaml
python pipeline/04_train_model.py       # -> ml_artifacts/runs/{RUN_NAME}/weights/best.pt
python pipeline/05_run_inference.py     # -> data/inference/{detections,corrected_coordinates}.*
python pipeline/06_build_map.py         # -> data/inference/facility_map.html
```

`config.py` is the single source of truth. Key constants: `TILE_SIZE_M=200`,
`OVERLAP_PCT=0.33`, `TARGET_RES_M=0.6`, `IMAGE_PX=333`, `EXPORT_CRS=4326`,
`PROJECTED_CRS=5070`, the 6-class `CLASSES` list, `RANDOM_SEED=42`,
`RUN_NAME="wwtp_v2"`, and all paths.

---

## Script 01 — `01_sample_sites.py` (site selection + sampling)

Consolidates the old R `create_training_plants.R` (pool build) and the sampling
block that was buried in `get_training_images.R`. In the R version those two
halves were never actually connected — the extraction loop ignored the sample.
This is fixed here: `01` writes the sample that `02` reads.

Steps:
1. Load `CWNS_Locations` from `Updates.gdb`
2. Build **correct-only** point set: `How_Corrected=="Parcel"` → `Corrected_X/Y`;
   `Original_Correct=="Yes"` → `Original_X/Y`; dedup on `CWNS_ID`
3. Attach county `geoid` via spatial join to Census counties (pygris download, or
   a local `COUNTIES_GPKG` if set) — needed to locate the parcel parquet file
4. Join residential population; bin; stratified weighted draw by region × pop bin,
   ~50/region, capped at **190 plants** (rare pop extremes oversampled by weight)
5. Sample **50 TRI** hard negatives from `TRI_Sample.gpkg`, region × NAICS sector
6. Write `data/samples/training_sample_round2.gpkg`, layers `plants` and `tri`

Notes:
- **Fresh draw:** Python's RNG ≠ R's, so the exact 190 plants differ from any
  prior R sample. Nothing downstream depends on the specific draw; round-1 tiles
  and annotations are keyed by filename and untouched.
- Sampling uses explicit group iteration + `pd.concat`, NOT `groupby().apply()`
  (newer pandas drops grouping columns inside `.apply`, which raised
  `KeyError: 'region'`).

---

## Script 02 — `02_extract_tiles.py` (NAIP tile extraction)

Pure extraction. Reads the sample, produces tiles + metadata. **Threaded** for
speed (network-bound work).

Steps:
1. Read `plants` and `tri` layers from the sample
2. For each county, load Regrid parcels from parquet (only `ll_uuid` + WKB
   geometry columns), decode WKB, reproject to EPSG:5070
3. Spatial-join plants → parcels; take primary parcel per plant, track alternates
4. Generate overlapping 200 m tile grid over each parcel (small parcels → 1 tile);
   TRI facilities → 1 centered tile, no parcel
5. Batch-reproject tile bboxes 5070 → 4326 (one transform per parcel)
6. Fetch 4-band NAIP from the `exportImage` endpoint (imageSR=4326)
7. Save RGB (bands 1-3) as PNG, NDWI = (Green−NIR)/(Green+NIR) as float GeoTIFF
8. Append one metadata row per tile to `data/tile_metadata.csv`

Performance design (relevant for HPC porting):
- Image fetches run **concurrently in a `ThreadPoolExecutor`** (`C.MAX_WORKERS`,
  default 10). Threads work because the bottleneck is HTTP wait (GIL released).
  Each worker thread keeps its own `requests.Session`.
- **Acquisition date fetched once per parcel**, not per tile (all tiles on a
  parcel share one NAIP flight). Toggle `C.FETCH_ACQ_DATE`.
- Parcel parquet read is column-limited; per-parcel bbox reprojection is batched.

Safeguards:
- **SKIP-EXISTING:** a tile whose RGB PNG already exists is never re-fetched
  (protects round-1 tiles + annotations).
- **ADDITIVE METADATA:** new rows appended; tiles already in the CSV are skipped.

### Metadata schema (`tile_metadata.csv`)
`tile_id`, `source`, `CWNS_ID`, `TRI_FACILITY_ID`, `ll_uuid_primary`,
`ll_uuid_alternates`, `st`, `geoid`, `tile_row`, `tile_col`, `ctr_lon`, `ctr_lat`,
`bbox_xmin`, `bbox_ymin`, `bbox_xmax`, `bbox_ymax`, `source_res_m`, `target_res_m`,
`image_px`, `acq_date`, `rgb_path`, `ndwi_path`, `label`

---

## Script 03 — `03_prepare_dataset.py` (dataset assembly)

1. Load class names from `annotation/ls_export/classes.txt`
2. Parse Label Studio YOLO export filenames (handles two patterns: URL-encoded
   path and clean hash-prefixed), match labels to source RGB tiles by tile stem
3. **Plant-level train/val split** (by `CWNS_ID`, so no plant is in both splits —
   prevents spatial autocorrelation inflating val metrics), 20% val
4. Copy images/labels into `dataset/images/{train,val}` and
   `dataset/labels/{train,val}`, rebuilding `dataset/` from scratch each run
5. Write `dataset.yaml`

Local knobs: `VAL_FRACTION=0.2`, `INCLUDE_NEGATIVES=True` (empty label files kept
as negatives).

---

## Script 04 — `04_train_model.py` (training)

YOLOv8s (`yolov8s.pt`, pretrained COCO), transfer-learned.
- `imgsz = C.IMAGE_PX` (333) — training size can never drift from tile size
- batch 16, 100 epochs, patience 20, Adam, LR 0.001
- Augmentation tuned for aerial (90° rotation, both flips, mosaic, scale jitter)
- Writes to `C.RUNS_DIR / C.RUN_NAME` (i.e. `ml_artifacts/runs/wwtp_v2/`);
  `exist_ok=False` so it won't clobber a prior run
- Validates with best weights at the end, prints per-class AP50

**Windows-specific workaround in play:** the checkpoint-write permission failure
(see PROJECT_OVERVIEW Known Issues) was mitigated by (a) `RUNS_DIR` living outside
the repo and (b) forcing legacy non-zip torch serialization. On Linux HPC this
workaround is expected to be unnecessary and can likely be removed.

### Round-1 baseline (wwtp_v1)
mAP50 0.733, mAP50-95 0.429, P 0.607, R 0.636. Per-class AP50: clarifier 0.976,
chlorine_contact 0.828, digester 0.762, drying_bed 0.745, oxidation_pond 0.689,
aeration_basin 0.396. Primary failure: oxidation-pond false positives on natural
water.

---

## Script 05 — `05_run_inference.py` (inference + coordinate correction)

1. Load best weights from `C.RUNS_DIR / C.RUN_NAME / weights/best.pt` (falls back
   to newest `best.pt` under `RUNS_DIR` if that exact path is missing)
2. Read `tile_metadata.csv`; filter to plant tiles (TRI excluded by default);
   resolve each tile's RGB PNG (by `tile_id`, falling back to metadata `rgb_path`)
3. Run batched, streamed prediction (`conf=0.25`, `iou=0.50`, `imgsz=IMAGE_PX`)
4. Convert each box center (px,py) → (lon,lat) from the tile's WGS84 bbox, with a
   **y-axis flip** (image row 0 = top = north = `bbox_ymax`)
5. **Geographic NMS** per plant and class (~12 m): merge duplicates seen across
   overlapping tiles into confidence-weighted object centroids
6. Derive one corrected coordinate per plant (default `weighted_centroid`; also
   supports `highest_confidence` and `closest_to_original`)
7. Write outputs and, if the sample gpkg is available, an `offset_m` (distance
   from reported point) per plant

### The two outputs
- **`detections.csv` / `.gpkg`** — every unique detected object (post-dedup), one
  point each, with class, confidence, `n_merged`, contributing `tile_ids`.
- **`corrected_coordinates.csv` / `.gpkg`** — one row per plant: corrected lon/lat,
  selection method, object/detection counts, dominant class, confidences, class
  breakdown, and `offset_m` when available.

GeoPackage output needs geopandas; without it, CSVs are still written.

---

## Script 06 — `06_build_map.py` (QA web map)

Builds a self-contained MapLibre GL JS map from the `05` outputs:
- Reported facility locations (sample `plants` layer, filtered to CWNS_IDs that
  have inference results) as dark points
- Detected infrastructure (`detections.csv`) as points colored by class (colors
  match the Label Studio schema)
- Click popups; confidence-scaled radii; **streets/imagery basemap toggle**

Implementation notes (hard-won — see PROJECT_OVERVIEW Known Issues):
- MapLibre JS/CSS are **vendored in `pipeline/vendor/` and inlined** into the HTML
  (the `unpkg.com` CDN is blocked on this network). Output makes no outbound
  request for the library.
- Both basemaps are Esri raster tiles (`server.arcgisonline.com`) held in ONE
  style; the toggle flips layer `visibility` (no `setStyle` — that destroyed the
  data layers). `OFFLINE_STYLE` fallback is available if the host is blocked.
- **Must be served over HTTP, not `file://`** (MapLibre Web Worker restriction):
  `cd data/inference && python -m http.server 8000`.

---

## HPC Porting Plan (NEXT PHASE — scripts ~02–05)

The compute-heavy stages move to the HPC cluster. Anticipated work and gotchas:

### General
- **Paths:** `config.py` external-input paths are Windows absolute paths. Add an
  HPC path set (env var, hostname switch, or a separate config profile). `pathlib`
  handles separators, but the absolute roots must change.
- **Environments:** replace the Windows venv with the cluster's module system /
  conda. Rebuild from `requirements.txt`, but pin CUDA to whatever the HPC GPUs
  need (the Blackwell cu128 nightly is laptop-specific).
- **The Windows-only workarounds drop away on Linux:** the training checkpoint
  permission failure, the legacy-torch-save workaround, and `RUNS_DIR`-outside-repo
  were all Windows/EDR artifacts. On HPC, `RUNS_DIR` can likely return to a normal
  project/scratch location.

### Script 02 (extraction) — the biggest rethink
- **Compute nodes often have no outbound internet.** NAIP `exportImage`, the pygris
  Census download (in 01), and Esri tiles (06) all need network. Options: run
  extraction on a login/data-transfer node, pre-stage all imagery, or use a
  cluster proxy. **Confirm node network policy first — this drives the design.**
- **Parallelism:** the current in-process `ThreadPoolExecutor` can be kept, but the
  natural HPC pattern is a **job array** sharding by county (plants) or by facility
  (TRI). Additive/skip-existing metadata already makes sharded, resumable runs
  safe, BUT concurrent appends to one `tile_metadata.csv` from array tasks would
  race — write per-shard metadata files and merge, or use a lock.
- **Parcel parquet + `Updates.gdb` access** must be staged onto cluster storage.

### Scripts 03–05
- **03** is CPU/file I/O — trivial to port; just paths + env.
- **04 / 05** are GPU — the main win of HPC. Bigger GPUs allow larger batch / model
  size (yolov8m/l). Remove the legacy-torch-save workaround; verify CUDA/torch on
  the cluster GPUs. `05` can shard inference across tiles as a job array the same
  way `02` shards extraction.
- **06** stays local (it's a viewer); run it wherever you have a browser + the
  `05` outputs.

### Suggested sequencing for the HPC phase
1. Stand up the env + paths on HPC; smoke-test `03` (no network, no GPU quirks).
2. Resolve compute-node network policy; port `02` accordingly (array vs. single).
3. Port `04`/`05`; retrain `wwtp_v2` on more-annotated data at HPC scale.
4. Return to annotation, expand the labeled set, rerun the full loop.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Extraction language | Python (was R) | One toolchain; sample now actually feeds extraction |
| Plant pool | Correct-only | Tiles reliably contain infrastructure to annotate |
| Hard negatives | TRI facilities | Industrial confusers with no water treatment |
| Tile size / overlap | 200 m / 33% | Whole facilities captured; objects appear in ≥1 tile |
| Resolution | 60 cm (333 px) | Standardized across variable NAIP source res |
| Export CRS | 4326 | Pixel grid aligns to lon/lat; inference math depends on it |
| Train/val split | By CWNS_ID | Prevents spatial autocorrelation inflating val metrics |
| Central `config.py` | One source of truth | Kills the "R said 4326, Python assumed Mercator" bug class |
| Training outputs outside repo | `ml_artifacts/runs/` | Windows write-permission workaround (revisit on HPC) |
| Vendored MapLibre | `pipeline/vendor/` | CDN blocked on EPA network |
| NDWI timing | Deferred to round 3+ | More labels help more than a 4th channel now |
