# Wastewater Treatment Plant Location Correction — Object Detection Module

## Project Overview

### Background
This project is part of a broader machine learning pipeline to automatically
correct incorrectly reported coordinates of wastewater treatment plants (WWTPs)
in the United States. An existing random forest model uses text-based fields and
locational data to perform corrections, but roughly 40% of reported plant
coordinates are incorrect. This object detection module adds a computer-vision
component: it detects visible treatment infrastructure in NAIP aerial imagery and
uses the detected locations to derive corrected coordinates.

### Goal
Train an object detection model that identifies specific WWTP infrastructure in
NAIP imagery. At inference, the model returns bounding boxes around detected
infrastructure; box centroids are converted to geographic coordinates and
aggregated per facility to produce a corrected lat/lon, which feeds back into the
random forest pipeline.

### Current state (end of this phase)
The full pipeline is **built and working end-to-end in Python** on a local EPA
Windows laptop, from site sampling through a trained model, inference, corrected
coordinates, and a web map for visual QA. The immediate next effort is porting the
compute-heavy stages (roughly scripts 02–05) to an HPC cluster, then doing more
annotation and running the full loop again. See `CURRENT_STATUS.md`.

---

## Repository Structure

The project was consolidated from two loose folders + two R scripts into a single
git repo. **The repo lives at a NON-OneDrive path** (`C:\Users\AMURRA02\wastewater_infrastructure_detection`)
deliberately — OneDrive should not try to sync thousands of image tiles or a
multi-GB virtual environment.

```
wastewater_infrastructure_detection/          <- git repo root (NON-OneDrive path)
├── config.py                     # SINGLE SOURCE OF TRUTH: paths, tile geometry,
│                                 #   class map, CRS, sampling params, run name
├── .gitignore
├── requirements.txt              # main env (extraction/training/inference)
├── requirements-labelstudio.txt  # Label Studio env (kept separate — see below)
├── dataset.yaml                  # written by 03 (gitignored)
├── pipeline/
│   ├── 01_sample_sites.py        # site selection + stratified sampling
│   ├── 02_extract_tiles.py       # NAIP tile extraction (threaded)
│   ├── 03_prepare_dataset.py     # LS export -> YOLO dataset (plant-level split)
│   ├── 04_train_model.py         # YOLOv8 training
│   ├── 05_run_inference.py       # inference -> corrected coordinates
│   ├── 06_build_map.py           # MapLibre web map for QA
│   └── vendor/                   # vendored JS libs (network-blocked CDNs)
│       ├── maplibre-gl.js        #   MapLibre GL JS 5.24.0 (inlined by 06)
│       ├── maplibre-gl.css
│       └── MAPLIBRE_LICENSE.txt
├── docs/
│   ├── PROJECT_OVERVIEW.md       # this file
│   ├── IMPLEMENTATION_PLAN.md    # phase-by-phase pipeline detail
│   └── CURRENT_STATUS.md         # done / next steps / HPC porting notes
├── data/                         # gitignored (except data/samples/*.gpkg)
│   ├── samples/
│   │   ├── training_sample_round2.gpkg   # written by 01 (layers: plants, tri)
│   │   └── TRI_Sample.gpkg               # RAW INPUT — dropped in manually
│   ├── tiles/
│   │   ├── rgb/png/              # RGB PNG tiles (annotation + training)
│   │   └── ndwi/                # NDWI GeoTIFF tiles (reserved for round 3+)
│   ├── tile_metadata.csv        # written by 02 — the tile index
│   └── inference/               # written by 05 and 06
│       ├── detections.csv / .gpkg
│       ├── corrected_coordinates.csv / .gpkg
│       └── facility_map.html
├── annotation/
│   └── ls_export/               # Label Studio YOLO export
│       ├── labels/              # .txt YOLO label files
│       └── classes.txt          # 6 class names, alphabetical
└── dataset/                     # gitignored — built fresh by 03 each run
    ├── images/train  · images/val
    └── labels/train  · labels/val
```

### Things that live OUTSIDE the repo (intentionally)

| Location | What | Why outside |
|---|---|---|
| `C:\Users\AMURRA02\yolo_env\` | Main Python venv | venvs hardcode absolute paths; can't be moved. Regenerated from `requirements.txt`. |
| `C:\Users\AMURRA02\ls_env\` | Label Studio venv | Separate env — LS pins deps that conflict with ultralytics + torch nightly. |
| `C:\Users\AMURRA02\ml_artifacts\runs\` | **Training outputs / weights** | Moved out of the repo to dodge a Windows write-permission failure (see Known Issues). `RUNS_DIR` in `config.py` points here. |
| `…/OneDrive…/Github/Location_Correction/data/Updates.gdb` | CWNS geodatabase | Large external input; belongs to sibling repo. |
| `…/OneDrive…/Github/Sewersheds/Data/POPULATION_*` | Population tables | Large external input; sibling repo. |
| `…/OneDrive…/Data/Regrid/Parquet_Storage/` | Regrid parcel parquet store | Huge; shared dataset. |

**Only the absolute external-input paths in `config.py` section 2 need editing when
moving machines.** Everything else is resolved relative to the repo root.

---

## Data Sources

### Treatment plant locations
- Source: `Updates.gdb`, layer `CWNS_Locations`
- **Correct-only pool** (chosen deliberately — tiles reliably contain infrastructure):
  - `How_Corrected == "Parcel"` → use `Corrected_X`, `Corrected_Y`
  - `Original_Correct == "Yes"` → use `Original_X`, `Original_Y`
- Population joined from `Sewersheds/Data/POPULATION_WASTEWATER.txt` and
  `POPULATION_WASTEWATER_CONFIRMED_updated06242024.csv` (field
  `TOTAL_RES_POPULATION_2022`)
- Round-2 sample: 190 plants stratified by Census region × population bin

### TRI facility locations (industrial hard negatives)
- Raw input: `data/samples/TRI_Sample.gpkg`, layer `tri`, ID field `TRI_FACILITY_ID`
- Purpose: circular tanks / rectangular industrial structures with no water
  treatment — teaches the model what is NOT a WWTP
- Round-2 sample: 50 facilities stratified by Census region × NAICS sector
  (`INDUSTRY_CODE`), optionally filtered to `ONSITE_WATER == 0` if that column
  is present

### Parcel boundaries
- Regrid parcel data, local parquet files, nested `state=XX/{county_geoid}.parquet`
- Geometry stored as WKB in column `wkb_geometry`; parcel id column `ll_uuid`
- Used for treatment plants only (TRI uses a single fixed tile per point)

### Aerial imagery
- NAIP (National Agriculture Imagery Program), USDA CONUS PRIME ArcGIS REST
  ImageServer: `https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer`
- 4 bands: Red, Green, Blue, NIR; native res 30/60cm, standardized to 60cm
- Two products per tile: RGB (PNG, for annotation) and NDWI (GeoTIFF, reserved)

---

## Infrastructure Classes (6)

YOLO numeric IDs follow alphabetical order (must stay in sync with Label Studio
and `config.py`):

| Class | ID | Description | Notes |
|---|---|---|---|
| `aeration_basin` | 0 | Rectangular mixing tanks | Turbulent brownish water; visually variable; weakest class in round 1 |
| `chlorine_contact` | 1 | Chlorine contact chambers | Elongated rectangular tanks near outlet |
| `clarifier` | 2 | Circular settling tanks | Very distinctive; best class (AP50 0.976 round 1) |
| `digester` | 3 | Covered circular tanks | Like clarifiers but covered/darker |
| `drying_bed` | 4 | Sludge drying beds | Rectangular sandy/brown cells |
| `oxidation_pond` | 5 | Large open lagoons | Greenish water; common at small rural plants; false-positives on natural water |

---

## Tile Design
- **Size:** 200 × 200 m
- **Resolution:** 60 cm → **333 × 333 px** (`IMAGE_PX`)
- **Overlap:** 33% (stride ~134 m) both directions
- **Small parcels:** parcels < 200 m in either dimension get one centered tile
- **TRI:** single centered tile per point (no parcel)
- **Bands:** RGB → PNG (Label Studio needs PNG); NDWI → float GeoTIFF

### Filename convention (THE PIPELINE INDEX — do not change)
```
plants: {CWNS_ID}_{ll_uuid}_r{row:02d}_c{col:02d}_rgb.png
TRI:    TRI_{TRI_FACILITY_ID}_r01_c01_rgb.png
```
This filename is what ties tiles to their Label Studio annotations and lets
`03_prepare_dataset.py` reconstruct everything without depending on the metadata
CSV. Round-1 tiles (3,561 of them) survived the whole repo reorg precisely because
filenames were preserved.

### CRS contract (do not break)
NAIP is exported with `imageSR=4326`, so each tile's pixel grid aligns to a
lon/lat bbox. The inference pixel→coordinate math in `05` depends on this. Do not
change `EXPORT_CRS` without updating `05_run_inference.py` in lockstep.
`PROJECTED_CRS = 5070` (Albers CONUS, meters) is used for tile grids and distances.

---

## Environment

| Item | Detail |
|---|---|
| Machine | EPA laptop, Windows 11 |
| Python | 3.13 (system install under `C:\Program Files\Python313`) |
| Main venv | `C:\Users\AMURRA02\yolo_env\` |
| GPU | NVIDIA RTX Pro 2000 Blackwell Generation Laptop GPU (~8.5 GB VRAM, sm_120) |
| PyTorch | **cu128 nightly required** — Blackwell (sm_120) is not in stable builds |
| Label Studio | separate venv `C:\Users\AMURRA02\ls_env\` |

### Environment setup order (important)
Install torch nightly FIRST, then the rest — otherwise `ultralytics` pulls a
stable CPU/older-CUDA torch that then has to be uninstalled:
```powershell
& "C:\Users\AMURRA02\yolo_env\Scripts\Activate.ps1"
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
```
Verify: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
→ expect `True` and a `cu128`/`dev` version string.

### PowerShell venv activation
Plain `activate` does NOT work in PowerShell. Always use:
```powershell
& "C:\Users\AMURRA02\yolo_env\Scripts\Activate.ps1"
```
Label Studio: activate `ls_env` the same way, then run `label-studio`.

### Running scripts
Run from the repo root, by path (numbered filenames can't be imported as modules,
so `-m` won't work — each script bootstraps `config.py` onto `sys.path` itself):
```powershell
python pipeline/01_sample_sites.py
```

---

## Known Issues & Failure Modes

### Environment / access issues encountered (and resolutions)

| Issue | Status | Detail / resolution |
|---|---|---|
| **Training checkpoint write fails** (`PermissionError [Errno 13]` on `last.pt`) | Worked around | Root cause never definitively confirmed. ACLs were clean (full control), Controlled Folder Access was OFF, no named EDR service found, and raw + 80 MB looped writes to the same folder succeeded — but `torch.save` (which writes a ZIP-of-pickles) failed every time, in multiple folders on two drives. **Strong suspect: an endpoint-security behavioral rule flagging the pickle/zip write pattern.** Two mitigations applied: (1) `RUNS_DIR` moved out of the repo to `C:\Users\AMURRA02\ml_artifacts\runs`; (2) forced legacy (non-zip) torch serialization. Training then completed. **A clean long-term fix is an EPA IT exclusion for `python.exe` / the training folder — worth a ticket. This issue is Windows/EDR-specific and should not exist on Linux HPC.** |
| Cannot write to `C:\` root | Expected | Non-admin account; create folders under the user profile instead. |
| venv can't be moved into repo | By design | `Activate.ps1` and `pyvenv.cfg` hardcode absolute paths. venvs stay outside the repo, captured by `requirements*.txt`. |
| pandas `groupby().apply()` drops grouping columns | Fixed | Newer pandas (2.2+/3.x) excludes grouping columns inside `.apply`, which stripped `region`/`pop_bin` and raised `KeyError: 'region'` in `01`. Rewritten to iterate groups explicitly and `pd.concat`. |
| Population path wrong | Fixed | Pop files are under `Github/Sewersheds/Data/`, NOT `Location_Correction`. `config.py` now has separate `LOCATION_CORRECTION` and `SEWERSHEDS` roots under one `GITHUB` base. |
| torch nightly resolution | Documented | Must install cu128 nightly BEFORE `requirements.txt` (see setup order). |

### Web-map / network issues (script 06)

| Issue | Status | Detail / resolution |
|---|---|---|
| `unpkg.com` (JS CDN) blocked | Fixed | MapLibre `maplibre-gl.js`/`.css` **vendored into `pipeline/vendor/`** and inlined directly into the generated HTML — output makes zero outbound requests for the library. |
| `file://` blank map | Documented | MapLibre spawns a Web Worker; Chrome/Edge block workers on `file://` pages. **Must serve locally:** `cd data/inference && python -m http.server 8000`, then open `http://localhost:8000/facility_map.html` (not double-click). |
| CARTO vector basemap approach fragile | Fixed | Basemap switching via `setStyle()` destroyed the data layers (points vanished on toggle). Rebuilt as a SINGLE style holding both basemaps as raster layers; the toggle flips layer `visibility` via `setLayoutProperty` — data layers are added once and never removed/reordered. |
| Basemap host | Note | Both basemaps (streets + imagery) are Esri raster tiles from `server.arcgisonline.com` — one host to clear through the proxy. Fallback `OFFLINE_STYLE` (flat background) is in `06` if that host is ever blocked. |

### Model failure modes (from round-1 model, wwtp_v1)
| Issue | Notes |
|---|---|
| Oxidation-pond false positives on natural water | Primary failure; mitigated by TRI hard negatives + golf-course empty annotations |
| Duplicate boxes on dense facilities | Inference-time confidence threshold tuning; handled by geographic NMS in `05` |
| `aeration_basin` weak (AP50 0.396) | Visually variable; needs more labeled examples |

---

## Downstream Use
`05_run_inference.py` produces, per detected object: class ID, confidence, and a
geographic coordinate (box centroid converted from pixel space, with a y-axis flip
because image row 0 = north = `bbox_ymax`). Objects seen across overlapping tiles
are merged by geographic NMS (~12 m, per class). One corrected coordinate per plant
is then derived (default: confidence-weighted centroid of all detected objects) and
written to `corrected_coordinates.gpkg` for the random forest pipeline. When the
sample gpkg is available, an `offset_m` column reports how far each correction moved
from the reported point.
