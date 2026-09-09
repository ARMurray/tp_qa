"""
config.py
=========
Single source of truth for the wastewater infrastructure detection pipeline.
Every script imports from here so that tile geometry, the class map, the
export CRS, and all paths are defined exactly once.

Path philosophy:
  - REPO_ROOT and everything under it is resolved RELATIVE to this file, so the
    repo can be cloned or moved without editing code.
  - External inputs that live OUTSIDE the repo (the CWNS geodatabase, the Regrid
    parcel store, population tables) are absolute paths and are the only things
    you should ever need to edit when moving to a new machine.
"""

from pathlib import Path

# ===========================================================================
# 1. REPO-RELATIVE PATHS  (do not hardcode machine paths below this line)
# ===========================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
# .parent.parent, NOT .parent -- config.py lives in detection/pipeline/, but
# the real data/annotation/dataset/models folders are one level up, at
# detection/ itself (confirmed 2026-08-28 via a directory listing after
# 02_extract_tiles.py/03_prepare_dataset.py/convert_tiles_to_500m.py were all
# silently pointed at empty detection/pipeline/data/ and
# detection/pipeline/annotation/ folders that never had anything in them --
# the real tiles were sitting at detection/data/tiles/... the whole time.
# pipeline/ is just where the scripts live; it was never meant to be the
# data root.

DATA_DIR        = REPO_ROOT / "data"
SAMPLES_DIR     = DATA_DIR / "samples"
TILES_DIR       = DATA_DIR / "tiles"
RGB_DIR         = TILES_DIR / "rgb" / "png"
NDWI_DIR        = TILES_DIR / "ndwi"
INFERENCE_DIR   = DATA_DIR / "inference"
METADATA_CSV    = DATA_DIR / "tile_metadata.csv"

ANNOTATION_DIR  = REPO_ROOT / "annotation" / "ls_export"
CLASSES_FILE    = ANNOTATION_DIR / "classes.txt"
DATASET_DIR     = REPO_ROOT / "dataset"
DATASET_YAML    = REPO_ROOT / "dataset.yaml"

# Training outputs write here rather than under the repo. models/runs/ inside
# the repo hit unresolved Windows write failures (permission denied on
# checkpoint saves) even with clean ACLs and Controlled Folder Access off —
# likely an EDR agent behavioral rule on the repo path. This location is
# outside the repo tree and confirmed working. Gitignored either way, since
# training runs were never meant to be committed.
RUNS_DIR = REPO_ROOT / "models" / "runs"
RUN_NAME        = "wwtp_v2"   # 04 trains into runs/{RUN_NAME}; 05 loads its best.pt

# Pipeline artifacts
SAMPLE_GPKG     = SAMPLES_DIR / "training_sample_round2.gpkg"  # written by 01
SAMPLE_LAYER_PLANTS = "plants"
SAMPLE_LAYER_TRI    = "tri"

# TRI raw input (you drop this in; read by 01)
TRI_SOURCE_GPKG   = SAMPLES_DIR / "TRI_Sample.gpkg"
TRI_SOURCE_LAYER  = "tri"
TRI_ID_FIELD      = "TRI_FACILITY_ID"

# ===========================================================================
# 2. EXTERNAL INPUTS  (absolute — the only machine-specific lines)
# ===========================================================================
# Two separate sibling repos live under Github/:
GITHUB = Path(
    "C:/Users/AMURRA02/OneDrive - Environmental Protection Agency (EPA)/Github"
)
LOCATION_CORRECTION = GITHUB / "Location_Correction"   # holds data/Updates.gdb
SEWERSHEDS          = GITHUB / "Sewersheds"            # holds Data/POPULATION_*

UPDATES_GDB       = LOCATION_CORRECTION / "data" / "Updates.gdb"
CWNS_LAYER        = "CWNS_Locations"
POP_FILE_1        = SEWERSHEDS / "Data" / "POPULATION_WASTEWATER.txt"
POP_FILE_2        = SEWERSHEDS / "Data" / "POPULATION_WASTEWATER_CONFIRMED_updated06242024.csv"

# Regrid parcel parquet store, nested as state=XX/{county_geoid}.parquet
PARCEL_BASE = Path(
    "C:/Users/AMURRA02/OneDrive - Environmental Protection Agency (EPA)"
    "/Data/Regrid/Parquet_Storage"
)
PARCEL_ID_FIELD  = "ll_uuid"
PARCEL_WKB_FIELD = "wkb_geometry"

# Optional local counties file. If None, 01 pulls counties via pygris (Census).
COUNTIES_GPKG = None

# ===========================================================================
# 3. NAIP IMAGE SERVICE
# ===========================================================================
NAIP_URL = ("https://gis.apfo.usda.gov/arcgis/rest/services"
            "/NAIP/USDA_CONUS_PRIME/ImageServer")

# Extraction performance
MAX_WORKERS    = 10     # concurrent NAIP requests (tune to network/server)
FETCH_ACQ_DATE = True   # fetch acquisition date once per parcel (1 extra call)

# ===========================================================================
# 4. TILE GEOMETRY  (must stay consistent across extraction and inference)
# ===========================================================================
TILE_SIZE_M   = 500            # tile width/height in meters
OVERLAP_PCT   = 0           # fractional overlap between adjacent tiles
STRIDE_M      = TILE_SIZE_M * (1 - OVERLAP_PCT)   # ~134 m
TARGET_RES_M  = 0.6            # standardize all imagery to 60 cm
IMAGE_PX      = round(TILE_SIZE_M / TARGET_RES_M)  # 333 px per side

# CRS contract:
#   WGS84 (4326) is the EXPORT crs — NAIP is requested with imageSR=4326 so the
#   pixel grid aligns to a lon/lat bbox. The inference pixel->coordinate math
#   depends on this. Do NOT change without updating 05_run_inference.py.
EXPORT_CRS   = 4326
PROJECTED_CRS = 5070           # Albers CONUS (meters) — used for grids/distances

# ===========================================================================
# 5. SAMPLING PARAMETERS  (used by 01_sample_sites.py)
# ===========================================================================
RANDOM_SEED       = 42
TARGET_N_PLANTS   = 190        # correct-only plants
PER_REGION_CAP    = 50         # geographic balancing cap
TARGET_N_TRI      = 50         # industrial hard negatives

# Census region -> state postal codes
REGIONS = {
    "Northeast": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "Midwest":   ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO",
                  "NE", "ND", "SD"],
    "South":     ["DE", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL",
                  "KY", "MS", "TN", "AR", "LA", "OK", "TX"],
    "West":      ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK",
                  "CA", "HI", "OR", "WA"],
}
STATE_TO_REGION = {s: r for r, states in REGIONS.items() for s in states}

# Population bins for stratification (residential population served, 2022)
POP_BIN_BREAKS = [0, 500, 2000, 10000, 50000, 250000, float("inf")]
POP_BIN_LABELS = ["<500", "500-2k", "2k-10k", "10k-50k", "50k-250k", ">250k"]
POP_BIN_WEIGHTS = {          # oversample rare extremes
    ">250k": 4, "50k-250k": 3, "<500": 3, "10k-50k": 2,
    "2k-10k": 1, "500-2k": 1,
}

# ===========================================================================
# 6. CLASS MAP  (YOLO numeric order = alphabetical; keep in sync with LS)
# ===========================================================================
CLASSES = [
    "aeration_basin",    # 0
    "chlorine_contact",  # 1
    "clarifier",         # 2
    "digester",          # 3
    "drying_bed",        # 4
    "oxidation_pond",    # 5
]

# ===========================================================================
# 7. METADATA SCHEMA  (columns 05_run_inference.py / downstream code read)
# ===========================================================================
METADATA_COLUMNS = [
    "tile_id", "source", "CWNS_ID", "TRI_FACILITY_ID",
    "ll_uuid_primary", "ll_uuid_alternates", "st", "geoid",
    "tile_row", "tile_col", "ctr_lon", "ctr_lat",
    "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax",
    "source_res_m", "target_res_m", "image_px", "acq_date",
    "rgb_path", "ndwi_path", "label",
]


def ensure_dirs():
    """Create output directories if missing (safe to call repeatedly)."""
    for d in (SAMPLES_DIR, RGB_DIR, NDWI_DIR, INFERENCE_DIR,
              ANNOTATION_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)


ALL_PLANTS_GPKG    = DATA_DIR / "samples" / "all_plants.gpkg"
ALL_PLANTS_LAYER   = "all_plants"
STATE_TILES_ROOT   = DATA_DIR / "tiles_state"
STATE_INFERENCE_ROOT = DATA_DIR / "inference_state"

PLANTS_RAW_GPKG   = DATA_DIR / "plants.gpkg"      # your R export
PLANTS_RAW_LAYER  = "treatment"