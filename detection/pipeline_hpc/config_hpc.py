"""
config_hpc.py
=============
HPC-side counterpart to the local config.py. Same contract (every HPC script
imports from here so tile geometry, the class map, the export CRS, and all
paths are defined exactly once) but trimmed/fixed for the cluster:

  - Windows-only local-training paths (RUNS_DIR, GITHUB, LOCATION_CORRECTION,
    SEWERSHEDS, UPDATES_GDB, POP_FILE_*, RUN_NAME) are NOT carried over here --
    they're meaningless on Linux and ensure_dirs() would try to mkdir a
    Windows drive-letter path as a bogus relative folder if they were. Local
    training (04_train_model.py, 05_run_inference.py, 01_sample_sites.py)
    stays on config.py on the Windows machine; those scripts never run here.
  - Everything actually read by the HPC scripts (00_build_full_plant_list_hpc.py,
    07_run_state_pipeline_hpc.py, state_fips_hpc.py) is present below.

Path philosophy:
  - REPO_ROOT and everything under it is resolved RELATIVE to this file, so
    the repo can be cloned or moved without editing code.
  - External inputs that live OUTSIDE the repo (the Regrid parcel store, a
    local counties file, the models directory) are absolute paths and are
    the only things you should need to edit when moving to a new HPC path.
"""

from pathlib import Path

# ===========================================================================
# 1. REPO-RELATIVE PATHS  (do not hardcode machine paths below this line)
# ===========================================================================
REPO_ROOT = Path(__file__).resolve().parent

DATA_DIR        = REPO_ROOT / "data"
SAMPLES_DIR      = DATA_DIR / "samples"

# --- Full production plant universe (00_build_full_plant_list_hpc.py) -----
PLANTS_RAW_GPKG  = DATA_DIR / "plants.gpkg"     # your R export, uploaded as-is
PLANTS_RAW_LAYER = "treatment"
ALL_PLANTS_GPKG  = SAMPLES_DIR / "all_plants.gpkg"   # written by 00, read by 07
ALL_PLANTS_LAYER = "all_plants"

# --- Per-state pipeline outputs (07_run_state_pipeline_hpc.py) ------------
STATE_TILES_ROOT      = DATA_DIR / "tiles_state"       # transient, self-pruning per state
STATE_INFERENCE_ROOT  = DATA_DIR / "inference_state"    # Parquet tables: tiles/detections/objects/plants

# ===========================================================================
# 2. EXTERNAL INPUTS  (absolute — the only machine-specific lines)
# ===========================================================================
# Regrid parcel parquet store, nested as state=XX/{county_geoid}.parquet
PARCEL_BASE      = Path("/work/GRDVULN/data/parcels")
PARCEL_ID_FIELD  = "ll_uuid"
PARCEL_WKB_FIELD = "wkb_geometry"

# Local counties file (avoids the pygris/Census-download network dependency
# on a compute node that may have no outbound internet).
COUNTIES_GPKG = Path("/work/GRDVULN/infrastructure/data/counties.gpkg")

# Trained model weights land here (uploaded from the local training machine).
# 07_run_state_pipeline_hpc.py auto-selects the most recently modified .pt
# in this folder unless --weights pins a specific one.
MODELS_ROOT = Path("/work/GRDVULN/infrastructure/models")

# ===========================================================================
# 3. NAIP IMAGE SERVICE
# ===========================================================================
NAIP_URL = ("https://gis.apfo.usda.gov/arcgis/rest/services"
            "/NAIP/USDA_CONUS_PRIME/ImageServer")

MAX_WORKERS = 10   # concurrent NAIP requests -- 07's --workers CLI flag
                    # currently defaults independently to 10 rather than
                    # reading this; kept in sync by hand for now.

# ===========================================================================
# 4. TILE GEOMETRY  (must stay consistent with the local config.py's copy --
#    extraction and inference math depend on this matching exactly)
# ===========================================================================
TILE_SIZE_M   = 200            # tile width/height in meters
OVERLAP_PCT   = 0.33           # fractional overlap between adjacent tiles
STRIDE_M      = TILE_SIZE_M * (1 - OVERLAP_PCT)   # ~134 m
TARGET_RES_M  = 0.6            # standardize all imagery to 60 cm
IMAGE_PX      = round(TILE_SIZE_M / TARGET_RES_M)  # 333 px per side

# CRS contract:
#   WGS84 (4326) is the EXPORT crs — NAIP is requested with imageSR=4326 so the
#   pixel grid aligns to a lon/lat bbox. The inference pixel->coordinate math
#   depends on this. Do NOT change without updating 07_run_state_pipeline_hpc.py.
EXPORT_CRS    = 4326
PROJECTED_CRS = 5070           # Albers CONUS (meters) — used for grids/distances

# ===========================================================================
# 5. CLASS MAP  (YOLO numeric order = alphabetical; keep in sync with LS
#    and with whatever KEEP_CLASSES filter 04_train_model.py used for the
#    model currently sitting in MODELS_ROOT -- this list itself isn't read
#    by the HPC scripts, since class names come from the loaded model, but
#    kept here for reference/consistency with the local config.py)
# ===========================================================================
CLASSES = [
    "aeration_basin",    # 0
    "chlorine_contact",  # 1
    "clarifier",         # 2
    "digester",          # 3
    "drying_bed",        # 4
    "oxidation_pond",    # 5
]


def ensure_dirs():
    """Create output directories if missing (safe to call repeatedly).
    Not currently called by any _hpc script (each creates its own working
    dirs as needed), but kept for parity with local config.py / manual use."""
    for d in (SAMPLES_DIR, STATE_TILES_ROOT, STATE_INFERENCE_ROOT):
        d.mkdir(parents=True, exist_ok=True)
