"""
config.py
=========
Local paths for the review app. This app runs entirely on your machine --
no HPC connection needed at review time. Parcel geometry is looked up live
via DuckDB against your local Regrid parquet mirror.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Regrid parcel store (local mirror)
# ---------------------------------------------------------------------------
REGRID_ROOT = Path(
    r"C:\Users\AMURRA02\OneDrive - Environmental Protection Agency (EPA)"
    r"\Data\Regrid\Parquet_Storage"
)

# ADJUST THIS if your local layout doesn't match HPC's state=XX/*.parquet
# pattern. This is the one thing in the whole app I could not verify --
# I don't have visibility into your local folder structure. If parcel
# geometry lookups return nothing, check this glob first.
#
# Expected (matches HPC's PARCEL_BASE convention):
#   Parquet_Storage/state=OH/<county>.parquet
#   Parquet_Storage/state=MS/<county>.parquet
#
# If instead it's flat (no state= subfolders), change this to:
#   REGRID_STATE_GLOB = "{state}/*.parquet"   # or whatever the real pattern is
REGRID_STATE_GLOB = "state={state}/*.parquet"

# Raw Regrid parquet's geometry column name -- confirmed from the HPC
# pipeline's own queries (01a_extract_parcels.py, 02_feature_engineering.py
# both use p.wkb_geometry). Change here if your local mirror differs.
REGRID_GEOM_COL = "wkb_geometry"
REGRID_UUID_COL = "ll_uuid"

# ---------------------------------------------------------------------------
# CWNS facility names (added 2026-08-26) -- separate from the Regrid parcel
# store above, and from a different local folder entirely. A modest flat
# file (30,881 rows), loaded once into memory rather than queried per
# request like the (much larger, state-partitioned) Regrid store.
# ---------------------------------------------------------------------------
FACILITIES_PATH = Path(
    r"C:\Users\AMURRA02\OneDrive - Environmental Protection Agency (EPA)"
    r"\Github\Sewersheds\Data\FACILITIES.txt"
)

# ---------------------------------------------------------------------------
# App data
# ---------------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
INCOMING_DIR = DATA_DIR / "incoming"     # synced down from HPC each round
OUTGOING_DIR = DATA_DIR / "outgoing"     # exported for syncing back to HPC
APP_DB_PATH = DATA_DIR / "app.db"

FRONTEND_DIR = APP_ROOT / "frontend"

# ---------------------------------------------------------------------------
# Review behavior
# ---------------------------------------------------------------------------
TOP_K_SHOWN = 5   # must match 10_build_review_queue.py's TOP_K_SHOWN

for d in (INCOMING_DIR, OUTGOING_DIR):
    d.mkdir(parents=True, exist_ok=True)
