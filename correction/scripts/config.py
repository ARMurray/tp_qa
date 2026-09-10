"""
config.py  (HPC / correction)
==============================
Single source of truth for every path and tunable used by the correction
pipeline on the HPC. All five scripts (01a, 01b, 02, 03, 04) import this and
nothing else touches the filesystem directly -- so moving the pipeline to a
new cluster path means editing ROOT below and nothing else.

Layout under ROOT (/work/GRDVULN/correction):
    scripts/          this file + 01a/01b/02/03/04 + slurm wrappers
    logs/             all SLURM .log output
    data/
        cwns/         CWNS text exports (PHYSICAL_LOCATION.txt, etc)
        training/     training_locations.gpkg + Updates.gdb upload
        nlcd_outputs/ 01a per-state parcel/NLCD parquet
        od_output/    01b's four tables (tiles/detections/objects/plants)
        features/     02's output parquet tables
        reference/    census gdb, OSM points, NLCD raster if kept local
    models/           03/04 trained models + diagnostics

EXTERNAL (not under ROOT, shared across projects):
    /work/GRDVULN/data/parcels/     Regrid parquet store, state=XX/*.parquet
    /work/GRDVULN/data/nlcd/        NLCD raster

DIFFERENCES FROM THE LOCAL config.py:
  - No NAIP .sid / MrSID paths. On the HPC, 01b streams NAIP from Microsoft
    Planetary Computer instead of reading local county mosaics -- there is no
    osgeo.gdal/arcgispro-py3-clone dependency here at all. (See the 2026-08-19/20
    session notes: USDA's hosted imagery service is gone, local .sid storage
    doesn't fit, and PC streaming tested clean on this cluster.)
  - detection/ is NOT ported. Object-detection model TRAINING stays local;
    only a trained best.pt gets uploaded to MODELS_DIR for 01b to consume.
"""

from pathlib import Path

# ===========================================================================
# 1. ROOT  (the only line that changes if the pipeline moves)
# ===========================================================================
ROOT = Path("/work/GRDVULN/tp_qa/correction")

SCRIPTS_DIR = ROOT / "scripts"
LOGS_DIR    = ROOT / "logs"
DATA_DIR    = ROOT / "data"
MODELS_DIR   = ROOT / "models"                       # 03/04 write stage1_*/stage2_* here
OD_MODEL_DIR = MODELS_DIR / "object_detection"       # best.pt from detection/ lands here

# --- data subdirectories ---------------------------------------------------
CWNS_DIR           = DATA_DIR / "cwns"           # PHYSICAL_LOCATION.txt, FACILITY_TYPES.txt, DISCHARGES.csv, POPULATION_WASTEWATER.txt
TRAINING_DIR       = DATA_DIR / "training"       # training_locations.gpkg, Updates.gdb
NLCD_OUTPUT_DIR    = DATA_DIR / "nlcd_features"  # 01a output: nlcd_{STATE}_k{K}.parquet
OD_OUTPUT_DIR      = DATA_DIR / "od_features"    # 01b output: tiles/ detections/ objects/ plants/
                                                    # (reported/possibly-wrong locations)
OD_OUTPUT_DIR_CORRECTED = DATA_DIR / "od_features_corrected"  # 01c output: same schema,
                                                    # but for corrections-bin plants' TRUE
                                                    # (Corrected_X/Y) locations -- kept in a
                                                    # separate root deliberately, so no existing
                                                    # reader of OD_OUTPUT_DIR ever accidentally
                                                    # unions reported and corrected-location rows
                                                    # for the same CWNS_ID together.
FEATURES_OUTPUT_DIR = DATA_DIR / "features"      # 02 output: 05_/10_/14_/15_*.parquet
REFERENCE_DIR      = DATA_DIR / "reference"      # census gdb, OSM gpkg

# ===========================================================================
# 2. EXTERNAL INPUTS  (shared, outside ROOT)
# ===========================================================================
PARCEL_BASE      = Path("/work/GRDVULN/data/parcels")   # state=XX/{county_geoid}.parquet
PARCEL_ID_FIELD  = "ll_uuid"
PARCEL_WKB_FIELD = "wkb_geometry"

NLCD_PATH = Path("/work/GRDVULN/data/nlcd/Annual_NLCD_LndCov_2023_CU_C1V1.tif")

# Census gdb -- shared/external, same convention as PARCEL_BASE/NLCD_PATH
# above (moved here 2026-08-21; previously assumed uploaded into
# correction/data/reference/, but it actually lives in the shared data tree).
CENSUS_GDB = Path("/work/GRDVULN/data/Census/tlgdb_2022_a_us_substategeo.gdb")

# ===========================================================================
# 3. TRAINING LABELS
# ===========================================================================
# Built by build_training_bins.py from the dated CWNS_Locations layer in
# Updates.gdb. Layers: 'classes' (Correct/Incorrect, geometry = reported
# Original_X/Y), 'corrections' (Original_X/Y + Corrected_X/Y), 'unverified'.
TRAINING_GPKG              = TRAINING_DIR / "training_locations.gpkg"
TRAINING_LAYER_CLASSES     = "classes"
TRAINING_LAYER_CORRECTIONS = "corrections"
TRAINING_LAYER_UNVERIFIED  = "unverified"

# The master manual-correction file, uploaded from the local machine. Only
# build_training_bins.py reads this; everything else reads TRAINING_GPKG.
MASTER_GDB   = TRAINING_DIR / "Updates.gdb"
MASTER_LAYER = "CWNS_Locations_20260820"

# ===========================================================================
# 4. REFERENCE LAYERS (optional -- 02 degrades gracefully if absent, but
#    silently: name-match features just come back empty/False, which looks
#    like a real negative to the models. Upload this.)
# ===========================================================================
OSM_PATH = REFERENCE_DIR / "Wastewater_Plants.gpkg"

# ===========================================================================
# 5. TILE / IMAGERY GEOMETRY  (01b)
# ===========================================================================
# TARGET_RES_M must match detection/config.py exactly -- the model was trained
# at this resolution (apparent object size in pixels), and changing it would
# feed the model images unlike anything it learned on.
#
# TILE_SIZE_M and OVERLAP_M are intentionally DIFFERENT from the detection
# training pipeline's 200m/33%: here we're tiling to fully cover an
# arbitrarily-shaped reported-location PARCEL, which the training pipeline
# never had to do, so tile footprint is chosen independently. IMAGE_PX scales
# proportionally to TILE_SIZE_M so the model still sees objects at the
# resolution it was trained on -- only the FIELD OF VIEW per tile changes.
TARGET_RES_M  = 0.6
TILE_SIZE_M   = 500
OVERLAP_M     = 50                              # absolute meters, not a %
STRIDE_M      = TILE_SIZE_M - OVERLAP_M         # 450m
IMAGE_PX      = round(TILE_SIZE_M / TARGET_RES_M)   # 833 px per side

EXPORT_CRS    = 4326   # WGS84
PROJECTED_CRS = 5070   # Albers CONUS (meters) -- tile bboxes + point-in-parcel

# Geographic dedup: same-class detections within this distance = one object.
# About merging duplicate detections of one real-world object across
# overlapping tiles -- unaffected by tile size.
NMS_DISTANCE_M = 12.0

CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.50

# Planetary Computer STAC endpoint (01b). Free, no API key required.
# Replaces the local config's NAIP_CCM_* / NAIP_URL block entirely: USDA's
# ImageServer is dead, and the Box-distributed county .sid mosaics don't fit
# on disk here. 01b streams windowed COG reads instead, so none of the
# MrSID/osgeo.gdal machinery is carried over to the HPC.
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Concurrent NAIP fetch threads. Note the local config's MAX_WORKERS = 4 was
# tuned around the EPA network's TLS-inspection proxy dropping connections;
# that constraint does not apply on the cluster, where scaling tested roughly
# linear to 47 workers (2026-08-20). Watch 01b's fetch-failure count.
NAIP_WORKERS = 32

# ===========================================================================
# 6. CANDIDATE SEARCH (01a / 02)
# ===========================================================================
# k=18 rings. NOTE: "~10km" in earlier comments was the DIAMETER, not the
# radius -- H3 res-9 cells are ~174 m on edge, so adjacent centers sit ~302 m
# apart and k=18 is a search RADIUS of roughly 5.4 km. Mean correction distance
# is ~5.05 km, i.e. the average correction lands near the window boundary.
# Measure before changing: 08_diagnose_candidate_coverage.py reports the
# observed meters-per-ring and how many corrections fall outside. This matches
# the R pipeline's standard per HPC_NOTES.md.
# HPC_NOTES.md. This drives 01a's candidate extraction AND Stage 2's
# candidate set size -- the local OH run produced 73k Stage 2 candidates
# off this radius, so it is not a knob to change casually.
K_RINGS = 18

# Some Regrid parcels are digitization artifacts, not real parcels -- an
# entire town represented as one polygon where actual parcel boundaries were
# never transferred in. Confirmed by inspection 2026-08-20: parcels on the
# order of 50 sq mi (~130 km2) showing up as OD "containing parcels" and
# (via 01a) Stage 2 candidates, which is useless as a location signal (an
# "in_parcel" check against a town-sized boundary passes for nearly anything
# nearby) and expensive (01b's tile grid over a 50 sq mi footprint would
# generate thousands of 500m tiles; 01a's NLCD zonal extraction over the
# same footprint is proportionally expensive too).
#
# 2 km2 (~494 acres, ~0.77 sq mi) is a starting default -- generous enough to
# cover legitimately large lagoon/oxidation-pond treatment systems (which
# need far more land than a mechanical plant), while excluding "entire town"
# artifacts by a wide margin (65x smaller than the 50 sq mi example above).
# This is a judgment call, not a measured value -- the better way to set it
# is empirically, from the actual area distribution of parcels matched to
# ALREADY-VERIFIED-CORRECT plants (the 'classes' layer's Correct rows have
# real, trustworthy parcel matches to check this against). Revisit once
# there's more than one state's worth of real Correct-bin parcel areas to
# look at.
MAX_PARCEL_AREA_M2 = 2_000_000

# generate_tile_grid() filters tiles to those that actually intersect the
# parcel's real polygon shape (not just its bounding box) -- see the
# function's own docstring for why that matters. This means a long, thin,
# diagonal parcel (confirmed 2026-08-21 on a real KY treatment plant strung
# out along a road, 1.24 km2 area comfortably under the area cap) tiles
# correctly along its actual length instead of either exploding into
# hundreds of mostly-empty bbox tiles OR getting discarded -- both of which
# earlier versions of this cap did, and both of which were wrong for a
# legitimately-shaped large plant.
#
# MAX_TILES_PER_PLANT is a LAST-RESORT safety net for pathological cases
# that still produce too many tiles even after real-polygon filtering.
#
# 150 (raised from an initial 60 on 2026-08-21): a thin DIAGONAL parcel
# crossing an axis-aligned tile grid touches many cells purely from the
# geometry of the "staircase" effect -- its bounding box is roughly square
# (x-extent ~= y-extent for a ~45deg line) even though the actual parcel is
# a tiny fraction of that area, so tile count scales with bbox dimensions,
# not the thin area itself. The KY plant that surfaced this needs 121 tiles
# for genuinely full coverage of its real ~8.5km diagonal extent -- and at
# ~7s/tile over 32 concurrent workers, that's ~27s of one plant's fetch
# time, not remotely expensive at this pipeline's scale. Capping it away
# was the wrong tradeoff for a legitimate large plant; 150 gives real
# headroom for shapes like this while still catching truly pathological
# geometry (a genuinely enormous or degenerate shape needing hundreds+
# more).
MAX_TILES_PER_PLANT = 150

# When a matched parcel exceeds MAX_PARCEL_AREA_M2 (checked BEFORE tiling --
# a whole-town digitization artifact isn't worth tiling at all) or, after
# real-polygon-filtered tiling, still exceeds MAX_TILES_PER_PLANT, 01b does
# not trust the parcel boundary -- it substitutes a small synthetic square
# centered on the plant's reported point instead, sized to one tile's worth
# of coverage rather than inventing a new constant.
CAPPED_PARCEL_FALLBACK_HALFWIDTH_M = TILE_SIZE_M / 2

# ===========================================================================
# 7. DETECTION CLASSES  (names come from the loaded model at runtime; this
#    list fixes the FEATURE SCHEMA so it stays stable across model retrains)
# ===========================================================================
CLASSES = [
    "aeration_basin",
    "chlorine_contact",
    "clarifier",
    "digester",
    "drying_bed",
    "oxidation_pond",
]

# ===========================================================================
# 8. NAME-MATCHING KEYWORD LISTS  (copied verbatim from the local config.py --
#    keep in sync; these change feature VALUES, not just schema)
# ===========================================================================
OWNER_KEYWORDS = [
    'area','auth','authority','bay','beach','board','borough',
    'city','co','commissioners','council','county','dist','district',
    'falls','fort','lake','metro','municipal','new','north','plant',
    'public','regional','river','san','sanitary','sanitation',
    'service','sewage','sewer','sewerage','state','town','township',
    'treatment','utilities','utility','village','wastewater',
    'water','works',
]
WW_KEYWORDS = [
    'wwtp','stp','system','cs','collection','wwtf','sewer','sewers',
    'wastewater','sd','wwt','stormwater','wpcp','sewerage','water',
    'plant','sanitary','treatment','sewage','decentralized','potw',
    'management','wrp','authority','sanitation',
]
# Cross-field keyword pattern used at the PARCEL level (has_ww_keyword) --
# a different, narrower list than OWNER_KEYWORDS/WW_KEYWORDS above, which
# are used for training-row NAME MATCHING against geography/owner text.
# Kept separate deliberately, matching the R pipeline's two distinct patterns.
PARCEL_WW_KEYWORDS = [
    "sewer", "sewage", "sewerage", "wastewater", "treatment",
    "disposal", "sanitary", "sanitation", "potw", "wwtp",
    "wwtf", "effluent", "lagoon", "digester", "biosolid", "wpcp",
]


# ===========================================================================
# 9. Directory creation
# ===========================================================================
def ensure_dirs():
    """Create every output directory if missing (safe to call repeatedly).
    Only creates dirs under ROOT -- never touches the shared external
    parcel/NLCD stores."""
    for d in (SCRIPTS_DIR, LOGS_DIR, DATA_DIR, MODELS_DIR, OD_MODEL_DIR,
              CWNS_DIR, TRAINING_DIR, NLCD_OUTPUT_DIR, OD_OUTPUT_DIR,
              OD_OUTPUT_DIR_CORRECTED, FEATURES_OUTPUT_DIR, REFERENCE_DIR):
        d.mkdir(parents=True, exist_ok=True)