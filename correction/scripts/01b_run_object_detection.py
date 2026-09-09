"""
01b_run_object_detection_hpc.py
=================================
HPC counterpart to 01b_run_object_detection.py. Same per-plant feature
schema and parcel-polygon in_parcel logic (this is what 02_feature_engineering.py
actually consumes from plants/*.parquet -- od_ran, od_has_detection,
od_n_objects, per-class od_has_{cls}/od_n_{cls}/od_max_conf_{cls}, etc.),
but two things are different for the HPC environment:

  1. Imagery comes from Microsoft Planetary Computer (streamed COG windowed
     reads over HTTPS), NOT local .sid mosaics. This removes the entire
     osgeo.gdal / arcgispro-py3-clone dependency -- this script runs in a
     PLAIN venv, no MrSID support needed at all. See TPQA_PROJECT_STATUS.md
     section 5 for why that was ever a constraint in the first place, and
     the 2026-08-19/20 session notes for the throughput testing that
     justified this rewrite (Planetary Computer, free, ~linear scaling up
     to at least 47 concurrent workers on this cluster's network path).

  2. Plant universe comes from the CWNS text exports in config.py's CWNS_DIR,
     restricted to training-labeled plants via TRAINING_GPKG -- identical to
     the local 01b. Deliberately NOT detection/'s ALL_PLANTS_GPKG.

Outputs (Parquet, partitioned by state, append-only -- same four-table
pattern as both 01b and 07):
    tiles/state=XX/part-*.parquet        one row per tile attempted
    detections/state=XX/part-*.parquet   one row per raw YOLO box, pre-dedup
    objects/state=XX/part-*.parquet      one row per merged real-world object,
                                          carries `in_parcel`
    plants/state=XX/part-*.parquet       one row per plant's OD summary features

ENVIRONMENT: plain venv, Python >= 3.9. Needs a newer python module loaded
on this cluster -- the default system python3 was found to be 3.6 during
testing, which is too old for this script's syntax and pulls badly stale
package wheels. See test_pc_naip_hpc.slurm's `module avail python` /
module-load-with-fallback pattern and copy it here if in doubt.

Requires: ultralytics, torch, duckdb (or geopandas+shapely for parcel
loading -- this version follows 07's geopandas sjoin approach), rasterio,
pystac-client, planetary-computer, pyproj, pandas, numpy, pillow

Usage:
    python 01b_run_object_detection_hpc.py --states OH,PA --workers 32
    python 01b_run_object_detection_hpc.py --states OH --limit 50 --no-resume
"""
import argparse
import math
import random
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import torch
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from shapely import from_wkb, Point
from shapely.geometry import box
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

NODATA_STD_THRESHOLD = 1.0   # near-uniform pixel values = no real imagery (coverage gap, edge padding)

MODEL_IMGSZ = -(-C.IMAGE_PX // 32) * 32   # ceiling to nearest multiple of 32

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "VSI_CACHE": "TRUE",
}
# ^ Without these, GDAL can fall back to slow, unpipelined HTTP range-request
# behavior against blob storage even for a proper COG -- this was the
# difference between ~10s and ~0.5s per tile in testing (2026-08-20 session).

# Thread-local storage: one open rasterio dataset handle + pyproj Transformer
# per (thread, url)/(thread, crs) pair. GDAL dataset handles are NOT
# thread-safe to share across threads, so each worker thread gets its own
# handle -- opening it once and reusing it for every tile that falls in the
# same COG (typical: all of one plant's tiles share one quad) is what makes
# repeat reads ~free (see 2026-08-20 session: 0.00s repeat reads).
_local = threading.local()


def _get_dataset(url: str):
    cache = getattr(_local, "ds_cache", None)
    if cache is None:
        cache = {}
        _local.ds_cache = cache
    ds = cache.get(url)
    if ds is None:
        ds = rasterio.open(url)
        cache[url] = ds
    return ds


def _get_transformer(dst_crs) -> Transformer:
    cache = getattr(_local, "transformer_cache", None)
    if cache is None:
        cache = {}
        _local.transformer_cache = cache
    key = str(dst_crs)
    tr = cache.get(key)
    if tr is None:
        tr = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
        cache[key] = tr
    return tr


# ===========================================================================
# Plant list -- same contract as the local 01b: CWNS text exports restricted
# to FACILITY_TYPE == "Treatment Plant", then (by default) further restricted
# to the plants that actually carry training labels. Deliberately NOT the
# detection-side ALL_PLANTS_GPKG: correction/ and detection/ have different
# plant universes and mixing them silently changes what gets processed.
# ===========================================================================
def load_training_plant_ids() -> set:
    """Union of CWNS_IDs across BOTH label layers -- Stage 1 needs 'classes',
    Stage 2 needs 'corrections'. Restricting to one would silently starve the
    other model of training rows it is supposed to have."""
    if not C.TRAINING_GPKG.exists():
        raise FileNotFoundError(
            f"{C.TRAINING_GPKG} not found. Run build_training_bins.py first "
            f"(00_build_training_bins.slurm), or pass --full-universe."
        )
    classes = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CLASSES)
    corrections = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CORRECTIONS)
    ids = set(classes["CWNS_ID"].astype(str)) | set(corrections["CWNS_ID"].astype(str))
    print(f"  Training plant universe: {len(classes)} classes + {len(corrections)} "
          f"corrections -> {len(ids)} unique CWNS_IDs")
    return ids


def load_treatment_plants(states, training_only: bool = True) -> pd.DataFrame:
    facility_types = pd.read_csv(C.CWNS_DIR / "FACILITY_TYPES.txt", dtype=str, encoding="latin1")
    treatment_ids = set(
        facility_types.loc[facility_types["FACILITY_TYPE"] == "Treatment Plant", "CWNS_ID"]
    )

    loc = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt", dtype={"CWNS_ID": str}, encoding="latin1")
    loc = loc[loc["CWNS_ID"].isin(treatment_ids)]
    loc["LATITUDE"] = pd.to_numeric(loc["LATITUDE"], errors="coerce")
    loc["LONGITUDE"] = pd.to_numeric(loc["LONGITUDE"], errors="coerce")
    n_before = len(loc)
    loc = loc.dropna(subset=["LATITUDE", "LONGITUDE"])
    if len(loc) < n_before:
        print(f"  Dropped {n_before - len(loc)} rows with non-numeric/missing coordinates")
    loc = loc.drop_duplicates(subset="CWNS_ID")
    loc = loc[["CWNS_ID", "STATE_CODE", "LATITUDE", "LONGITUDE"]]

    if training_only:
        training_ids = load_training_plant_ids()
        n_before = len(loc)
        loc = loc[loc["CWNS_ID"].isin(training_ids)]
        print(f"  Restricted to training-labeled plants: {n_before} -> {len(loc)}")

    if states:
        loc = loc[loc["STATE_CODE"].isin(states)]

    return loc.reset_index(drop=True)


def find_containing_parcels(con, state: str, plants_state: pd.DataFrame) -> pd.DataFrame:
    """One DuckDB spatial query per state against the Regrid parquet store --
    same approach as the local 01b and extract_parcels.R. Plants with no
    containing parcel are simply absent from the result; the caller counts
    them separately rather than writing a misleading all-negative row."""
    con.register("pts", plants_state[["CWNS_ID", "LATITUDE", "LONGITUDE"]])
    try:
        return con.execute(f"""
            SELECT pts.CWNS_ID, p.{C.PARCEL_ID_FIELD} AS ll_uuid,
                   p.{C.PARCEL_WKB_FIELD} AS parcel_wkb
            FROM read_parquet('{C.PARCEL_BASE.as_posix()}/state={state}/*.parquet') p
            JOIN pts ON ST_Intersects(
                ST_GeomFromWKB(p.{C.PARCEL_WKB_FIELD}),
                ST_Point(pts.LONGITUDE, pts.LATITUDE)
            )
        """).df()
    except Exception as e:
        print(f"  Parcel lookup failed for state {state}: {e}")
        return pd.DataFrame(columns=["CWNS_ID", "ll_uuid", "parcel_wkb"])
    finally:
        con.unregister("pts")


# ===========================================================================
# Tile geometry (identical to 01b / 07 -- kept in sync manually across all
# three scripts; do not change tile math here without updating the others)
# ===========================================================================
def generate_tile_grid(parcel_geom) -> list[dict]:
    """Tiles built from the parcel's bounding box, then FILTERED to only
    those that actually intersect the parcel's real polygon shape.

    Why the filter matters (added 2026-08-21, after a KY plant surfaced
    this): bounds-only tiling is fine for a compact parcel, where the
    bounding box is close to the parcel's own footprint. For a long, thin,
    DIAGONAL parcel (e.g. a treatment plant strung out along a road/property
    line), the axis-aligned bounding box has to stretch corner-to-corner to
    contain the whole diagonal shape -- creating large empty triangular
    regions that get tiled anyway despite the parcel never actually
    occupying them. One KY plant with a perfectly reasonable 1.24 km2 area
    generated 234 bbox tiles this way, ~29% of which failed for reasons tied
    to the tile falling well outside real coverage. Filtering to actual
    polygon intersection keeps full coverage along the parcel's whole real
    extent while dropping only the tiles that were never really "on the
    parcel" in the first place -- this is NOT the same as capping/shrinking
    the parcel (see MAX_TILES_PER_PLANT in config.py, which now only
    exists as a last-resort safety net for when even the filtered tile
    count is still unreasonable, not the primary mechanism)."""
    bounds = parcel_geom.bounds
    xmin, ymin, xmax, ymax = bounds
    if (xmax - xmin) < C.TILE_SIZE_M or (ymax - ymin) < C.TILE_SIZE_M:
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        return [dict(x_min=cx - C.TILE_SIZE_M / 2, y_min=cy - C.TILE_SIZE_M / 2,
                     x_max=cx + C.TILE_SIZE_M / 2, y_max=cy + C.TILE_SIZE_M / 2,
                     row=1, col=1)]
    x_mins = np.arange(xmin, xmax - C.TILE_SIZE_M + 1e-6, C.STRIDE_M)
    y_mins = np.arange(ymin, ymax - C.TILE_SIZE_M + 1e-6, C.STRIDE_M)
    tiles = []
    for r, y0 in enumerate(y_mins, start=1):
        for c, x0 in enumerate(x_mins, start=1):
            tile_box = box(x0, y0, x0 + C.TILE_SIZE_M, y0 + C.TILE_SIZE_M)
            if tile_box.intersects(parcel_geom):
                tiles.append(dict(x_min=x0, y_min=y0, x_max=x0 + C.TILE_SIZE_M,
                                  y_max=y0 + C.TILE_SIZE_M, row=r, col=c))
    if not tiles:
        # Every bbox tile missed the actual polygon (shouldn't normally
        # happen, but a degenerate/self-intersecting geometry could produce
        # this) -- fall back to one tile centered on the polygon's own
        # centroid rather than returning nothing.
        c = parcel_geom.centroid
        tiles = [dict(x_min=c.x - C.TILE_SIZE_M / 2, y_min=c.y - C.TILE_SIZE_M / 2,
                      x_max=c.x + C.TILE_SIZE_M / 2, y_max=c.y + C.TILE_SIZE_M / 2,
                      row=1, col=1)]
    return tiles


def tiles_to_wgs84(tiles: list[dict]):
    from shapely.geometry import box
    boxes = gpd.GeoSeries(
        [box(t["x_min"], t["y_min"], t["x_max"], t["y_max"]) for t in tiles],
        crs=C.PROJECTED_CRS,
    ).to_crs(C.EXPORT_CRS)
    b = boxes.bounds.to_numpy()
    return [tuple(map(float, row)) for row in b]


# ===========================================================================
# STAC item resolution (once per plant, in the main thread -- cheap, ~0.5-2s
# per search in testing, and avoids N threads hammering the STAC API with
# redundant identical searches for tiles that all fall in one quad anyway)
# ===========================================================================
def with_retry(fn, *args, max_retries=5, base_delay=8, **kwargs):
    """Retry with exponential backoff + jitter, for transient/rate-limit
    errors against Planetary Computer. Per Microsoft's own guidance
    (github.com/microsoft/PlanetaryComputer discussions #246, #77):
    the STAC search endpoint itself isn't hard-rate-limited, but SAS token
    issuance (which happens per-item during search.items() when using the
    sign_inplace modifier, and again on each asset read) IS rate-limited,
    more aggressively for requests without a subscription key and/or not
    originating from PC's own West Europe datacenter -- both true for this
    cluster. Confirmed 2026-08-21: running 52 concurrent 01b array tasks
    (each with its own 32-worker fetch pool) tripped
    "You have exceeded a rate limit" during a STAC item search.

    Does NOT fix the underlying concurrency pressure by itself -- pair with
    a SLURM array throttle (--array=0-51%N) and/or a free PC_SDK_SUBSCRIPTION_KEY
    (https://planetarycomputer.developer.azure-api.net/, auto-detected from
    the environment by the planetary_computer package, no code change
    needed here) to actually raise the ceiling rather than just retrying
    into the same wall more slowly."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            # TileOutsideItemCoverage is NEVER transient/retryable here --
            # retrying against the SAME item will fail identically every
            # time (the tile just isn't in that item's raster). The correct
            # response is timed_fetch's fallback to a freshly-resolved item,
            # not backoff-and-retry, so always re-raise immediately.
            # Checked by TYPE, not message content -- see the string-check
            # bug below for why that distinction matters.
            if isinstance(e, TileOutsideItemCoverage):
                raise
            msg = str(e).lower()
            # Word-boundary regex, not plain substring matching: a coordinate
            # like "11429" contains the substring "429" and would otherwise
            # false-positive-match the HTTP-429 check below. Confirmed
            # 2026-08-21: this exact collision caused a TileOutsideItemCoverage
            # error (message included pixel offset 11429) to be misclassified
            # as a rate-limit error and retried 4 times with exponential
            # backoff (~134s wasted) before ever reaching the real fallback.
            is_transient = (
                "rate limit" in msg or "timeout" in msg or "timed out" in msg
                or re.search(r"\b(429|503|504)\b", msg) is not None
            )
            if not is_transient or attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
            print(f"    Transient error ({e}) -- retrying in {delay:.0f}s "
                  f"(attempt {attempt + 1}/{max_retries})", flush=True)
            time.sleep(delay)
    raise last_exc


def find_naip_item(catalog, lon: float, lat: float):
    def _search():
        search = catalog.search(
            collections=["naip"],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            limit=5,
        )
        return list(search.items())

    items = with_retry(_search)
    if not items:
        return None
    items.sort(key=lambda it: it.datetime, reverse=True)
    return items[0]


# ===========================================================================
# NAIP fetch -- streamed windowed read from a Planetary Computer COG, in
# place of 01b's local .sid read and 07's dead-ImageServer HTTP call. Same
# output contract as both: a (4, IMAGE_PX, IMAGE_PX) uint8 array (R,G,B,NIR)
# covering exactly the given WGS84 bbox.
# ===========================================================================
class TileOutsideItemCoverage(Exception):
    """Raised when a tile's computed window falls outside the resolved NAIP
    item's actual raster extent. Distinguishable from a genuine network/
    rate-limit failure so callers can retry against a freshly-resolved item
    for the tile's own location, instead of just backing off and retrying
    against an item that was never going to cover it."""
    pass


def fetch_naip_tile(bbox_wgs84, item_url: str) -> np.ndarray:
    ds = _get_dataset(item_url)
    transformer = _get_transformer(ds.crs)
    xmin, ymin, xmax, ymax = bbox_wgs84
    left, bottom = transformer.transform(xmin, ymin)
    right, top = transformer.transform(xmax, ymax)
    window = from_bounds(left, bottom, right, top, transform=ds.transform)

    # Validate the window actually falls within the raster's real bounds
    # BEFORE attempting the read. A large/elongated parcel's tiles can span
    # multiple NAIP quads even though only ONE item was resolved for the
    # whole plant (from its single reported point) -- tiles near the far
    # end can genuinely belong to a different quad. Confirmed 2026-08-21 on
    # a KY plant: GDAL's own error for this is a cryptic "Access window out
    # of range in RasterIO()" with the offset landing exactly at the
    # raster's edge. Checking explicitly here (rather than letting that
    # opaque error propagate) is what lets the caller retry against a
    # freshly-resolved item for just this tile, instead of wasting retries
    # against an item that was never going to cover it.
    col_off, row_off = window.col_off, window.row_off
    win_w, win_h = window.width, window.height
    if (col_off < 0 or row_off < 0
            or col_off + win_w > ds.width or row_off + win_h > ds.height):
        raise TileOutsideItemCoverage(
            f"tile window ({col_off:.0f},{row_off:.0f} size {win_w:.0f}x{win_h:.0f}) "
            f"falls outside item raster ({ds.width}x{ds.height}) -- likely a "
            f"different NAIP quad than the one resolved for this plant"
        )

    arr = ds.read([1, 2, 3, 4], window=window,
                  out_shape=(4, C.IMAGE_PX, C.IMAGE_PX),
                  resampling=Resampling.bilinear)
    return arr.astype(np.uint8)


def timed_fetch(bbox_wgs84, item_url: str, catalog=None):
    t0 = time.time()
    try:
        arr = with_retry(fetch_naip_tile, bbox_wgs84, item_url)
    except TileOutsideItemCoverage:
        if catalog is None:
            raise
        # Fall back: resolve a FRESH item for this tile's own centroid,
        # rather than assuming the whole parcel shares one quad. Only
        # triggers for the tile(s) that actually need it -- most tiles for
        # most plants never hit this path, so this doesn't cost anything
        # for the common case.
        xmin, ymin, xmax, ymax = bbox_wgs84
        tile_lon, tile_lat = (xmin + xmax) / 2, (ymin + ymax) / 2
        fresh_item = with_retry(find_naip_item, catalog, tile_lon, tile_lat)
        if fresh_item is None:
            raise
        fresh_url = fresh_item.assets["image"].href
        if fresh_url == item_url:
            # Same item resolved again -- this tile genuinely isn't covered
            # by any item here, not a wrong-quad problem after all.
            raise
        arr = with_retry(fetch_naip_tile, bbox_wgs84, fresh_url)
    return arr, time.time() - t0


def detect_nodata(arr4: np.ndarray) -> bool:
    return bool(arr4.max() == 0) or bool(arr4.std() < NODATA_STD_THRESHOLD)


def pixel_to_lonlat(cx, cy, W, H, bbox):
    xmin, ymin, xmax, ymax = bbox
    return (xmin + (cx / W) * (xmax - xmin), ymax - (cy / H) * (ymax - ymin))


def meters_between(lon1, lat1, lon2, lat2):
    mlat = math.radians((lat1 + lat2) / 2.0)
    dx = (lon2 - lon1) * 111_320.0 * math.cos(mlat)
    dy = (lat2 - lat1) * 111_320.0
    return math.hypot(dx, dy)


# ===========================================================================
# Detections -> objects (geographic NMS), then the in_parcel check
# (identical logic to 01b_run_object_detection.py)
# ===========================================================================
def geographic_nms(raw_dets: list[dict], nms_dist: float) -> list[dict]:
    if not raw_dets:
        return []
    df = pd.DataFrame(raw_dets)
    objects = []
    for cls_id, grp in df.groupby("class_id", sort=False):
        grp = grp.sort_values("confidence", ascending=False).reset_index(drop=True)
        used = np.zeros(len(grp), dtype=bool)
        for i in range(len(grp)):
            if used[i]:
                continue
            seed = grp.iloc[i]
            members = [i]
            used[i] = True
            for j in range(i + 1, len(grp)):
                if not used[j] and meters_between(
                    seed["lon"], seed["lat"], grp.iloc[j]["lon"], grp.iloc[j]["lat"]
                ) <= nms_dist:
                    members.append(j)
                    used[j] = True
            m = grp.iloc[members]
            w = m["confidence"].to_numpy()
            objects.append(dict(
                class_id=int(cls_id), class_name=seed["class_name"],
                lon=float(np.average(m["lon"], weights=w)),
                lat=float(np.average(m["lat"], weights=w)),
                max_confidence=float(m["confidence"].max()),
                mean_confidence=float(m["confidence"].mean()),
                n_merged=int(len(members)),
                member_detection_ids=list(m["detection_id"]),
            ))
    return objects


def flag_in_parcel(objects: list[dict], parcel_geom_4326) -> list[dict]:
    for o in objects:
        o["in_parcel"] = bool(parcel_geom_4326.contains(Point(o["lon"], o["lat"])))
    return objects


def select_plant_features(objects: list[dict], n_tiles_kept: int, n_tiles_success: int) -> dict:
    """Identical contract to 01b_run_object_detection.py's version -- this
    is what 02_feature_engineering.py's OD join expects to find in
    plants/*.parquet. Do not change column names/semantics here without
    updating that script too."""
    in_parcel_objs = [o for o in objects if o["in_parcel"]]

    base = dict(
        od_ran=n_tiles_success > 0,
        od_has_detection=len(in_parcel_objs) > 0,
        od_n_objects=len(in_parcel_objs),
        od_n_classes=len({o["class_name"] for o in in_parcel_objs}),
        od_n_detections_raw=int(sum(o["n_merged"] for o in in_parcel_objs)),
        od_max_confidence=float(max((o["max_confidence"] for o in in_parcel_objs), default=np.nan)),
        od_mean_confidence=float(np.mean([o["max_confidence"] for o in in_parcel_objs])) if in_parcel_objs else np.nan,
        od_dominant_class=None,
        od_frac_tiles_positive=(n_tiles_kept / n_tiles_success) if n_tiles_success else np.nan,
    )

    if in_parcel_objs:
        cls_counts = pd.Series([o["class_name"] for o in in_parcel_objs]).value_counts()
        base["od_dominant_class"] = str(cls_counts.index[0])

    for cls in C.CLASSES:
        cls_objs = [o for o in in_parcel_objs if o["class_name"] == cls]
        base[f"od_has_{cls}"] = len(cls_objs) > 0
        base[f"od_n_{cls}"] = len(cls_objs)
        base[f"od_max_conf_{cls}"] = float(max((o["max_confidence"] for o in cls_objs), default=np.nan))

    return base


# ===========================================================================
# Per-plant orchestration
# ===========================================================================
def prepare_plant(cwns_id, pgeom_5070, orig_lon, orig_lat, catalog):
    """Phase 1 (sequential, main thread): resolve the NAIP STAC item and
    build the tile grid for one plant. Does NOT fetch any imagery -- that
    happens across the WHOLE BATCH at once in main(), not per-plant. See
    the 2026-08-20 note on why: most parcels are smaller than one tile
    (generate_tile_grid's single-tile fallback), so submitting fetches
    per-plant left almost nothing for the thread pool to parallelize --
    16s/tile wall-clock with 32 workers configured, no better than
    sequential. Submitting fetches for many plants together is what
    actually uses the worker pool."""
    parcel_geom_4326 = gpd.GeoSeries([pgeom_5070], crs=C.PROJECTED_CRS).to_crs(C.EXPORT_CRS).iloc[0]
    item = find_naip_item(catalog, orig_lon, orig_lat)
    grid = generate_tile_grid(pgeom_5070)
    bboxes = tiles_to_wgs84(grid)
    return dict(
        cwns_id=cwns_id, orig_lon=orig_lon, orig_lat=orig_lat,
        parcel_geom_4326=parcel_geom_4326, item=item, grid=grid, bboxes=bboxes,
        tile_rows=[], detection_rows=[], kept=0, deleted=0, failed=0,
        fetch_time=0.0, infer_time=0.0,
    )


def finalize_plant(task: dict, state_abbr: str) -> dict:
    """Phase 3 (sequential, main thread, cheap): once all of a plant's
    tiles have been fetched+inferred (phase 2, batched across plants), run
    geographic NMS + in_parcel + the feature summary. Mutates nothing in
    task except reading from it; returns the plant_row dict."""
    objects = geographic_nms(task["detection_rows"], task["_nms_dist"])
    objects = flag_in_parcel(objects, task["parcel_geom_4326"])
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for idx, o in enumerate(objects):
        o.update(object_id=f"{task['cwns_id']}_obj{idx:02d}", CWNS_ID=task["cwns_id"],
                  state=state_abbr, processed_at=now)

    plant_row = select_plant_features(
        objects, n_tiles_kept=task["kept"], n_tiles_success=task["kept"] + task["deleted"])
    plant_row.update(
        CWNS_ID=task["cwns_id"], state=state_abbr,
        orig_lon=task["orig_lon"], orig_lat=task["orig_lat"],
        n_objects_total=len(objects),
        n_objects_in_parcel=sum(o["in_parcel"] for o in objects),
        processed_at=now,
    )
    task["objects"] = objects
    task["plant_row"] = plant_row
    return plant_row


def run_batch(tasks: dict, state_abbr: str, model, class_names, args, executor, catalog):
    """Phase 2: submit every tile-fetch across EVERY plant in this batch to
    the executor at once, so the thread pool actually has enough concurrent
    work to use all `args.workers` -- this is the fix for the 16s/tile,
    effectively-sequential behavior seen when fetches were submitted one
    plant at a time. Inference stays sequential in the main thread (it's
    CPU-bound and single-threaded anyway) but overlaps with other plants'
    in-flight network fetches via the as_completed loop below.

    tasks: dict of cwns_id -> prepare_plant() output (mutated in place with
    results). Plants with item=None (no STAC coverage) are handled here too,
    so the caller doesn't need a separate no-coverage code path.

    catalog is passed through to timed_fetch so it can resolve a fresh NAIP
    item for an individual tile that turns out to fall outside the plant's
    primary resolved item's coverage (see TileOutsideItemCoverage)."""
    futures = {}
    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    for cwns_id, task in tasks.items():
        task["_nms_dist"] = args.nms_dist
        if task["item"] is None:
            for t, bbox in zip(task["grid"], task["bboxes"]):
                tile_id = f"{cwns_id}_r{t['row']:02d}_c{t['col']:02d}"
                task["failed"] += 1
                task["tile_rows"].append(dict(
                    tile_id=tile_id, CWNS_ID=cwns_id, state=state_abbr,
                    tile_row=t["row"], tile_col=t["col"],
                    bbox_xmin=bbox[0], bbox_ymin=bbox[1], bbox_xmax=bbox[2], bbox_ymax=bbox[3],
                    processed_at=now, outcome="fetch_failed", is_nodata=False,
                    n_raw_detections=0, fetch_error="no NAIP STAC item found for this location",
                ))
            continue
        item_url = task["item"].assets["image"].href
        for t, bbox in zip(task["grid"], task["bboxes"]):
            fut = executor.submit(timed_fetch, bbox, item_url, catalog)
            futures[fut] = (cwns_id, t, bbox)

    for fut in as_completed(futures):
        cwns_id, t, bbox = futures[fut]
        task = tasks[cwns_id]
        tile_id = f"{cwns_id}_r{t['row']:02d}_c{t['col']:02d}"
        processed_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        base = dict(
            tile_id=tile_id, CWNS_ID=cwns_id, state=state_abbr,
            tile_row=t["row"], tile_col=t["col"],
            bbox_xmin=bbox[0], bbox_ymin=bbox[1], bbox_xmax=bbox[2], bbox_ymax=bbox[3],
            processed_at=processed_at,
        )

        try:
            arr, fetch_time = fut.result()
            task["fetch_time"] += fetch_time
        except Exception as e:
            task["failed"] += 1
            # rasterio/GDAL often wraps the REAL cause in a generic message
            # ("Read failed. See previous exception for details.") and loses
            # it unless the exception chain is walked explicitly -- str(e)
            # alone hid the actual root cause for KY's 2026-08-21 failures.
            # Walk __cause__/__context__ to get the real underlying error.
            detail = str(e)
            cause = e.__cause__ or e.__context__
            depth = 0
            while cause is not None and depth < 5:
                detail += f" | caused by: {type(cause).__name__}: {cause}"
                cause = cause.__cause__ or cause.__context__
                depth += 1
            task["tile_rows"].append(dict(**base, outcome="fetch_failed", is_nodata=False,
                                           n_raw_detections=0, fetch_error=detail))
            continue

        is_nodata = detect_nodata(arr)
        t0 = time.time()
        result = model.predict(np.transpose(arr[0:3], (1, 2, 0)),
                                conf=args.conf, iou=args.iou,
                                imgsz=MODEL_IMGSZ, device=args.device, verbose=False)[0]
        task["infer_time"] += time.time() - t0
        boxes = result.boxes
        n = 0 if boxes is None else len(boxes)

        if n == 0:
            task["deleted"] += 1
            task["tile_rows"].append(dict(**base, outcome="no_detections", is_nodata=is_nodata,
                                           n_raw_detections=0, fetch_error=None))
        else:
            task["kept"] += 1
            xywh = boxes.xywh.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cids = boxes.cls.cpu().numpy().astype(int)
            for i, ((cx, cy, bw, bh), c, cid) in enumerate(zip(xywh, conf, cids)):
                lon, lat = pixel_to_lonlat(cx, cy, C.IMAGE_PX, C.IMAGE_PX, bbox)
                task["detection_rows"].append(dict(
                    detection_id=f"{tile_id}_d{i:03d}", tile_id=tile_id, CWNS_ID=cwns_id,
                    class_id=int(cid), class_name=class_names.get(int(cid), str(cid)),
                    confidence=float(c), lon=float(lon), lat=float(lat),
                    processed_at=processed_at,
                ))
            task["tile_rows"].append(dict(**base, outcome="kept", is_nodata=is_nodata,
                                           n_raw_detections=n, fetch_error=None))

    # Phase 3: finalize every plant now that all its tiles are done.
    for cwns_id, task in tasks.items():
        finalize_plant(task, state_abbr)


# ===========================================================================
# Parquet writer (append-only, partitioned by state -- identical pattern to
# 01b and 07)
# ===========================================================================
def flush_parquet(rows: list[dict], out_root: Path, table_name: str, state: str):
    if not rows:
        return
    df = pd.DataFrame(rows)
    part_dir = out_root / table_name / f"state={state}"
    part_dir.mkdir(parents=True, exist_ok=True)
    part_path = part_dir / f"part-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}.parquet"
    df.to_parquet(part_path, engine="pyarrow", index=False)


def load_processed_ids(plants_state_dir: Path) -> set:
    """Resume support: union of CWNS_IDs already present across ALL part
    files in this state's plants/ partition, regardless of how many separate
    runs/retries produced them. A plant is skipped on the next run only if
    it shows up here -- so --no-resume is the only way to force a full
    reprocess, and a crashed/killed job can always just be resubmitted."""
    if not plants_state_dir.exists() or not any(plants_state_dir.glob("*.parquet")):
        return set()
    ids = set()
    for f in plants_state_dir.glob("*.parquet"):
        try:
            ids |= set(pd.read_parquet(f, columns=["CWNS_ID"])["CWNS_ID"].astype(str))
        except Exception:
            pass
    return ids


def resolve_weights(explicit: str | None, models_dir: Path) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        print(f"  NOTE: --weights {p} not found, falling back to newest .pt in {models_dir}")
    candidates = sorted(models_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No .pt weights found in {models_dir}. Upload a trained model first "
            f"(models/object_detection/best.pt from the local training machine)."
        )
    chosen = candidates[0]
    print(f"  Using weights: {chosen.name}"
          + (f" (newest of {len(candidates)} found)" if len(candidates) > 1 else ""))
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=str, default=None,
                     help="comma-separated 2-letter STATE_CODE values, e.g. OH,PA. "
                          "Omit to process every state in the plant universe.")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N plants (testing)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--models-dir", default=str(C.OD_MODEL_DIR),
                     help="where to look for detection weights. Defaults to "
                          "models/object_detection/ -- deliberately NOT models/, "
                          "which is where 03/04 write the stage1/stage2 models.")
    ap.add_argument("--conf", type=float, default=C.CONF_THRESHOLD)
    ap.add_argument("--iou", type=float, default=C.IOU_THRESHOLD)
    ap.add_argument("--nms-dist", type=float, default=C.NMS_DISTANCE_M)
    ap.add_argument("--workers", type=int, default=C.NAIP_WORKERS,
                     help="concurrent NAIP fetch threads. Tested up to 47 with roughly "
                          "linear scaling on this cluster's network path (2026-08-20 "
                          "session) -- if you bump this much higher, watch the failure "
                          "count in the summary for signs of rate-limiting.")
    ap.add_argument("--device", type=str, default="cpu",
                     help="'cpu' (default) or 'cuda' -- inference runs sequentially in "
                          "the main thread regardless; only the NAIP fetch is threaded.")
    ap.add_argument("--full-universe", action="store_true",
                     help="process ALL treatment plants, not just those carrying "
                          "training labels. Only meaningful once a trained Stage 1 "
                          "model exists to decide who needs correcting at inference "
                          "time -- not for training/development runs.")
    ap.add_argument("--no-resume", action="store_true",
                     help="reprocess plants even if already present in the plants/ "
                          "Parquet table for that state. Default (resume ON) skips any "
                          "CWNS_ID already written by a prior run/attempt.")
    args = ap.parse_args()

    C.ensure_dirs()
    states = [s.strip().upper() for s in args.states.split(",")] if args.states else None
    out_root = C.OD_OUTPUT_DIR

    print("=== 01b_run_object_detection_hpc.py ===")
    print(f"Device: {args.device}  |  fetch workers: {args.workers}  |  "
          f"tile={C.TILE_SIZE_M}m imgsz={MODEL_IMGSZ}px  |  "
          f"imagery: Planetary Computer NAIP (streamed, no local download)\n")

    weights_path = resolve_weights(args.weights, Path(args.models_dir))
    model = YOLO(str(weights_path))
    class_names = {int(k): v for k, v in model.names.items()}
    print(f"Model: {weights_path.name}  |  Classes: {class_names}\n")

    print("Connecting to Planetary Computer STAC API...")
    catalog = pystac_client.Client.open(C.STAC_URL, modifier=planetary_computer.sign_inplace)
    print("Connected.\n")

    plants = load_treatment_plants(states, training_only=not args.full_universe)
    print(f"Treatment plants loaded: {len(plants)}")

    if not args.no_resume:
        # Resume is checked PER STATE below (each state's plants/ partition is
        # independent), but pre-filter here too for an accurate up-front count.
        all_done = set()
        for state_dir in (out_root / "plants").glob("state=*"):
            all_done |= load_processed_ids(state_dir)
        if all_done:
            before = len(plants)
            plants = plants[~plants["CWNS_ID"].isin(all_done)]
            print(f"Resuming: skipping {before - len(plants)} already-processed plants "
                  f"(found across all state partitions)")

    if args.limit:
        plants = plants.head(args.limit)
        print(f"--limit active: processing {len(plants)} plants")

    if len(plants) == 0:
        print("Nothing to do.")
        return

    totals = dict(plants=0, kept=0, deleted=0, failed=0, no_parcel=0,
                  parcels_capped=0, fetch_time=0.0, infer_time=0.0)
    t_start = time.time()

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; SET enable_geoparquet_conversion = false;")

    with rasterio.Env(**GDAL_ENV):
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for state_abbr, state_plants in plants.groupby("STATE_CODE"):
                print(f"\n--- State {state_abbr}: {len(state_plants)} plants ---")

                plants_state_dir = out_root / "plants" / f"state={state_abbr}"
                state_plants_todo = state_plants
                if not args.no_resume:
                    done_ids = load_processed_ids(plants_state_dir)
                    if done_ids:
                        before = len(state_plants_todo)
                        state_plants_todo = state_plants_todo[
                            ~state_plants_todo["CWNS_ID"].isin(done_ids)]
                        print(f"  Resuming: skipping {before - len(state_plants_todo)} "
                              f"already-processed plants in this state")

                if len(state_plants_todo) == 0:
                    print("  Nothing to do for this state.")
                    continue

                parcel_matches = find_containing_parcels(con, state_abbr, state_plants_todo)
                matched_ids = set(parcel_matches["CWNS_ID"])
                unmatched = len(state_plants_todo) - len(matched_ids)
                if unmatched:
                    print(f"  No containing parcel found for {unmatched} plants -- skipped, "
                          f"not written to any table (rerun later if parcel coverage improves)")
                    totals["no_parcel"] += unmatched

                # A reported point exactly on a shared parcel boundary can match
                # more than one parcel under ST_Intersects (boundary-inclusive).
                # Keep one deterministic match per CWNS_ID so a plant is never
                # processed twice -- sorted by ll_uuid purely for reproducibility.
                n_before = len(parcel_matches)
                parcel_matches = parcel_matches.sort_values("ll_uuid").drop_duplicates(
                    subset="CWNS_ID", keep="first")
                if len(parcel_matches) < n_before:
                    print(f"  {n_before - len(parcel_matches)} plants matched >1 parcel "
                          f"(boundary edge case) -- kept one match each")

                merged = state_plants_todo.merge(parcel_matches, on="CWNS_ID", how="inner")

                tiles_buf, dets_buf, objs_buf, plants_buf = [], [], [], []
                FLUSH_EVERY = 200   # bounds memory AND is the concurrency batch
                                     # size (see run_batch()) -- all of one
                                     # batch's tile fetches are submitted to the
                                     # executor together, then finalized+flushed
                                     # as one resume checkpoint

                def _flush_all():
                    flush_parquet(tiles_buf, out_root, "tiles", state_abbr)
                    flush_parquet(dets_buf, out_root, "detections", state_abbr)
                    flush_parquet(objs_buf, out_root, "objects", state_abbr)
                    flush_parquet(plants_buf, out_root, "plants", state_abbr)

                # Process in batches of FLUSH_EVERY plants: resolve each
                # plant's parcel/STAC item first (cheap, sequential), then
                # submit the WHOLE batch's tile fetches to the executor
                # together via run_batch(). This is what actually uses
                # --workers -- submitting fetches one plant at a time left
                # nothing to parallelize for the ~90% of parcels smaller
                # than one tile (16s/tile wall-clock with 32 workers
                # configured, confirmed 2026-08-20 -- no better than
                # sequential). Batch size doubles as the flush/resume
                # checkpoint granularity, same as before.
                merged_rows = list(merged.iterrows())
                for batch_start in range(0, len(merged_rows), FLUSH_EVERY):
                    batch = merged_rows[batch_start:batch_start + FLUSH_EVERY]
                    tasks = {}
                    area_by_id = {}

                    for _, row in batch:
                        parcel_geom_4326 = from_wkb(bytes(row["parcel_wkb"]))
                        pgeom_5070 = gpd.GeoSeries([parcel_geom_4326], crs=C.EXPORT_CRS) \
                            .to_crs(C.PROJECTED_CRS).iloc[0]

                        parcel_area_m2 = pgeom_5070.area
                        bxmin, bymin, bxmax, bymax = pgeom_5070.bounds
                        bbox_max_dim_m = max(bxmax - bxmin, bymax - bymin)
                        area_exceeded = parcel_area_m2 > C.MAX_PARCEL_AREA_M2
                        parcel_capped = area_exceeded
                        cap_reason = None

                        if area_exceeded:
                            # Whole-town digitization artifact -- not worth
                            # tiling at all, even with real-polygon filtering
                            # (the actual area itself is the problem here, not
                            # the shape). Substitute a small square centered on
                            # the plant's REPORTED point.
                            cap_reason = "area"
                            totals["parcels_capped"] += 1
                            pt_5070 = gpd.GeoSeries(
                                [Point(float(row["LONGITUDE"]), float(row["LATITUDE"]))],
                                crs=C.EXPORT_CRS
                            ).to_crs(C.PROJECTED_CRS).iloc[0]
                            hw = C.CAPPED_PARCEL_FALLBACK_HALFWIDTH_M
                            pgeom_5070 = box(pt_5070.x - hw, pt_5070.y - hw,
                                              pt_5070.x + hw, pt_5070.y + hw)

                        cwns_id = row["CWNS_ID"]
                        try:
                            tasks[cwns_id] = prepare_plant(
                                cwns_id, pgeom_5070,
                                float(row["LONGITUDE"]), float(row["LATITUDE"]), catalog)
                        except Exception as e:
                            # with_retry already exhausted its backoff budget for
                            # this plant's STAC search -- rather than let one
                            # stubborn failure (e.g. sustained rate-limiting) crash
                            # the whole batch and lose every other already-prepared
                            # plant, skip this one and keep going. It'll be picked
                            # up on the next resumed run since nothing was written
                            # for it.
                            print(f"    WARNING: prepare_plant failed for {cwns_id} "
                                  f"after retries ({e}) -- skipping, will retry on next run")
                            totals["failed"] += 1
                            continue

                        # Last-resort safety net: even with tiles filtered to
                        # actual polygon intersection (see generate_tile_grid),
                        # a pathological shape could still produce an
                        # unreasonable tile count. Genuine plants, however
                        # oddly shaped (e.g. the long diagonal KY case that
                        # motivated this whole rework), should essentially
                        # never hit this -- it exists for shapes we haven't
                        # seen yet, not as the normal path.
                        if not area_exceeded and len(tasks[cwns_id]["grid"]) > C.MAX_TILES_PER_PLANT:
                            cap_reason = "tile_count"
                            parcel_capped = True
                            totals["parcels_capped"] += 1
                            print(f"    {cwns_id}: {len(tasks[cwns_id]['grid'])} tiles even "
                                  f"after real-polygon filtering (area={parcel_area_m2/1e6:.2f} km2, "
                                  f"bbox={bbox_max_dim_m/1e3:.1f} km) -- exceeds "
                                  f"MAX_TILES_PER_PLANT={C.MAX_TILES_PER_PLANT}, falling back "
                                  f"to point footprint")
                            pt_5070 = gpd.GeoSeries(
                                [Point(float(row["LONGITUDE"]), float(row["LATITUDE"]))],
                                crs=C.EXPORT_CRS
                            ).to_crs(C.PROJECTED_CRS).iloc[0]
                            hw = C.CAPPED_PARCEL_FALLBACK_HALFWIDTH_M
                            fallback_geom = box(pt_5070.x - hw, pt_5070.y - hw,
                                                 pt_5070.x + hw, pt_5070.y + hw)
                            tasks[cwns_id] = prepare_plant(
                                cwns_id, fallback_geom,
                                float(row["LONGITUDE"]), float(row["LATITUDE"]), catalog)

                        area_by_id[cwns_id] = (float(parcel_area_m2), parcel_capped,
                                                float(bbox_max_dim_m), cap_reason,
                                                len(tasks[cwns_id]["grid"]))

                    run_batch(tasks, state_abbr, model, class_names, args, executor, catalog)

                    for cwns_id, task in tasks.items():
                        parcel_area_m2, parcel_capped, bbox_max_dim_m, cap_reason, n_tiles_generated = \
                            area_by_id[cwns_id]
                        task["plant_row"]["parcel_area_m2"] = parcel_area_m2
                        task["plant_row"]["parcel_bbox_max_dim_m"] = bbox_max_dim_m
                        task["plant_row"]["parcel_capped"] = parcel_capped
                        task["plant_row"]["parcel_cap_reason"] = cap_reason
                        task["plant_row"]["n_tiles_generated"] = n_tiles_generated

                        tiles_buf.extend(task["tile_rows"])
                        dets_buf.extend(task["detection_rows"])
                        objs_buf.extend(task["objects"])
                        plants_buf.append(task["plant_row"])

                        totals["plants"] += 1
                        totals["kept"] += task["kept"]
                        totals["deleted"] += task["deleted"]
                        totals["failed"] += task["failed"]
                        totals["fetch_time"] += task["fetch_time"]
                        totals["infer_time"] += task["infer_time"]

                    elapsed = time.time() - t_start
                    print(f"    ...{totals['plants']} plants done "
                          f"({elapsed/60:.1f} min elapsed, batch of {len(batch)})")

                    _flush_all()
                    print(f"    flushed {len(plants_buf)} plants (resume point advanced)")
                    tiles_buf, dets_buf, objs_buf, plants_buf = [], [], [], []

                print(f"  State {state_abbr} done")

    con.close()
    elapsed = time.time() - t_start
    n_ok = totals["kept"] + totals["deleted"]
    print("\n=== Summary ===")
    print(f"  Plants processed : {totals['plants']}")
    print(f"  No parcel found  : {totals['no_parcel']}")
    print(f"  Parcels capped (area > {C.MAX_PARCEL_AREA_M2/1e6:.1f} km2, OR "
          f"still >{C.MAX_TILES_PER_PLANT} tiles after real-polygon-filtered "
          f"tiling -- used point-fallback footprint instead): {totals['parcels_capped']}")
    print(f"  Tiles w/ detections   : {totals['kept']}")
    print(f"  Tiles empty           : {totals['deleted']}")
    print(f"  Fetch failures        : {totals['failed']}"
          + ("  <-- check for rate-limiting if this is high relative to plants processed"
             if totals["plants"] and totals["failed"] / max(totals["plants"], 1) > 0.05 else ""))
    print(f"  Elapsed               : {elapsed/60:.1f} min")
    if n_ok:
        print(f"  Mean fetch time  : {totals['fetch_time']/n_ok:.3f}s/tile "
              f"(network, concurrent across {args.workers} workers)")
        print(f"  Mean infer time  : {totals['infer_time']/n_ok:.3f}s/tile "
              f"(device={args.device}, sequential in main thread)")
    print(f"\nOutputs (Parquet, partitioned by state) in: {out_root}")


if __name__ == "__main__":
    main()