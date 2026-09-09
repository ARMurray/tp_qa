"""
convert_tiles_to_500m.py
==========================
Converts already-labeled 200m NAIP tiles into consolidated 500m tiles,
remapping every bounding box from old pixel-space -> real-world WGS84 ->
new pixel-space, and de-duplicating boxes that were labeled redundantly
across multiple overlapping old tiles for the same real object.

WHY: annotating at 200m with 0.33 overlap produces many tiles per parcel for
any site larger than ~130m across -- 3x3 grids are common, all showing the
same facility from slightly different crops. Collapsing to one 500m tile per
(CWNS_ID, ll_uuid_primary) group means one image to review per site instead
of many, while keeping every existing label. Matches the correction/
pipeline's own tile size (TILE_SIZE_M=500, IMAGE_PX=833) at the SAME 0.6m/px
resolution -- see TPQA_MASTER_REFERENCE.md's CRS contract table -- so this
was purely a bigger crop at the same pixel scale, no resampling needed.

GEOMETRY: every old tile's real-world bounding box comes directly from
tile_metadata.csv's bbox_xmin/ymin/xmax/ymax (WGS84) -- the exact same
values 02_extract_tiles.py's save_ndwi_tif() already treats as defining a
simple linear affine transform via rasterio.transform.from_bounds(). This
script uses that identical linear-WGS84-bbox convention for the remap, not a
"more correct" geodesic approach, specifically to stay consistent with how
the rest of this codebase already treats tile georeferencing.

IMAGERY SOURCE (changed 2026-08-28): Planetary Computer, not USDA's ArcGIS
NAIP service (which this script originally used, matching
02_extract_tiles.py -- that USDA endpoint went dead). Ported directly from
01b_run_object_detection.py's fetch machinery -- STAC item search, SAS-token
signing, windowed COG reads, retry-with-backoff, and cross-quad fallback --
rather than reimplemented, since that code has several dated, hard-won bug
fixes (see with_retry()'s docstring) that are easy to silently lose in a
rewrite. A 500m tile is LARGER than 01b's own correction/-scale tiles, so
the cross-quad fallback matters more here, not less.

OUTPUT LOCATION (2026-08-28): writes into NEW sibling folders alongside the
existing ones -- rgb_500/ and ndwi_500/ next to rgb/ and ndwi/ under
data/tiles/, labels_500/ next to labels/ under annotation/ls_export/, and a
separate tile_metadata_500.csv next to tile_metadata.csv. Flattened (no
extra png/ subfolder under rgb_500/, unlike the original rgb/png/ layout).
Kept separate from the live files rather than merged in automatically, so
you can review before folding into what 03_prepare_dataset.py actually
trains on. See "FOLDING IN" below.

Usage:
    python convert_tiles_to_500m.py                    # everything labeled
    python convert_tiles_to_500m.py --limit 5           # test on 5 groups first
    python convert_tiles_to_500m.py --cwns-ids-file ids.txt   # scoped subset
    python convert_tiles_to_500m.py --source-root "..."       # override input location

FOLDING IN, once you've reviewed the output (e.g. by pointing label_app.R's
IMAGE_DIR/OUTPUT_ROOT at the _500 folders for a quick pass):
    - Copy data/tiles/rgb_500/*.png   -> data/tiles/rgb/png/  (or wherever
      03_prepare_dataset.py should read from)
    - Copy data/tiles/ndwi_500/*.tif  -> data/tiles/ndwi/
    - Copy annotation/ls_export/labels_500/*.txt -> annotation/ls_export/labels/
    - Append tile_metadata_500.csv's rows to tile_metadata.csv
    - Decide whether to DELETE the superseded old 200m tiles/labels/metadata
      rows for the same (CWNS_ID, ll_uuid_primary) groups, or leave them --
      not automated here since mixing 200m and 500m tiles of the same real
      object into one training run is a modeling decision, not a data-
      hygiene one. Not made for you.
"""
import argparse
import re
import random
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import rasterio.transform
from pyproj import Transformer
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

NEW_TILE_SIZE_M = 500
NEW_IMAGE_PX = round(NEW_TILE_SIZE_M / C.TARGET_RES_M)  # 833, matches correction/'s tiles

# NOT in detection/pipeline/config.py -- that config was built for the USDA
# ArcGIS NAIP service, which went dead 2026-08-28. Same STAC endpoint the
# HPC correction/ pipeline already uses (its own config.py, a different
# file). Hardcoded here rather than imported since detection/'s config has
# no equivalent constant yet -- worth adding there if more of detection/
# pipeline moves off the USDA service.
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Output folders, per explicit naming from 2026-08-28: siblings of the
# existing rgb/ and ndwi/ folders (not a separate parallel tree), and
# flattened -- no extra png/ subfolder under rgb_500/, unlike the original
# rgb/png/ layout. Labels and metadata follow the same _500 suffix
# convention for consistency, though those two weren't explicitly named --
# say if you want different names for those two specifically.
OUT_RGB_DIR = C.TILES_DIR / "rgb_500"
OUT_NDWI_DIR = C.TILES_DIR / "ndwi_500"
OUT_METADATA_CSV = C.DATA_DIR / "tile_metadata_500.csv"
OUT_LABEL_DIR = C.ANNOTATION_DIR / "labels_500"

IOU_DEDUP_THRESHOLD = 0.5   # same-class boxes above this IoU after remapping
                            # are treated as duplicate labels of one real object


# ===========================================================================
# Planetary Computer STAC fetch -- ported from 01b_run_object_detection.py.
# USDA's ArcGIS NAIP service (this script's original imagery source) went
# dead 2026-08-28. Same STAC endpoint, same retry/quad-boundary-fallback
# logic the HPC correction/ pipeline already relies on -- kept as close to
# verbatim as this script's single-threaded context allows, rather than
# reimplemented, since 01b's version has several dated, hard-won bug fixes
# (see with_retry's docstring) that are easy to silently lose in a rewrite.
# ===========================================================================
_dataset_cache = {}
_transformer_cache = {}


def _get_dataset(url: str):
    ds = _dataset_cache.get(url)
    if ds is None:
        ds = rasterio.open(url)
        _dataset_cache[url] = ds
    return ds


def _get_transformer(dst_crs) -> Transformer:
    key = str(dst_crs)
    tr = _transformer_cache.get(key)
    if tr is None:
        tr = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
        _transformer_cache[key] = tr
    return tr


class TileOutsideItemCoverage(Exception):
    """Raised when a tile's computed window falls outside the resolved NAIP
    item's actual raster extent -- distinguishable from a genuine network/
    rate-limit failure so callers can retry against a freshly-resolved item
    instead of just backing off against an item that was never going to
    cover it."""
    pass


def with_retry(fn, *args, max_retries=5, base_delay=8, **kwargs):
    """Retry with exponential backoff + jitter, for transient/rate-limit
    errors against Planetary Computer. Ported verbatim from
    01b_run_object_detection.py, including two dated fixes worth keeping:
    checking TileOutsideItemCoverage by TYPE (never retryable -- retrying
    against the same item fails identically every time) rather than by
    message content, and a word-boundary regex for the 429/503/504 check
    (a plain substring match on "429" false-positived on a pixel offset
    like "11429" in a real error message, confirmed 2026-08-21)."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if isinstance(e, TileOutsideItemCoverage):
                raise
            msg = str(e).lower()
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


def timed_fetch(bbox_wgs84, item_url: str, image_px: int, catalog=None):
    try:
        arr = with_retry(fetch_naip_tile, bbox_wgs84, item_url, image_px)
    except TileOutsideItemCoverage:
        if catalog is None:
            raise
        # Fall back: resolve a FRESH item for this tile's own centroid,
        # rather than assuming the whole group shares one quad.
        xmin, ymin, xmax, ymax = bbox_wgs84
        tile_lon, tile_lat = (xmin + xmax) / 2, (ymin + ymax) / 2
        fresh_item = with_retry(find_naip_item, catalog, tile_lon, tile_lat)
        if fresh_item is None:
            raise
        fresh_url = fresh_item.assets["image"].href
        if fresh_url == item_url:
            raise  # genuinely not covered by any item, not a wrong-quad problem
        arr = with_retry(fetch_naip_tile, bbox_wgs84, fresh_url, image_px)
    return arr


# ===========================================================================
# Load existing labeled tiles
# ===========================================================================
def load_labeled_tiles(source_metadata_csv: Path, source_annotation_dir: Path,
                       cwns_filter: set | None) -> pd.DataFrame:
    meta = pd.read_csv(source_metadata_csv, dtype={"CWNS_ID": str, "ll_uuid_primary": str})
    if cwns_filter:
        meta = meta[meta["CWNS_ID"].isin(cwns_filter)]

    rows = []
    for _, row in meta.iterrows():
        label_path = source_annotation_dir / "labels" / f"{row['tile_id']}_rgb.txt"
        if not label_path.exists():
            continue  # unreviewed -- not guessed at, just skipped
        rows.append({**row.to_dict(), "label_path": label_path})
    df = pd.DataFrame(rows)
    print(f"Loaded {len(meta)} total tile(s) in metadata, {len(df)} with a "
          f"label file (reviewed, positive or confirmed-negative)")
    return df


def read_yolo_boxes(label_path: Path) -> list[tuple]:
    """[(class_id, cx, cy, w, h), ...] normalized 0-1. Empty list for a
    confirmed-negative (zero-byte) file -- that's real signal, not missing
    data, per label_app.R's own convention."""
    if label_path.stat().st_size == 0:
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        boxes.append((int(parts[0]), float(parts[1]), float(parts[2]),
                     float(parts[3]), float(parts[4])))
    return boxes


# ===========================================================================
# Geometry: old tile pixel-space -> real world -> new tile pixel-space
# ===========================================================================
def remap_normalized_box(box_norm: tuple, old_bbox_wgs84: tuple, old_image_px: int,
                         new_bbox_wgs84: tuple, new_image_px: int) -> tuple | None:
    """box_norm = (class_id, cx, cy, w, h), normalized to OLD tile.
    Returns the same box normalized to the NEW tile, or None if it falls
    entirely outside the new tile's extent after remapping (clipped to
    nothing) -- e.g. a group whose old tiles spread wider than the new
    tile's 500m footprint.

    Linear affine, matching rasterio.transform.from_bounds(west, south,
    east, north, width, height)'s own convention exactly: pixel row 0 is
    the NORTH edge, row=height is the SOUTH edge; col 0 is WEST, col=width
    is EAST. YOLO's (cx, cy) already uses this same top-left-origin,
    y-increases-downward convention, so no extra flip is needed beyond
    what's written out below explicitly for clarity."""
    class_id, cx, cy, w, h = box_norm
    old_w, old_s, old_e, old_n = old_bbox_wgs84
    new_w, new_s, new_e, new_n = new_bbox_wgs84

    # old normalized -> old pixel corners
    xmin_px, xmax_px = (cx - w / 2) * old_image_px, (cx + w / 2) * old_image_px
    ymin_px, ymax_px = (cy - h / 2) * old_image_px, (cy + h / 2) * old_image_px

    # old pixel -> real world WGS84 (linear, matching from_bounds' own convention)
    lon_min = old_w + (xmin_px / old_image_px) * (old_e - old_w)
    lon_max = old_w + (xmax_px / old_image_px) * (old_e - old_w)
    lat_max = old_n - (ymin_px / old_image_px) * (old_n - old_s)  # smaller row = further north
    lat_min = old_n - (ymax_px / old_image_px) * (old_n - old_s)

    # real world -> new tile pixel space
    if new_e == new_w or new_n == new_s:
        return None  # degenerate new bbox -- shouldn't happen, guard anyway
    new_xmin = ((lon_min - new_w) / (new_e - new_w)) * new_image_px
    new_xmax = ((lon_max - new_w) / (new_e - new_w)) * new_image_px
    new_ymin = ((new_n - lat_max) / (new_n - new_s)) * new_image_px
    new_ymax = ((new_n - lat_min) / (new_n - new_s)) * new_image_px

    # clip to the new tile's actual extent
    new_xmin, new_xmax = max(0, new_xmin), min(new_image_px, new_xmax)
    new_ymin, new_ymax = max(0, new_ymin), min(new_image_px, new_ymax)
    if new_xmax - new_xmin < 2 or new_ymax - new_ymin < 2:
        return None  # collapsed to nothing (or a sliver) -- outside the new tile

    new_cx = (new_xmin + new_xmax) / 2 / new_image_px
    new_cy = (new_ymin + new_ymax) / 2 / new_image_px
    new_w = (new_xmax - new_xmin) / new_image_px
    new_h = (new_ymax - new_ymin) / new_image_px
    return (class_id, new_cx, new_cy, new_w, new_h)


def box_iou(a: tuple, b: tuple) -> float:
    """IoU of two normalized (cx, cy, w, h) boxes."""
    ax1, ax2 = a[0] - a[2] / 2, a[0] + a[2] / 2
    ay1, ay2 = a[1] - a[3] / 2, a[1] + a[3] / 2
    bx1, bx2 = b[0] - b[2] / 2, b[0] + b[2] / 2
    by1, by2 = b[1] - b[3] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def dedup_boxes(boxes: list[tuple]) -> tuple[list, int]:
    """Greedy per-class IoU merge. Returns (deduped_boxes, n_dropped).
    Keeps the first-seen box in each duplicate cluster -- there's no
    per-box confidence to prefer one over another, and averaging duplicate
    coordinates would blur an otherwise-precise original annotation."""
    kept = []
    n_dropped = 0
    for b in boxes:
        class_id = b[0]
        is_dup = False
        for k in kept:
            if k[0] == class_id and box_iou(b[1:], k[1:]) > IOU_DEDUP_THRESHOLD:
                is_dup = True
                break
        if is_dup:
            n_dropped += 1
        else:
            kept.append(b)
    return kept, n_dropped


# ===========================================================================
# New tile center / bbox, and NAIP fetch (mirrors 02_extract_tiles.py)
# ===========================================================================
def compute_new_tile_bbox(group: pd.DataFrame) -> tuple:
    """Averages the group's old tile centers in PROJECTED_CRS (meters) --
    not naive lat/lon degree-averaging -- then builds a NEW_TILE_SIZE_M
    square around that point, reprojected to WGS84 the same way
    02_extract_tiles.py's tiles_to_wgs84() does. Returns
    (new_bbox_wgs84, new_ctr_lon, new_ctr_lat, spread_m)."""
    centers = gpd.GeoSeries(
        [Point(lon, lat) for lon, lat in zip(group["ctr_lon"], group["ctr_lat"])],
        crs=C.EXPORT_CRS,
    ).to_crs(C.PROJECTED_CRS)

    mean_x, mean_y = centers.x.mean(), centers.y.mean()
    spread_m = max(
        centers.x.max() - centers.x.min(), centers.y.max() - centers.y.min()
    ) if len(centers) > 1 else 0.0

    half = NEW_TILE_SIZE_M / 2
    new_box_projected = gpd.GeoSeries(
        [box(mean_x - half, mean_y - half, mean_x + half, mean_y + half)],
        crs=C.PROJECTED_CRS,
    ).to_crs(C.EXPORT_CRS)
    bounds = new_box_projected.bounds.iloc[0]
    new_ctr = gpd.GeoSeries([Point(mean_x, mean_y)], crs=C.PROJECTED_CRS).to_crs(C.EXPORT_CRS).iloc[0]

    return (
        (float(bounds.minx), float(bounds.miny), float(bounds.maxx), float(bounds.maxy)),
        float(new_ctr.x), float(new_ctr.y), spread_m,
    )


def fetch_naip_tile(bbox_wgs84: tuple, item_url: str, image_px: int) -> np.ndarray:
    """Ported from 01b_run_object_detection.py, adapted for this script's
    single-threaded/per-group use rather than 01b's multi-threaded pool
    (plain dict caches here instead of threading.local -- no benefit to
    thread-local caching in a sequential loop). Same output contract:
    (4, image_px, image_px) uint8 array (R,G,B,NIR)."""
    ds = _get_dataset(item_url)
    transformer = _get_transformer(ds.crs)
    xmin, ymin, xmax, ymax = bbox_wgs84
    left, bottom = transformer.transform(xmin, ymin)
    right, top = transformer.transform(xmax, ymax)
    window = from_bounds(left, bottom, right, top, transform=ds.transform)

    # Validate the window falls within the raster's real bounds BEFORE
    # reading -- a 500m tile is LARGER than 01b's original correction/-scale
    # tiles, so it's MORE likely (not less) to straddle two NAIP quads even
    # though only one item was resolved from the tile's center point.
    col_off, row_off = window.col_off, window.row_off
    win_w, win_h = window.width, window.height
    if (col_off < 0 or row_off < 0
            or col_off + win_w > ds.width or row_off + win_h > ds.height):
        raise TileOutsideItemCoverage(
            f"tile window ({col_off:.0f},{row_off:.0f} size {win_w:.0f}x{win_h:.0f}) "
            f"falls outside item raster ({ds.width}x{ds.height}) -- likely a "
            f"different NAIP quad than the one resolved for this tile")

    arr = ds.read([1, 2, 3, 4], window=window,
                  out_shape=(4, image_px, image_px),
                  resampling=Resampling.bilinear)
    return arr.astype(np.uint8)


def save_rgb_png(arr4: np.ndarray, path: Path):
    Image.fromarray(np.transpose(arr4[0:3], (1, 2, 0)), mode="RGB").save(path)


def save_ndwi_tif(arr4: np.ndarray, bbox_wgs84: tuple, path: Path):
    green, nir = arr4[1].astype(np.float32), arr4[3].astype(np.float32)
    denom = green + nir
    ndwi = np.where(denom == 0, 0.0, (green - nir) / denom).astype(np.float32)
    h, w = ndwi.shape
    transform = rasterio.transform.from_bounds(*bbox_wgs84, w, h)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="float32", crs=f"EPSG:{C.EXPORT_CRS}",
                       transform=transform) as dst:
        dst.write(ndwi, 1)


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=str, default=None,
                     help="override where existing labeled 200m tiles are read "
                          "from (expects <root>/data/tile_metadata.csv and "
                          "<root>/annotation/ls_export/labels/). Defaults to "
                          "config.py's own DATA_DIR/ANNOTATION_DIR -- now that "
                          "REPO_ROOT resolves correctly (fixed 2026-08-28), this "
                          "shouldn't be needed for normal use, only if you're "
                          "pointing at a different tree deliberately.")
    ap.add_argument("--limit", type=int, default=None,
                     help="process only the first N groups -- test before running everything")
    ap.add_argument("--cwns-ids-file", type=str, default=None,
                     help="text file, one CWNS_ID per line -- scope to just these plants")
    args = ap.parse_args()

    print("=== convert_tiles_to_500m.py ===")
    print("Opening Planetary Computer STAC catalog...")
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)

    if args.source_root:
        source_root = Path(args.source_root)
        source_metadata_csv = source_root / "data" / "tile_metadata.csv"
        source_annotation_dir = source_root / "annotation" / "ls_export"
    else:
        source_metadata_csv = C.METADATA_CSV
        source_annotation_dir = C.ANNOTATION_DIR

    if not source_metadata_csv.exists():
        raise FileNotFoundError(
            f"{source_metadata_csv} not found. Pass --source-root to point at a "
            f"different tree, or check config.py's DATA_DIR is resolving where "
            f"you expect (REPO_ROOT should be detection/, not detection/pipeline/).")

    print(f"Reading existing tiles from: {source_metadata_csv.parent}")
    print(f"New tile size: {NEW_TILE_SIZE_M}m -> {NEW_IMAGE_PX}px "
          f"(at {C.TARGET_RES_M}m/px, same resolution as the old {C.TILE_SIZE_M}m tiles)")

    cwns_filter = None
    if args.cwns_ids_file:
        cwns_filter = set(Path(args.cwns_ids_file).read_text().split())
        print(f"Scoped to {len(cwns_filter)} CWNS_ID(s) from {args.cwns_ids_file}")

    tiles = load_labeled_tiles(source_metadata_csv, source_annotation_dir, cwns_filter)
    if tiles.empty:
        print("Nothing to convert.")
        return

    groups = list(tiles.groupby(["CWNS_ID", "ll_uuid_primary"]))
    if args.limit:
        groups = groups[:args.limit]
    print(f"\n{len(groups)} (CWNS_ID, ll_uuid_primary) group(s) to convert "
          f"(collapsing {len(tiles)} old tile(s))")

    OUT_RGB_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NDWI_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    n_boxes_in, n_boxes_out, n_boxes_dropped_outside, n_boxes_deduped = 0, 0, 0, 0
    n_boxes_out_written = 0  # only counts boxes for groups that actually wrote a tile
    n_wide_spread_groups = 0
    failed_groups = []  # [(tile_id, reason)] -- explicit, not left to scrollback

    for i, ((cwns, primary), group) in enumerate(groups, 1):
        new_tile_id = f"{cwns}_{primary}_500m"
        new_bbox, new_ctr_lon, new_ctr_lat, spread_m = compute_new_tile_bbox(group)

        if spread_m > NEW_TILE_SIZE_M * 0.9:
            n_wide_spread_groups += 1
            print(f"  [{i}/{len(groups)}] {new_tile_id}: WARNING -- old tiles spread "
                  f"{spread_m:.0f}m, close to or over the new {NEW_TILE_SIZE_M}m tile's "
                  f"footprint. Some boxes may fall outside it below.")

        # ---- gather + remap every box from every old tile in this group ----
        remapped = []
        for _, old_tile in group.iterrows():
            old_bbox = (old_tile["bbox_xmin"], old_tile["bbox_ymin"],
                       old_tile["bbox_xmax"], old_tile["bbox_ymax"])
            old_boxes = read_yolo_boxes(old_tile["label_path"])
            n_boxes_in += len(old_boxes)
            for b in old_boxes:
                r = remap_normalized_box(b, old_bbox, int(old_tile["image_px"]),
                                        new_bbox, NEW_IMAGE_PX)
                if r is None:
                    n_boxes_dropped_outside += 1
                else:
                    remapped.append(r)

        deduped, n_dropped_dup = dedup_boxes(remapped)
        n_boxes_deduped += n_dropped_dup
        n_boxes_out += len(deduped)

        # ---- fetch the new bigger tile via Planetary Computer ----
        item = with_retry(find_naip_item, catalog, new_ctr_lon, new_ctr_lat)
        if item is None:
            reason = "no NAIP STAC item found for this location"
            print(f"  [{i}/{len(groups)}] {new_tile_id}: SKIPPED -- {reason}")
            failed_groups.append((new_tile_id, reason))
            continue
        try:
            arr4 = timed_fetch(new_bbox, item.assets["image"].href, NEW_IMAGE_PX,
                              catalog=catalog)
        except Exception as e:
            reason = f"NAIP fetch failed ({type(e).__name__}: {e})"
            print(f"  [{i}/{len(groups)}] {new_tile_id}: SKIPPED -- {reason}")
            failed_groups.append((new_tile_id, reason))
            continue
        src_res = ""  # not tracked by the STAC/COG read path the way the old
                       # USDA export response header was -- left blank rather
                       # than a fabricated value

        rgb_path = OUT_RGB_DIR / f"{new_tile_id}_rgb.png"
        ndwi_path = OUT_NDWI_DIR / f"{new_tile_id}_ndwi.tif"
        save_rgb_png(arr4, rgb_path)
        save_ndwi_tif(arr4, new_bbox, ndwi_path)

        label_path = OUT_LABEL_DIR / f"{new_tile_id}_rgb.txt"
        if deduped:
            lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in deduped]
            label_path.write_text("\n".join(lines) + "\n")
        else:
            label_path.touch()  # zero-byte -- confirmed negative, same convention as label_app.R

        meta_rows.append({
            "tile_id": new_tile_id, "source": "upscaled_500m", "CWNS_ID": cwns,
            "TRI_FACILITY_ID": "", "ll_uuid_primary": primary, "ll_uuid_alternates": "",
            "st": group["st"].iloc[0], "geoid": group["geoid"].iloc[0],
            "tile_row": 0, "tile_col": 0, "ctr_lon": new_ctr_lon, "ctr_lat": new_ctr_lat,
            "bbox_xmin": new_bbox[0], "bbox_ymin": new_bbox[1],
            "bbox_xmax": new_bbox[2], "bbox_ymax": new_bbox[3],
            "source_res_m": src_res, "target_res_m": C.TARGET_RES_M, "image_px": NEW_IMAGE_PX,
            "acq_date": "", "rgb_path": str(rgb_path), "ndwi_path": str(ndwi_path),
            "label": "", "n_old_tiles_merged": len(group), "n_boxes": len(deduped),
        })

        print(f"  [{i}/{len(groups)}] {new_tile_id}: {len(group)} old tile(s) -> 1, "
              f"{len(deduped)} box(es) ({n_dropped_dup} dup(s) merged)")
        n_boxes_out_written += len(deduped)

    pd.DataFrame(meta_rows).reindex(columns=C.METADATA_COLUMNS + ["n_old_tiles_merged", "n_boxes"]) \
        .to_csv(OUT_METADATA_CSV, index=False)

    print(f"\n=== Summary ===")
    print(f"  Groups converted        : {len(meta_rows)} / {len(groups)}")
    print(f"  Old tiles collapsed     : {len(tiles)} -> {len(meta_rows)}")
    print(f"  Boxes in (old)          : {n_boxes_in}  (across ALL {len(groups)} "
          f"attempted groups, including any that failed below)")
    print(f"  Boxes out (remapped+deduped, all attempted groups): {n_boxes_out}")
    print(f"  Boxes ACTUALLY WRITTEN (only the {len(meta_rows)} successful group(s)): "
          f"{n_boxes_out_written}")
    print(f"  Duplicate boxes merged  : {n_boxes_deduped}")
    print(f"  Boxes dropped (fell outside new tile): {n_boxes_dropped_outside}")
    print(f"  Groups with wide spread (check these): {n_wide_spread_groups}")
    if failed_groups:
        print(f"\n  Failed group(s) ({len(failed_groups)}):")
        for tile_id, reason in failed_groups:
            print(f"    {tile_id}: {reason}")
    print(f"\nWritten to:")
    print(f"  {OUT_RGB_DIR}")
    print(f"  {OUT_NDWI_DIR}")
    print(f"  {OUT_LABEL_DIR}")
    print(f"  {OUT_METADATA_CSV}")
    print(f"See this script's docstring (FOLDING IN) for how to merge into the live dataset.")


if __name__ == "__main__":
    main()