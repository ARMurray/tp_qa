"""
07_run_state_pipeline_hpc.py
=============================
Fused per-state HPC pipeline: extract NAIP tiles for one state's treatment
plants, run trained YOLO inference on each tile as it's produced, and delete
tiles with zero detections immediately -- keeping disk usage bounded to
roughly one state's worth of tiles at a time.

Writes FOUR append-only Parquet tables under C.STATE_INFERENCE_ROOT, one
partition folder per state, one immutable part-file per flush (never
overwritten -- re-running or retrying just adds more history):

    tiles/state={fips}/part-*.parquet        one row per tile ATTEMPTED
        (kept, deleted, or fetch_failed -- every tile gets a row, unlike the
        old deleted_tiles_log.csv which only recorded deletions)
    detections/state={fips}/part-*.parquet   one row per raw YOLO box, pre-dedup
    objects/state={fips}/part-*.parquet      one row per merged real-world object
    plants/state={fips}/part-*.parquet       one row per plant's final corrected coordinate

Nothing spatial is lost when a tile is deleted: its bbox lives in `tiles`
regardless of outcome, and since a NAIP tile is fully determined by its bbox
+ the fixed 200m/60cm/333px geometry, any deleted tile can be re-fetched
later from that logged bbox alone.

State selection: CWNS_ID's first two characters are the Census state FIPS
code (see state_fips.py). This is both the --state argument and the filter.

Requires the FULL plant universe gpkg from 00_build_full_plant_list_hpc.py --
NOT the capped/stratified/correct-only training sample from 01_sample_sites.py.

Model TRAINING stays local. This script only loads an already-trained best.pt.

Usage:
    python 07_run_state_pipeline_hpc.py --state 39 --weights /path/to/best.pt
"""
import argparse
import math
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import torch
from PIL import Image
from rasterio.io import MemoryFile
from shapely import from_wkb
from shapely.geometry import box
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from state_fips import fips_to_abbr

REQUEST_TIMEOUT = 60

# A raw detection box within this many pixels of the tile boundary counts as
# "touching the edge" -- small tolerance for boxes that land exactly on the
# boundary due to normal rounding, not meant to be a tight threshold.
EDGE_TOUCH_THRESHOLD_PX = 2.0

# Heuristic for flagging near-blank/nodata NAIP responses (ocean, coverage
# gaps, flight-line seams) -- these come back as a SUCCESSFUL fetch with
# near-zero pixel variance, not an HTTP error, so they need their own flag
# rather than being indistinguishable from "looked and found nothing."
NODATA_STD_THRESHOLD = 1.0

# YOLO's backbone requires imgsz to be a multiple of its max stride (32).
# C.IMAGE_PX (333px) comes from tile geometry, not YOLO's constraint, so
# Ultralytics would otherwise silently round up to 352 and warn on every
# single tile -- log spam at HPC scale. Box coordinates still come back in
# the original C.IMAGE_PX pixel space either way (Ultralytics rescales
# internally), so pixel_to_lonlat and everything downstream is unaffected.
MODEL_IMGSZ = -(-C.IMAGE_PX // 32) * 32   # ceiling to nearest multiple of 32 -> 352

_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        _local.session = s
    return s


# ===========================================================================
# Tile geometry (mirrors 02_extract_tiles.py -- kept in sync manually)
# ===========================================================================
def generate_tile_grid(bounds) -> list[dict]:
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
            tiles.append(dict(x_min=x0, y_min=y0, x_max=x0 + C.TILE_SIZE_M,
                              y_max=y0 + C.TILE_SIZE_M, row=r, col=c))
    return tiles


def tiles_to_wgs84(tiles: list[dict]):
    boxes = gpd.GeoSeries(
        [box(t["x_min"], t["y_min"], t["x_max"], t["y_max"]) for t in tiles],
        crs=C.PROJECTED_CRS,
    ).to_crs(C.EXPORT_CRS)
    b = boxes.bounds.to_numpy()
    return [tuple(map(float, row)) for row in b]


def load_parcels(state: str, geoid: str):
    path = C.PARCEL_BASE / f"state={state}" / f"{geoid}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, columns=[C.PARCEL_ID_FIELD, C.PARCEL_WKB_FIELD])
    except Exception:
        df = pd.read_parquet(path)
    if C.PARCEL_WKB_FIELD not in df.columns:
        return None
    geom = from_wkb(df[C.PARCEL_WKB_FIELD].to_numpy())
    df = df.drop(columns=[C.PARCEL_WKB_FIELD])
    return gpd.GeoDataFrame(df, geometry=geom, crs=C.EXPORT_CRS)


# ===========================================================================
# NAIP fetch (mirrors 02_extract_tiles.py)
# ===========================================================================
def fetch_naip_tile(bbox):
    params = {
        "bbox": ",".join(map(str, bbox)), "bboxSR": C.EXPORT_CRS,
        "size": f"{C.IMAGE_PX},{C.IMAGE_PX}", "imageSR": C.EXPORT_CRS,
        "format": "tiff", "pixelType": "U8",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_NearestNeighbor",
        "bandIds": "0,1,2,3", "f": "image",
    }
    resp = _session().get(C.NAIP_URL + "/exportImage", params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    with MemoryFile(resp.content) as mem, mem.open() as src:
        arr = src.read()
    return arr.astype(np.uint8)


def save_rgb_png(arr4, path):
    Image.fromarray(np.transpose(arr4[0:3], (1, 2, 0)), mode="RGB").save(path)


def timed_fetch(bbox):
    """Wraps fetch_naip_tile to measure per-tile network time, so we can
    compare against inference time and make an evidence-based CPU-vs-GPU call."""
    t0 = time.time()
    arr = fetch_naip_tile(bbox)
    return arr, time.time() - t0


def detect_nodata(arr4: np.ndarray) -> bool:
    """NAIP returns near-empty imagery (no coverage, ocean, flight-line gaps)
    as a SUCCESSFUL fetch with near-uniform pixel values, not an HTTP error --
    flag these separately from a tile that genuinely has real imagery with
    nothing detected in it."""
    return bool(arr4.max() == 0) or bool(arr4.std() < NODATA_STD_THRESHOLD)


# ===========================================================================
# Inference geometry (mirrors 05_run_inference.py)
# ===========================================================================
def pixel_to_lonlat(cx, cy, W, H, bbox):
    xmin, ymin, xmax, ymax = bbox
    return (xmin + (cx / W) * (xmax - xmin), ymax - (cy / H) * (ymax - ymin))


def meters_between(lon1, lat1, lon2, lat2):
    mlat = math.radians((lat1 + lat2) / 2.0)
    dx = (lon2 - lon1) * 111_320.0 * math.cos(mlat)
    dy = (lat2 - lat1) * 111_320.0
    return math.hypot(dx, dy)


# ===========================================================================
# Detections -> objects -> plant coordinate (now three explicit stages,
# each producing rows for its own table, instead of collapsing straight to
# a single plant-level result the way the old dedup_and_select() did)
# ===========================================================================
def geographic_nms(raw_dets: list[dict], nms_dist: float) -> list[dict]:
    """Greedy geographic NMS per class -> one row per merged real-world
    object. Each result carries member_detection_ids (audit trail back to
    `detections`) and the union geo-bbox across all merged raw boxes."""
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
                geo_xmin=float(m["geo_xmin"].min()), geo_ymin=float(m["geo_ymin"].min()),
                geo_xmax=float(m["geo_xmax"].max()), geo_ymax=float(m["geo_ymax"].max()),
            ))
    return objects


def select_plant_coordinate(objects: list[dict], strategy: str = "weighted_centroid") -> dict | None:
    """Collapses a plant's merged objects (possibly several classes) into
    ONE corrected coordinate. Weighted by each object's max_confidence,
    matching the original dedup_and_select's weighting."""
    if not objects:
        return None
    objs = pd.DataFrame(objects)
    w = objs["max_confidence"].to_numpy()
    cls_counts = objs["class_name"].value_counts()
    return dict(
        corrected_lon=float(np.average(objs["lon"], weights=w)),
        corrected_lat=float(np.average(objs["lat"], weights=w)),
        selection_strategy=strategy,
        n_objects=int(len(objs)),
        n_detections=int(objs["n_merged"].sum()),
        dominant_class=cls_counts.index[0],
        class_counts=[{"class_name": k, "count": int(v)} for k, v in cls_counts.items()],
    )


# ===========================================================================
# Per-plant: extract -> infer -> keep-or-delete each tile -> nms -> select
# ===========================================================================
def process_plant(cwns_id, primary_id, pgeom, state_abbr, geoid, state_fips,
                   orig_lon, orig_lat, model, class_names, args, executor,
                   tile_dir, model_weights_name):
    grid = generate_tile_grid(pgeom.bounds)
    bboxes = tiles_to_wgs84(grid)

    futures = {executor.submit(timed_fetch, bbox): (t, bbox)
               for t, bbox in zip(grid, bboxes)}

    tile_rows, detection_rows = [], []
    kept_n, deleted_n, failed_n = 0, 0, 0
    fetch_time_total, infer_time_total = 0.0, 0.0

    for fut in as_completed(futures):
        t, bbox = futures[fut]
        tile_id = f"{cwns_id}_{primary_id}_r{t['row']:02d}_c{t['col']:02d}"
        processed_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        base = dict(
            tile_id=tile_id, CWNS_ID=cwns_id, st=state_abbr, geoid=geoid, state_fips=state_fips,
            parcel_id=str(primary_id), tile_row=t["row"], tile_col=t["col"],
            bbox_xmin=bbox[0], bbox_ymin=bbox[1], bbox_xmax=bbox[2], bbox_ymax=bbox[3],
            tile_size_m=C.TILE_SIZE_M, target_res_m=C.TARGET_RES_M, image_px=C.IMAGE_PX,
            model_weights=model_weights_name, conf_threshold=args.conf, iou_threshold=args.iou,
            processed_at=processed_at,
        )

        try:
            arr, fetch_time = fut.result()
            fetch_time_total += fetch_time
        except requests.RequestException as e:
            failed_n += 1
            tile_rows.append(dict(**base, outcome="fetch_failed", is_nodata=False,
                                   n_raw_detections=0, image_path=None,
                                   fetch_time_s=None, infer_time_s=None, fetch_error=str(e)))
            continue

        is_nodata = detect_nodata(arr)
        rgb_path = tile_dir / f"{tile_id}_rgb.png"
        save_rgb_png(arr, rgb_path)

        t0 = time.time()
        result = model.predict(str(rgb_path), conf=args.conf, iou=args.iou,
                                imgsz=MODEL_IMGSZ, device=args.device, verbose=False)[0]
        infer_time = time.time() - t0
        infer_time_total += infer_time
        boxes = result.boxes
        n = 0 if boxes is None else len(boxes)

        if n == 0:
            rgb_path.unlink(missing_ok=True)
            deleted_n += 1
            tile_rows.append(dict(**base, outcome="deleted", is_nodata=is_nodata,
                                   n_raw_detections=0, image_path=None,
                                   fetch_time_s=fetch_time, infer_time_s=infer_time, fetch_error=None))
        else:
            kept_n += 1
            H = W = C.IMAGE_PX
            xywh = boxes.xywh.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cids = boxes.cls.cpu().numpy().astype(int)

            for i, ((cx, cy, bw, bh), (pxmin, pymin, pxmax, pymax), c, cid) in enumerate(
                    zip(xywh, xyxy, conf, cids)):
                lon, lat = pixel_to_lonlat(cx, cy, W, H, bbox)
                geo_x0, geo_y1 = pixel_to_lonlat(pxmin, pymin, W, H, bbox)
                geo_x1, geo_y0 = pixel_to_lonlat(pxmax, pymax, W, H, bbox)
                edge_margin_px = float(min(pxmin, pymin, W - pxmax, H - pymax))
                detection_rows.append(dict(
                    detection_id=f"{tile_id}_d{i:03d}", tile_id=tile_id, CWNS_ID=cwns_id,
                    class_id=int(cid), class_name=class_names.get(int(cid), str(cid)),
                    confidence=float(c),
                    px_cx=float(cx), px_cy=float(cy), px_w=float(bw), px_h=float(bh),
                    px_xmin=float(pxmin), px_ymin=float(pymin),
                    px_xmax=float(pxmax), px_ymax=float(pymax),
                    lon=float(lon), lat=float(lat),
                    geo_xmin=float(min(geo_x0, geo_x1)), geo_ymin=float(min(geo_y0, geo_y1)),
                    geo_xmax=float(max(geo_x0, geo_x1)), geo_ymax=float(max(geo_y0, geo_y1)),
                    touches_tile_edge=bool(edge_margin_px <= EDGE_TOUCH_THRESHOLD_PX),
                    edge_margin_px=edge_margin_px,
                    model_weights=model_weights_name, processed_at=processed_at,
                ))
            tile_rows.append(dict(**base, outcome="kept", is_nodata=is_nodata,
                                   n_raw_detections=n, image_path=str(rgb_path),
                                   fetch_time_s=fetch_time, infer_time_s=infer_time, fetch_error=None))

    objects = geographic_nms(detection_rows, args.nms_dist)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for idx, o in enumerate(objects):
        o.update(object_id=f"{cwns_id}_obj{idx:02d}", CWNS_ID=cwns_id, st=state_abbr,
                  geoid=geoid, state_fips=state_fips,
                  model_weights=model_weights_name, processed_at=now)

    plant_row = select_plant_coordinate(objects, strategy="weighted_centroid")
    if plant_row:
        plant_row.update(
            CWNS_ID=cwns_id, st=state_abbr, geoid=geoid, state_fips=state_fips,
            member_object_ids=[o["object_id"] for o in objects],
            orig_lon=orig_lon, orig_lat=orig_lat,
            offset_m=meters_between(orig_lon, orig_lat,
                                     plant_row["corrected_lon"], plant_row["corrected_lat"])
                      if orig_lon is not None else None,
            model_weights=model_weights_name, processed_at=now,
        )

    return plant_row, objects, detection_rows, tile_rows, dict(
        kept=kept_n, deleted=deleted_n, failed=failed_n,
        fetch_time=fetch_time_total, infer_time=infer_time_total)


# ===========================================================================
# Parquet writer (append-only: every flush is a new, uniquely-named part
# file -- nothing is ever overwritten, so retries/resumed runs just add
# more history, which is exactly what you want for a "re-test this tile"
# audit trail)
# ===========================================================================
def flush_parquet(rows: list[dict], out_root: Path, table_name: str, state_fips: str):
    if not rows:
        return
    df = pd.DataFrame(rows)
    part_dir = out_root / table_name / f"state={state_fips}"
    part_dir.mkdir(parents=True, exist_ok=True)
    part_path = part_dir / f"part-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}.parquet"
    df.to_parquet(part_path, engine="pyarrow", index=False)


def load_processed_ids(plants_state_dir: Path) -> set:
    if not plants_state_dir.exists() or not any(plants_state_dir.glob("*.parquet")):
        return set()
    try:
        df = pd.read_parquet(plants_state_dir, columns=["CWNS_ID"])
        return set(df["CWNS_ID"].astype(str))
    except Exception:
        ids = set()
        for f in plants_state_dir.glob("*.parquet"):
            try:
                ids |= set(pd.read_parquet(f, columns=["CWNS_ID"])["CWNS_ID"].astype(str))
            except Exception:
                pass
        return ids


def resolve_weights(explicit: str | None, models_dir: Path) -> Path:
    """Use --weights if given and it exists; otherwise pick the most recently
    modified .pt file in models_dir. Avoids hardcoding a specific filename
    that goes stale the moment you train a new round."""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        print(f"  NOTE: --weights {p} not found, falling back to newest .pt in {models_dir}")

    candidates = sorted(models_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No .pt weights found in {models_dir}, and --weights was not given "
            f"(or didn't point at a real file). Upload a trained model first."
        )
    chosen = candidates[0]
    if len(candidates) > 1:
        print(f"  Auto-selected newest weights: {chosen.name} "
              f"({len(candidates)} .pt files found in {models_dir}, picked most recently modified)")
    else:
        print(f"  Using weights: {chosen.name}")
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="2-digit Census state FIPS code, e.g. 39")
    ap.add_argument("--weights", default=None,
                     help="path to a specific .pt file. Omit to auto-select the newest .pt in --models-dir")
    ap.add_argument("--models-dir", default=str(C.MODELS_ROOT),
                     help="directory to search for weights when --weights isn't given")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--nms-dist", type=float, default=12.0)
    ap.add_argument("--workers", type=int, default=10, help="concurrent NAIP fetch threads")
    ap.add_argument("--device", type=str, default="cpu",
                     help="'cpu' (default -- avoids queuing for shared GPUs) or 'cuda' to opt in")
    ap.add_argument("--no-resume", action="store_true",
                     help="reprocess plants even if already in the plants Parquet table")
    args = ap.parse_args()

    state_fips = str(args.state).zfill(2)
    state_abbr = fips_to_abbr(state_fips)
    print(f"=== 07_run_state_pipeline_hpc.py: state {state_fips} ({state_abbr}) ===\n")
    print(f"Device: {args.device}  |  Workers: {args.workers}  |  "
          f"conf={args.conf} iou={args.iou} nms_dist={args.nms_dist}m\n")

    tile_dir = C.STATE_TILES_ROOT / state_fips
    tile_dir.mkdir(parents=True, exist_ok=True)
    out_root = C.STATE_INFERENCE_ROOT

    weights_path = resolve_weights(args.weights, Path(args.models_dir))
    model_weights_name = weights_path.name
    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))
    class_names = {int(k): v for k, v in model.names.items()}
    print(f"Classes: {class_names}\n")

    plants = gpd.read_file(C.ALL_PLANTS_GPKG, layer=C.ALL_PLANTS_LAYER)
    plants["CWNS_ID"] = plants["CWNS_ID"].astype(str)
    plants = plants[plants["CWNS_ID"].str.startswith(state_fips)]
    # Capture the reported lon/lat BEFORE reprojecting to the projected CRS
    # used for tiling -- these ride along as plain numeric columns through
    # the CRS conversion and spatial join, ending up in the `plants` table
    # as orig_lon/orig_lat/offset_m.
    plants["orig_lon"] = plants.geometry.x
    plants["orig_lat"] = plants.geometry.y
    plants = plants.to_crs(C.PROJECTED_CRS)
    print(f"Plants in state {state_fips}: {len(plants)}")

    plants_state_dir = out_root / "plants" / f"state={state_fips}"
    if not args.no_resume:
        done_ids = load_processed_ids(plants_state_dir)
        if done_ids:
            before = len(plants)
            plants = plants[~plants["CWNS_ID"].isin(done_ids)]
            print(f"Resuming: skipping {before - len(plants)} already-processed plants")

    if len(plants) == 0:
        print("Nothing to do.")
        return

    counties = plants[["st", "geoid"]].drop_duplicates()
    print(f"Counties to process: {len(counties)}\n")

    totals = dict(plants=0, kept=0, deleted=0, failed=0, fetch_time=0.0, infer_time=0.0)
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for ci, (_, cc) in enumerate(counties.iterrows(), start=1):
            geoid = cc["geoid"]
            parcels = load_parcels(state_abbr, geoid)
            if parcels is None or len(parcels) == 0:
                print(f"[{ci}/{len(counties)}] {state_abbr}/{geoid}: no parcels, skipping")
                continue
            parcels = parcels.to_crs(C.PROJECTED_CRS)[[C.PARCEL_ID_FIELD, "geometry"]]

            county_plants = plants[(plants["st"] == state_abbr) & (plants["geoid"] == geoid)]
            joined = gpd.sjoin(county_plants, parcels, how="left", predicate="intersects")

            # Per-county buffers, flushed to Parquet once at the end of each
            # county -- bounds memory to ~one county's worth of rows at a time.
            tiles_buf, dets_buf, objs_buf, plants_buf = [], [], [], []

            for cwns, grp in joined.groupby("CWNS_ID"):
                uuids = list(grp[C.PARCEL_ID_FIELD].dropna().unique())
                if not uuids:
                    continue
                primary = uuids[0]
                pgeom = parcels.loc[parcels[C.PARCEL_ID_FIELD] == primary, "geometry"].iloc[0]
                orig_lon = float(grp["orig_lon"].iloc[0])
                orig_lat = float(grp["orig_lat"].iloc[0])

                plant_row, objects, det_rows, tile_rows, counts = process_plant(
                    cwns, primary, pgeom, state_abbr, geoid, state_fips,
                    orig_lon, orig_lat, model, class_names, args, executor,
                    tile_dir, model_weights_name)

                tiles_buf.extend(tile_rows)
                dets_buf.extend(det_rows)
                objs_buf.extend(objects)
                if plant_row:
                    plants_buf.append(plant_row)

                totals["plants"] += 1
                totals["kept"]    += counts["kept"]
                totals["deleted"] += counts["deleted"]
                totals["failed"]  += counts["failed"]
                totals["fetch_time"] += counts["fetch_time"]
                totals["infer_time"] += counts["infer_time"]

                if totals["plants"] % 25 == 0:
                    elapsed = time.time() - t_start
                    print(f"  ...{totals['plants']} plants done "
                          f"({totals['kept']} tiles kept, {totals['deleted']} deleted, "
                          f"{elapsed/60:.1f} min elapsed)")

            flush_parquet(tiles_buf, out_root, "tiles", state_fips)
            flush_parquet(dets_buf, out_root, "detections", state_fips)
            flush_parquet(objs_buf, out_root, "objects", state_fips)
            flush_parquet(plants_buf, out_root, "plants", state_fips)

            print(f"[{ci}/{len(counties)}] {state_abbr}/{geoid}: done "
                  f"({len(tiles_buf)} tiles, {len(plants_buf)} plants flushed)")

    elapsed = time.time() - t_start
    n_ok = totals["kept"] + totals["deleted"]
    print("\n=== Summary ===")
    print(f"  State           : {state_fips} ({state_abbr})")
    print(f"  Plants processed: {totals['plants']}")
    print(f"  Tiles kept      : {totals['kept']}")
    print(f"  Tiles deleted   : {totals['deleted']}")
    print(f"  Fetch failures  : {totals['failed']}")
    print(f"  Elapsed         : {elapsed/60:.1f} min")
    if n_ok:
        print(f"  Mean fetch time : {totals['fetch_time']/n_ok:.3f}s/tile (network, concurrent across {args.workers} workers)")
        print(f"  Mean infer time : {totals['infer_time']/n_ok:.3f}s/tile (device={args.device}, sequential in main thread)")
    print(f"\nOutputs (Parquet, partitioned by state) in: {out_root}")
    print(f"  {out_root}/tiles/state={state_fips}/")
    print(f"  {out_root}/detections/state={state_fips}/")
    print(f"  {out_root}/objects/state={state_fips}/")
    print(f"  {out_root}/plants/state={state_fips}/")


if __name__ == "__main__":
    main()
