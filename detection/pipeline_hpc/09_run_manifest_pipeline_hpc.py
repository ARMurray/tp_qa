"""
09_run_manifest_pipeline_hpc.py
================================
Manifest-driven counterpart to 07_run_state_pipeline_hpc.py.

07 answers "what infrastructure is on each plant's best-available parcel" for a
whole state. This answers "what infrastructure is on THIS SPECIFIC parcel" for
an arbitrary list of (sample, parcel) pairs. That generalization is what makes
the same code serve three consumers:

    Stage 1 training   -- manifest from 08_build_training_manifest_hpc.py
                          (label-matched parcels, Correct and Incorrect)
    Stage 1 inference  -- manifest of each plant's REPORTED parcel
    Stage 3            -- manifest of each flagged plant's shortlist parcels

Only the manifest changes. Tile geometry, model, and output schema do not.

WHAT CHANGED FROM 07, AND WHY
-----------------------------
1. Primary key is `sample_id`, not `CWNS_ID`. A plant appears twice in the
   training manifest (Correct + Incorrect) and up to ~20 times in a Stage 3
   shortlist. 07's tile_id construction and resume logic both assume CWNS_ID
   is unique and would collide.

2. The parcel comes from the manifest's `ll_uuid`, not from a spatial join.
   07 did `primary = uuids[0]` off an unsorted unique() -- non-deterministic
   when a point intersects overlapping Regrid parcels. Here the parcel is an
   explicit input, decided once in 08 and auditable.

3. **A `samples` row is ALWAYS emitted, including when zero objects are
   detected.** 07 does `if plant_row: plants_buf.append(...)`, and
   select_plant_coordinate() returns None on no detections -- so plants with
   nothing found vanish from the output entirely. For production inference
   that's fine (nothing to correct to). For TRAINING it is fatal: "clean
   imagery, no infrastructure found" is the single most informative negative
   in the whole dataset, and silently dropping it would leave Stage 1 trained
   only on parcels where something was detected. Every manifest row gets a row
   out, with n_objects = 0 where applicable.

4. Per-class object counts are emitted for a FIXED class list (--class-list,
   default C.CLASSES -- all six), not just the classes the current model
   happens to predict. The deployed model is a 3-class subset today and will
   gain oxidation_pond later. Holding the column set stable means dropping in
   a new model adds VALUES to existing columns rather than adding columns, so
   the downstream Stage 1 feature join and any trained model keep working.
   `model_classes` records what the model could actually predict, so an
   all-zero n_obj_oxidation_pond is distinguishable from a true absence.

5. Output goes to its own root (TRAIN_INFERENCE_ROOT), not the production
   inference_state/ tree. Training detections must never silently mix with
   production ones -- they're computed on different parcels for the same plants.

Usage:
    python 09_run_manifest_pipeline_hpc.py --state 39
    python 09_run_manifest_pipeline_hpc.py --state 39 --manifest path/to/other.parquet
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
from PIL import Image
from rasterio.io import MemoryFile
from shapely import from_wkb
from shapely.geometry import box, Point as ShapelyPoint
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

try:
    from state_fips import fips_to_abbr
except ImportError:
    from state_fips_hpc import fips_to_abbr

REQUEST_TIMEOUT = 60
EDGE_TOUCH_THRESHOLD_PX = 2.0
NODATA_STD_THRESHOLD = 1.0
MODEL_IMGSZ = -(-C.IMAGE_PX // 32) * 32   # 333 -> 352

# Defaults resolved via getattr so this runs before config.py is edited.
DEFAULT_MANIFEST = getattr(C, "TRAIN_MANIFEST", C.SAMPLES_DIR / "training_manifest.parquet")
DEFAULT_TILES_ROOT = getattr(C, "TRAIN_TILES_ROOT", C.DATA_DIR / "tiles_train")
DEFAULT_OUT_ROOT = getattr(C, "TRAIN_INFERENCE_ROOT", C.DATA_DIR / "inference_train")

_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        _local.session = s
    return s


# ===========================================================================
# Tile geometry / NAIP fetch -- byte-for-byte the same contract as 07.
# Any divergence here silently breaks comparability between training features
# and inference features, which is the one thing that must not happen.
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
    return [tuple(map(float, row)) for row in boxes.bounds.to_numpy()]


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
    t0 = time.time()
    arr = fetch_naip_tile(bbox)
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


def geographic_nms(raw_dets: list[dict], nms_dist: float) -> list[dict]:
    """Unchanged from 07. Note the open issue: nms_dist defaults to 12m, which
    is smaller than a large clarifier's diameter, so one tank can yield several
    'objects'. That inflates n_objects. Until it's fixed, prefer
    has_<class> / n_distinct_classes over raw counts as Stage 1 features --
    they're robust to the over-splitting."""
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


# ===========================================================================
# Per-sample: extract -> infer -> keep-or-delete -> nms -> feature row
# ===========================================================================

# ===========================================================================
# Analysis regions: window (primary) + parcel (secondary, capped)
# ===========================================================================
def build_regions(row, pgeom, args):
    """Returns (window_geom, parcel_geom_or_None, tile_bounds, meta).

    The WINDOW is a fixed square centred on the reported point -- identical
    spatial support for every sample nationally, and the only region available
    where no parcel exists (Tier2_no_parcel).

    The PARCEL is secondary and suppressed when it looks like a Regrid coverage
    gap. NOTE the suppression test is on bounding-box EXTENT, not area. Area
    alone does not bound tiling cost: generate_tile_grid works off bounds, and
    multipart or sliver parcels -- utility corridors, river easements,
    scattered municipal holdings -- have tiny area with enormous extent. The
    national run caught 41000011001 at 0.032 km2 (30x under the area cap)
    pulling 168 tiles, and 24000102001 at 0.72 km2 pulling 1,287.

    Finally the tile region is HARD-CLIPPED to --max-extent-m centred on the
    reported point, so worst-case cost is bounded no matter what the parcel
    geometry does. parcel_clipped records when that clip actually bit, since
    the parcel_* counts are then an undercount rather than a measurement.
    """
    from shapely.geometry import Point
    pt = gpd.GeoSeries([Point(row["rep_lon"], row["rep_lat"])],
                       crs=C.EXPORT_CRS).to_crs(C.PROJECTED_CRS).iloc[0]
    h = args.window_m / 2.0
    window = box(pt.x - h, pt.y - h, pt.x + h, pt.y + h)

    area_km2 = float("nan")
    extent_m = float("nan")
    reason = None
    parcel = None

    if pgeom is not None:
        area_km2 = float(pgeom.area / 1e6)
        px0, py0, px1, py1 = pgeom.bounds
        extent_m = float(max(px1 - px0, py1 - py0))
        if area_km2 > args.max_parcel_km2:
            reason = "area"
        elif extent_m > args.max_parcel_extent_m:
            reason = "extent"
        else:
            parcel = pgeom

    if parcel is None:
        bounds = window.bounds
    else:
        wx0, wy0, wx1, wy1 = window.bounds
        px0, py0, px1, py1 = parcel.bounds
        bounds = (min(wx0, px0), min(wy0, py0), max(wx1, px1), max(wy1, py1))

    # Hard clip -- the backstop that makes cost predictable regardless of geometry.
    m = args.max_extent_m / 2.0
    clip = (pt.x - m, pt.y - m, pt.x + m, pt.y + m)
    clipped_bounds = (max(bounds[0], clip[0]), max(bounds[1], clip[1]),
                      min(bounds[2], clip[2]), min(bounds[3], clip[3]))
    was_clipped = clipped_bounds != bounds
    bounds = clipped_bounds

    meta = dict(parcel_area_km2=area_km2, parcel_extent_m=extent_m,
                parcel_oversized=bool(reason is not None),
                parcel_oversized_reason=reason,
                parcel_clipped=bool(was_clipped and parcel is not None),
                tiling_mode="window" if parcel is None else "window+parcel",
                window_m=args.window_m, max_parcel_km2=args.max_parcel_km2,
                max_parcel_extent_m=args.max_parcel_extent_m,
                max_extent_m=args.max_extent_m)
    return window, parcel, bounds, meta


def region_features(objects, region_geom, row, prefix, class_list):
    """Feature block for one analysis region. Objects are filtered by whether
    their merged centroid falls inside the region, so both blocks come from
    the same detection pass."""
    f = {}
    if region_geom is None:
        f[f"{prefix}_n_objects"] = -1          # sentinel: region not evaluated
        f[f"{prefix}_max_confidence"] = np.nan
        f[f"{prefix}_n_distinct_classes"] = -1
        f[f"{prefix}_dominant_class"] = None
        f[f"{prefix}_dist_nearest_object_m"] = np.nan
        f[f"{prefix}_offset_m"] = np.nan
        f[f"{prefix}_objects_per_ha"] = np.nan
        for c in class_list:
            f[f"{prefix}_n_obj_{c}"] = -1
            f[f"{prefix}_has_{c}"] = None
        return f

    pts = gpd.GeoSeries(
        [ShapelyPoint(o["lon"], o["lat"]) for o in objects],
        crs=C.EXPORT_CRS).to_crs(C.PROJECTED_CRS) if objects else None
    inside = [o for o, g in zip(objects, pts) if region_geom.contains(g)] if objects else []

    f[f"{prefix}_n_objects"] = len(inside)
    if inside:
        odf = pd.DataFrame(inside)
        w = odf["max_confidence"].to_numpy()
        dlon = float(np.average(odf["lon"], weights=w))
        dlat = float(np.average(odf["lat"], weights=w))
        dists = [meters_between(row["rep_lon"], row["rep_lat"], o["lon"], o["lat"])
                 for o in inside]
        f[f"{prefix}_max_confidence"] = float(odf["max_confidence"].max())
        f[f"{prefix}_n_distinct_classes"] = int(odf["class_name"].nunique())
        f[f"{prefix}_dominant_class"] = str(odf["class_name"].value_counts().index[0])
        f[f"{prefix}_dist_nearest_object_m"] = float(min(dists))
        f[f"{prefix}_offset_m"] = meters_between(row["rep_lon"], row["rep_lat"], dlon, dlat)
        counts = odf["class_name"].value_counts().to_dict()
    else:
        f[f"{prefix}_max_confidence"] = np.nan
        f[f"{prefix}_n_distinct_classes"] = 0
        f[f"{prefix}_dominant_class"] = None
        f[f"{prefix}_dist_nearest_object_m"] = np.nan
        f[f"{prefix}_offset_m"] = np.nan
        counts = {}

    for c in class_list:
        f[f"{prefix}_n_obj_{c}"] = int(counts.get(c, 0))
        f[f"{prefix}_has_{c}"] = bool(counts.get(c, 0) > 0)

    ha = region_geom.area / 10_000.0
    f[f"{prefix}_objects_per_ha"] = float(len(inside) / ha) if ha > 0 else np.nan
    return f


def detection_state(n_objects, region_evaluated, imagery_ok):
    """Three-state encoding. THIS IS THE COLUMN STAGE 1 SHOULD USE, not a raw count.

    National paired result: among the 297 matched pairs, the detector fires on
    only 53.7% of KNOWN-GOOD locations, and 45.8% of pairs are zero-on-both.
    But among pairs it does decide, it is right 84.1% of the time (132 vs 25).

    So the detector is a high-precision, low-recall witness. Feeding
    n_objects == 0 to Stage 1 as a low value on a continuous scale asserts
    "no infrastructure here", which is wrong roughly half the time -- and
    wrong non-randomly, since misses concentrate in small rural plants
    (the deployed model has no oxidation_pond class). Encoding non-detection
    as its own level lets Stage 1 learn how much a silence is worth instead
    of being told it means absence.
    """
    if not region_evaluated:
        return "not_evaluated"
    if not imagery_ok:
        return "no_imagery"
    return "detected" if n_objects > 0 else "none_found"


def process_sample(row, pgeom, model, class_names, args, executor,
                   tile_dir, model_weights_name):
    sample_id = row["sample_id"]
    window_geom, parcel_geom, tile_bounds, region_meta = build_regions(row, pgeom, args)
    grid = generate_tile_grid(tile_bounds)
    bboxes = tiles_to_wgs84(grid)

    futures = {executor.submit(timed_fetch, bbox): (t, bbox)
               for t, bbox in zip(grid, bboxes)}

    tile_rows, detection_rows = [], []
    kept_n = deleted_n = failed_n = nodata_n = 0
    fetch_time_total = infer_time_total = 0.0

    for fut in as_completed(futures):
        t, bbox = futures[fut]
        tile_id = f"{sample_id}_r{t['row']:02d}_c{t['col']:02d}"
        processed_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        base = dict(
            tile_id=tile_id, sample_id=sample_id, CWNS_ID=row["CWNS_ID"],
            label_class=row["label_class"], st=row["st"], geoid=row["geoid"],
            state_fips=row["state_fips"], parcel_id=str(row["ll_uuid"]),
            tile_row=t["row"], tile_col=t["col"],
            bbox_xmin=bbox[0], bbox_ymin=bbox[1], bbox_xmax=bbox[2], bbox_ymax=bbox[3],
            tile_size_m=C.TILE_SIZE_M, target_res_m=C.TARGET_RES_M, image_px=C.IMAGE_PX,
            model_weights=model_weights_name, conf_threshold=args.conf,
            iou_threshold=args.iou, processed_at=processed_at,
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
        if is_nodata:
            nodata_n += 1
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
                                  fetch_time_s=fetch_time, infer_time_s=infer_time,
                                  fetch_error=None))
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
                    detection_id=f"{tile_id}_d{i:03d}", tile_id=tile_id,
                    sample_id=sample_id, CWNS_ID=row["CWNS_ID"],
                    label_class=row["label_class"],
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
                                  fetch_time_s=fetch_time, infer_time_s=infer_time,
                                  fetch_error=None))

    objects = geographic_nms(detection_rows, args.nms_dist)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for idx, o in enumerate(objects):
        o.update(object_id=f"{sample_id}_obj{idx:02d}", sample_id=sample_id,
                 CWNS_ID=row["CWNS_ID"], label_class=row["label_class"],
                 st=row["st"], geoid=row["geoid"], state_fips=row["state_fips"],
                 parcel_id=str(row["ll_uuid"]),
                 model_weights=model_weights_name, processed_at=now)

    sample_row = build_sample_features(
        row, objects, tile_rows, detection_rows, args, model_weights_name,
        class_names, now, window_geom, parcel_geom, region_meta,
        counts=dict(kept=kept_n, deleted=deleted_n, failed=failed_n, nodata=nodata_n))

    return sample_row, objects, detection_rows, tile_rows, dict(
        kept=kept_n, deleted=deleted_n, failed=failed_n,
        fetch_time=fetch_time_total, infer_time=infer_time_total)


def build_sample_features(row, objects, tile_rows, detection_rows, args,
                          model_weights_name, class_names, now,
                          window_geom, parcel_geom, region_meta, counts):
    """One row per manifest sample -- ALWAYS, including zero-detection parcels.

    Emits TWO parallel feature blocks from the same detection pass:
      win_*    fixed --window-m square on the reported point. Primary. Constant
               spatial support nationally, so counts are comparable and the
               block exists even where no parcel does.
      parcel_* actual parcel geometry. Secondary, and suppressed (-1 / NaN
               sentinels, parcel_oversized=True) above --max-parcel-km2.

    Sentinels are deliberately distinct from zeros: -1 means "region not
    evaluated", 0 means "looked and found nothing". Collapsing those two into
    NA would let Stage 1 read a Regrid coverage gap as an absence of
    infrastructure.
    """
    n_attempted = len(tile_rows)
    n_ok = counts["kept"] + counts["deleted"]

    feat = dict(
        sample_id=row["sample_id"], CWNS_ID=row["CWNS_ID"],
        label_class=row["label_class"], st=row["st"], geoid=row["geoid"],
        state_fips=row["state_fips"], ll_uuid=str(row["ll_uuid"]),
        parcel_area_m2=float(row["parcel_area_m2"]) if pd.notna(row["parcel_area_m2"]) else np.nan,
        n_parcel_matches=int(row["n_parcel_matches"]),
        rep_lon=float(row["rep_lon"]), rep_lat=float(row["rep_lat"]),
        n_tiles_attempted=n_attempted,
        n_tiles_kept=counts["kept"],
        n_tiles_deleted=counts["deleted"],
        n_tiles_fetch_failed=counts["failed"],
        n_tiles_nodata=counts["nodata"],
        imagery_ok=bool(n_ok > 0 and counts["nodata"] < n_ok),
        n_detections_raw=len(detection_rows),
        n_objects_total=len(objects),
    )
    feat.update(region_meta)
    feat.update(region_features(objects, window_geom, row, "win", args.class_list))
    feat.update(region_features(objects, parcel_geom, row, "parcel", args.class_list))

    for pfx, geom in (("win", window_geom), ("parcel", parcel_geom)):
        feat[f"{pfx}_detection_state"] = detection_state(
            feat[f"{pfx}_n_objects"], geom is not None, feat["imagery_ok"])
        feat[f"{pfx}_any_object"] = (
            None if geom is None else bool(feat[f"{pfx}_n_objects"] > 0))

    feat.update(
        model_weights=model_weights_name,
        model_classes=",".join(sorted(class_names.values())),
        tile_size_m=C.TILE_SIZE_M, target_res_m=C.TARGET_RES_M, image_px=C.IMAGE_PX,
        conf_threshold=args.conf, iou_threshold=args.iou, nms_dist_m=args.nms_dist,
        processed_at=now,
    )
    return feat


def flush_parquet(rows, out_root: Path, table_name: str, state_fips: str):
    if not rows:
        return
    part_dir = out_root / table_name / f"state={state_fips}"
    part_dir.mkdir(parents=True, exist_ok=True)
    part_path = part_dir / f"part-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}.parquet"
    pd.DataFrame(rows).to_parquet(part_path, engine="pyarrow", index=False)


def load_processed_ids(samples_state_dir: Path) -> set:
    if not samples_state_dir.exists() or not any(samples_state_dir.glob("*.parquet")):
        return set()
    ids = set()
    for f in samples_state_dir.glob("*.parquet"):
        try:
            ids |= set(pd.read_parquet(f, columns=["sample_id"])["sample_id"].astype(str))
        except Exception:
            pass
    return ids


def resolve_weights(explicit, models_dir: Path) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        print(f"  NOTE: --weights {p} not found, falling back to newest .pt in {models_dir}")
    candidates = sorted(models_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No .pt weights in {models_dir} and --weights unusable.")
    chosen = candidates[0]
    print(f"  Using weights: {chosen.name}"
          + (f"  ({len(candidates)} .pt found, picked newest by mtime)" if len(candidates) > 1 else ""))
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="2-digit Census state FIPS, e.g. 39")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--tiles-root", default=str(DEFAULT_TILES_ROOT))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--models-dir", default=str(C.MODELS_ROOT))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--nms-dist", type=float, default=12.0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--window-m", type=float, default=600.0,
                    help="side length of the fixed analysis window centred on the reported "
                         "point. PRIMARY feature region -- identical support for every "
                         "sample nationally, and the only region available where no parcel "
                         "exists. 600m ~= 9 tiles at 134m stride.")
    ap.add_argument("--max-parcel-extent-m", type=float, default=1200.0,
                    help="suppress parcel_* features when the parcel bounding box is "
                         "wider/taller than this. Catches sliver and multipart parcels "
                         "that pass the area cap but have huge extent -- the actual "
                         "driver of tiling cost.")
    ap.add_argument("--max-extent-m", type=float, default=1600.0,
                    help="hard clip on the tile region, centred on the reported point. "
                         "Backstop that bounds worst-case cost (~100 tiles) regardless "
                         "of parcel geometry.")
    ap.add_argument("--max-parcel-km2", type=float, default=1.0,
                    help="parcels larger than this are treated as Regrid coverage gaps: "
                         "parcel_* features are suppressed (sentinel -1) and only the window "
                         "is used. 1.0 km2 is the national 95th percentile (130 of 2,591 "
                         "samples, 24 of 254 Incorrect).")
    ap.add_argument("--class-list", type=str, default=",".join(C.CLASSES),
                    help="fixed class column set for the samples table; keep constant "
                         "across model versions so the schema never shifts")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    args.class_list = [c.strip() for c in args.class_list.split(",") if c.strip()]

    state_fips = str(args.state).zfill(2)
    state_abbr = fips_to_abbr(state_fips)
    out_root = Path(args.out_root)
    tile_dir = Path(args.tiles_root) / state_fips
    tile_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== 09_run_manifest_pipeline_hpc.py: state {state_fips} ({state_abbr}) ===\n")
    print(f"Manifest: {args.manifest}")
    print(f"Device: {args.device} | Workers: {args.workers} | "
          f"conf={args.conf} iou={args.iou} nms_dist={args.nms_dist}m")
    print(f"Class columns: {args.class_list}")
    print(f"Window: {args.window_m}m  |  Parcel cap: {args.max_parcel_km2} km2 / "
          f"{args.max_parcel_extent_m}m extent  |  Hard clip: {args.max_extent_m}m\n")

    man = pd.read_parquet(args.manifest)
    man["state_fips"] = man["state_fips"].astype(str).str.zfill(2)
    man = man[man["state_fips"] == state_fips]
    n_all = len(man)

    man = man.copy()
    n_noparcel = int((~man["parcel_found"].astype(bool)).sum())
    print(f"Manifest rows for {state_fips}: {n_all} "
          f"({n_all - n_noparcel} with a parcel, {n_noparcel} without)")
    if n_noparcel:
        print("  No-parcel samples ARE processed: the fixed window doesn't need a parcel, "
              "and it is the only detection evidence available for the Tier2_no_parcel "
              "population. They get win_* features and parcel_* sentinels.")

    samples_state_dir = out_root / "samples" / f"state={state_fips}"
    if not args.no_resume:
        done = load_processed_ids(samples_state_dir)
        if done:
            before = len(man)
            man = man[~man["sample_id"].isin(done)]
            print(f"Resuming: skipping {before - len(man)} already-processed samples")

    if len(man) == 0:
        print("Nothing to do.")
        return

    weights_path = resolve_weights(args.weights, Path(args.models_dir))
    model_weights_name = weights_path.name
    model = YOLO(str(weights_path))
    class_names = {int(k): v for k, v in model.names.items()}
    print(f"Model classes: {sorted(class_names.values())}")
    missing = set(args.class_list) - set(class_names.values())
    if missing:
        print(f"  NOTE: {sorted(missing)} are in --class-list but NOT predicted by this "
              f"model. Their columns will be all-zero by construction, not by absence "
              f"of infrastructure. `model_classes` records this per row -- Stage 1 must "
              f"not read those zeros as evidence.\n")

    counties = man[["st", "geoid"]].dropna().drop_duplicates()
    print(f"Counties to process: {len(counties)}\n")

    totals = dict(samples=0, kept=0, deleted=0, failed=0, fetch_time=0.0, infer_time=0.0)
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for ci, (_, cc) in enumerate(counties.iterrows(), start=1):
            geoid = cc["geoid"]
            sub = man[(man["st"] == cc["st"]) & (man["geoid"] == geoid)]

            parcels = load_parcels(cc["st"], geoid)
            if parcels is not None and len(parcels):
                parcels = parcels.to_crs(C.PROJECTED_CRS).set_index(C.PARCEL_ID_FIELD)
            else:
                parcels = None
                print(f"[{ci}/{len(counties)}] {cc['st']}/{geoid}: no parcel file -- "
                      f"{len(sub)} samples run window-only")

            tiles_buf, dets_buf, objs_buf, samples_buf = [], [], [], []

            for _, row in sub.iterrows():
                pgeom = None
                if parcels is not None and pd.notna(row["ll_uuid"]):
                    try:
                        pgeom = parcels.loc[row["ll_uuid"], "geometry"]
                        if hasattr(pgeom, "__len__"):   # duplicate ll_uuid in the file
                            pgeom = pgeom.iloc[0]
                    except KeyError:
                        print(f"  WARNING: ll_uuid {row['ll_uuid']} not in {geoid}.parquet "
                              f"({row['sample_id']}) -- parcel store changed since 08 ran; "
                              f"running window-only")

                sample_row, objects, det_rows, tile_rows, counts = process_sample(
                    row, pgeom, model, class_names, args, executor,
                    tile_dir, model_weights_name)

                tiles_buf.extend(tile_rows)
                dets_buf.extend(det_rows)
                objs_buf.extend(objects)
                samples_buf.append(sample_row)   # ALWAYS -- see docstring #3

                totals["samples"] += 1
                for k in ("kept", "deleted", "failed"):
                    totals[k] += counts[k]
                totals["fetch_time"] += counts["fetch_time"]
                totals["infer_time"] += counts["infer_time"]

                if totals["samples"] % 25 == 0:
                    print(f"  ...{totals['samples']} samples done "
                          f"({totals['kept']} tiles kept, {totals['deleted']} deleted, "
                          f"{(time.time() - t_start)/60:.1f} min)")

            flush_parquet(tiles_buf, out_root, "tiles", state_fips)
            flush_parquet(dets_buf, out_root, "detections", state_fips)
            flush_parquet(objs_buf, out_root, "objects", state_fips)
            flush_parquet(samples_buf, out_root, "samples", state_fips)
            print(f"[{ci}/{len(counties)}] {cc['st']}/{geoid}: "
                  f"{len(samples_buf)} samples, {len(tiles_buf)} tiles flushed")

    elapsed = time.time() - t_start
    n_ok = totals["kept"] + totals["deleted"]
    print("\n=== Summary ===")
    print(f"  State            : {state_fips} ({state_abbr})")
    print(f"  Samples processed: {totals['samples']}")
    print(f"  Tiles kept       : {totals['kept']}")
    print(f"  Tiles deleted    : {totals['deleted']}")
    print(f"  Fetch failures   : {totals['failed']}")
    print(f"  Elapsed          : {elapsed/60:.1f} min")
    if n_ok:
        print(f"  Mean fetch time  : {totals['fetch_time']/n_ok:.3f}s/tile")
        print(f"  Mean infer time  : {totals['infer_time']/n_ok:.3f}s/tile")
        print(f"  Tiles per sample : {n_ok/max(totals['samples'],1):.1f}")
    print(f"\nOutputs in: {out_root}")
    for t in ("tiles", "detections", "objects", "samples"):
        print(f"  {out_root}/{t}/state={state_fips}/")
    print("\nStage 1 joins on: samples/  ->  sample_id / CWNS_ID + label_class")


if __name__ == "__main__":
    main()
