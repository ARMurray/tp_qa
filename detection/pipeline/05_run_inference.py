"""
05_run_inference.py
==================
Phase 5 — Inference & Coordinate Correction.

Applies the best trained YOLOv8 model to NAIP tiles, converts each detected box
back into a real-world (lon, lat) using the tile bboxes in tile_metadata.csv,
deduplicates detections seen across overlapping tiles, and derives one corrected
coordinate per treatment plant.

Requires: ultralytics, torch, pandas, numpy, geopandas, shapely

Outputs (in data/inference/):
    detections.csv / .gpkg            - every unique detected object (one point each)
    corrected_coordinates.csv / .gpkg - one corrected coordinate per plant + data
"""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAVE_GEOPANDAS = True
except Exception:
    HAVE_GEOPANDAS = False

# ---------------------------------------------------------------------------
# Shared config from config.py:
#   C.RUNS_DIR, C.METADATA_CSV, C.RGB_DIR, C.CLASSES_FILE,
#   C.INFERENCE_DIR, C.IMAGE_PX, C.EXPORT_CRS
# (C.RUN_NAME is intentionally NOT used to locate weights -- see resolve_weights)
# ---------------------------------------------------------------------------

# Optional: measure how far each correction moved the reported point (and enable
# the "closest_to_original" strategy). Defaults to the round-2 sample's plant
# geometry. Set to None to skip. Requires geopandas.
ORIGINAL_COORDS_GPKG  = C.SAMPLE_GPKG
ORIGINAL_COORDS_LAYER = C.SAMPLE_LAYER_PLANTS

# Inference parameters (inference-specific tuning knobs)
CONF_THRESHOLD = 0.25    # detection confidence floor (tune upward if noisy)
IOU_THRESHOLD  = 0.50    # YOLO's built-in per-tile NMS IoU
BATCH_SIZE     = 32
INCLUDE_TRI    = False   # run on TRI hard-negative tiles too? (usually no)

# Geographic dedup: same-class detections within this distance = one object
NMS_DISTANCE_M = 12.0

# YOLO's backbone requires imgsz to be a multiple of its max stride (32).
# C.IMAGE_PX (333px) comes from tile geometry, not YOLO's constraint, so
# Ultralytics would otherwise silently round up to 352 and warn every call.
MODEL_IMGSZ = -(-C.IMAGE_PX // 32) * 32   # ceiling to nearest multiple of 32 -> 352

# Corrected-coordinate strategy:
#   "weighted_centroid"   - confidence-weighted centroid of all objects (default)
#   "highest_confidence"  - location of the single most confident object
#   "closest_to_original" - object nearest the reported coordinate (needs coords)
SELECTION_STRATEGY = "weighted_centroid"
# ---------------------------------------------------------------------------


# Optional: pin an exact weights file (e.g. to reproduce an old state's results
# on an old model for comparison). Leave as None for the normal case -- the
# script will always pick the most recently trained best.pt automatically.
# Do NOT rely on C.RUN_NAME here as a proxy for "current model": an old run's
# best.pt is never deleted, and RUN_NAME easily goes stale after a re-run
# (e.g. a class-filtered retrain) unless you remember to bump it every time.
EXPLICIT_WEIGHTS = None
# ---------------------------------------------------------------------------


def resolve_weights(explicit) -> Path:
    """Always resolves to the most recently modified best.pt under
    C.RUNS_DIR, unless EXPLICIT_WEIGHTS is set and points at a real file.
    Deliberately does NOT check a RUN_NAME-derived path first and stop there
    -- that pattern silently keeps resolving to a stale model forever once
    RUN_NAME goes out of sync with whatever you last trained, since the old
    run's best.pt still exists and 'exists' was being treated as 'current'."""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        print(f"  NOTE: EXPLICIT_WEIGHTS={p} not found; falling back to newest under {C.RUNS_DIR}")

    cands = sorted(C.RUNS_DIR.rglob("weights/best.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No trained weights found anywhere under {C.RUNS_DIR}.")

    chosen = cands[0]
    print(f"  Auto-selected newest weights: {chosen}")
    if len(cands) > 1:
        print(f"  ({len(cands)} best.pt files found under {C.RUNS_DIR}; also present, older:")
        for c in cands[1:4]:
            print(f"     {c}")
        if len(cands) > 4:
            print(f"     ... and {len(cands) - 4} more")
        print("  If this picked the wrong one, set EXPLICIT_WEIGHTS at the top of this script.)")
    return chosen


def load_class_names(model: YOLO) -> dict:
    if getattr(model, "names", None):
        return {int(k): v for k, v in model.names.items()}
    if C.CLASSES_FILE.exists():
        with open(C.CLASSES_FILE) as f:
            return dict(enumerate(ln.strip() for ln in f if ln.strip()))
    return dict(enumerate(C.CLASSES))


def is_tri_tile(row) -> bool:
    if str(row.get("source", "")).lower() == "tri":
        return True
    return str(row.get("tile_id", "")).upper().startswith("TRI_")


def resolve_rgb_path(row):
    tid = row.get("tile_id")
    if isinstance(tid, str):
        cand = C.RGB_DIR / f"{tid}_rgb.png"
        if cand.exists():
            return cand
    p = row.get("rgb_path")
    if isinstance(p, str) and Path(p).exists():
        return Path(p)
    return None


def pixel_to_lonlat(cx, cy, W, H, row):
    """
    Pixel center (cx, cy) -> (lon, lat) from the tile's WGS84 bbox.
    Y-axis flip: image row 0 is the TOP = NORTH edge (bbox_ymax), so latitude
    DECREASES as py increases.
    """
    xmin, ymin = row["bbox_xmin"], row["bbox_ymin"]
    xmax, ymax = row["bbox_xmax"], row["bbox_ymax"]
    return (xmin + (cx / W) * (xmax - xmin),
            ymax - (cy / H) * (ymax - ymin))


def meters_between(lon1, lat1, lon2, lat2):
    mlat = math.radians((lat1 + lat2) / 2.0)
    dx = (lon2 - lon1) * 111_320.0 * math.cos(mlat)
    dy = (lat2 - lat1) * 111_320.0
    return math.hypot(dx, dy)


def dedup_detections(dets: pd.DataFrame, dist_m: float) -> pd.DataFrame:
    """Greedy geographic NMS per plant and class; merge into weighted centroids."""
    kept = []
    for (cwns, cls_id), grp in dets.groupby(["CWNS_ID", "class_id"], sort=False):
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
                ) <= dist_m:
                    members.append(j)
                    used[j] = True
            m = grp.iloc[members]
            w = m["confidence"].to_numpy()
            kept.append({
                "CWNS_ID": cwns, "class_id": int(cls_id), "class": seed["class"],
                "lon": float(np.average(m["lon"], weights=w)),
                "lat": float(np.average(m["lat"], weights=w)),
                "confidence": float(m["confidence"].max()),
                "n_merged": int(len(members)),
                "tile_ids": "|".join(sorted(set(m["tile_id"]))),
            })
    return pd.DataFrame(kept)


def select_coordinate(objs: pd.DataFrame, strategy: str, orig=None):
    w = objs["confidence"].to_numpy()
    if strategy == "highest_confidence":
        top = objs.loc[objs["confidence"].idxmax()]
        return top["lon"], top["lat"], f"highest_conf({top['class']})"
    if strategy == "closest_to_original" and orig is not None:
        d = objs.apply(lambda r: meters_between(orig[0], orig[1], r["lon"], r["lat"]), axis=1)
        near = objs.loc[d.idxmin()]
        return near["lon"], near["lat"], f"closest_to_original({near['class']})"
    return (float(np.average(objs["lon"], weights=w)),
            float(np.average(objs["lat"], weights=w)), "weighted_centroid")


def load_original_coords():
    if not ORIGINAL_COORDS_GPKG or not HAVE_GEOPANDAS:
        return {}
    if not Path(ORIGINAL_COORDS_GPKG).exists():
        print(f"  NOTE: original coords not found: {ORIGINAL_COORDS_GPKG}")
        return {}
    g = gpd.read_file(ORIGINAL_COORDS_GPKG, layer=ORIGINAL_COORDS_LAYER).to_crs(C.EXPORT_CRS)
    out = {}
    for _, r in g.iterrows():
        if r.geometry is not None and not r.geometry.is_empty:
            c = r.geometry.centroid
            out[str(r["CWNS_ID"])] = (c.x, c.y)
    return out


def write_points(df, csv_path, gpkg_path, lon_col="lon", lat_col="lat", layer="points"):
    df.to_csv(csv_path, index=False)
    print(f"  Wrote {csv_path.name} ({len(df)} rows)")
    if HAVE_GEOPANDAS and len(df):
        gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry=[Point(xy) for xy in zip(df[lon_col], df[lat_col])],
            crs=C.EXPORT_CRS,
        )
        gdf.to_file(gpkg_path, layer=layer, driver="GPKG")
        print(f"  Wrote {gpkg_path.name} (layer '{layer}')")
    elif not HAVE_GEOPANDAS:
        print("  (geopandas missing — skipped .gpkg)")


def main():
    print("=== 05_run_inference.py ===\n")
    C.INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    weights = resolve_weights(EXPLICIT_WEIGHTS)
    print(f"Loading model: {weights}")
    model = YOLO(str(weights))
    class_names = load_class_names(model)
    print(f"Classes: {class_names}\n")

    if not C.METADATA_CSV.exists():
        print(f"ERROR: {C.METADATA_CSV} not found. Run 02_extract_tiles.py first.")
        return
    meta = pd.read_csv(C.METADATA_CSV, dtype={"CWNS_ID": str})
    print(f"Loaded metadata: {len(meta)} tiles")

    if not INCLUDE_TRI:
        meta = meta[~meta.apply(is_tri_tile, axis=1)]
    meta = meta.dropna(subset=["CWNS_ID", "bbox_xmin", "bbox_ymin",
                               "bbox_xmax", "bbox_ymax"]).copy()
    meta["_img"] = meta.apply(resolve_rgb_path, axis=1)
    missing = meta["_img"].isna().sum()
    if missing:
        print(f"  WARNING: {missing} tiles have no locatable RGB PNG — skipped")
    meta = meta[meta["_img"].notna()].reset_index(drop=True)
    if not len(meta):
        print("ERROR: no tiles to run on.")
        return
    print(f"  Inference on {len(meta)} tiles across {meta['CWNS_ID'].nunique()} plants\n")

    print("Running detection...")
    paths = [str(p) for p in meta["_img"].tolist()]
    det_rows = []
    results = model.predict(
        source=paths, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
        imgsz=MODEL_IMGSZ, device=device, batch=BATCH_SIZE,
        stream=True, verbose=False,
    )
    for idx, res in enumerate(results):
        row = meta.iloc[idx]
        H, W = res.orig_shape
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xywh = boxes.xywh.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cids = boxes.cls.cpu().numpy().astype(int)
        for (cx, cy, _w, _h), c, cid in zip(xywh, conf, cids):
            lon, lat = pixel_to_lonlat(cx, cy, W, H, row)
            det_rows.append({
                "tile_id": row["tile_id"], "CWNS_ID": row["CWNS_ID"],
                "class_id": int(cid), "class": class_names.get(int(cid), str(cid)),
                "confidence": float(c), "px": float(cx), "py": float(cy),
                "lon": float(lon), "lat": float(lat),
            })

    if not det_rows:
        print("\nNo detections above threshold. Try lowering CONF_THRESHOLD.")
        return
    dets = pd.DataFrame(det_rows)
    print(f"  {len(dets)} raw detections across {dets['CWNS_ID'].nunique()} plants\n")

    print(f"Deduplicating (per class, {NMS_DISTANCE_M:.0f} m)...")
    objects = dedup_detections(dets, NMS_DISTANCE_M)
    print(f"  {len(objects)} unique objects\n")

    originals = load_original_coords()
    print("Deriving corrected coordinates...")
    plant_rows = []
    for cwns, objs in objects.groupby("CWNS_ID", sort=False):
        orig = originals.get(str(cwns))
        strat = SELECTION_STRATEGY
        if strat == "closest_to_original" and not orig:
            strat = "weighted_centroid"
        lon, lat, note = select_coordinate(objs, strat, orig)
        cls_counts = objs["class"].value_counts()
        rec = {
            "CWNS_ID": cwns, "corrected_lon": lon, "corrected_lat": lat,
            "selection": note, "n_objects": int(len(objs)),
            "n_detections": int(objs["n_merged"].sum()),
            "dominant_class": cls_counts.index[0],
            "max_conf": float(objs["confidence"].max()),
            "mean_conf": float(objs["confidence"].mean()),
            "classes": "|".join(f"{k}:{v}" for k, v in cls_counts.items()),
        }
        if orig:
            rec["orig_lon"], rec["orig_lat"] = orig
            rec["offset_m"] = meters_between(orig[0], orig[1], lon, lat)
        plant_rows.append(rec)
    plants = pd.DataFrame(plant_rows)
    print(f"  Corrected {len(plants)} plants\n")

    print("Writing outputs...")
    write_points(objects, C.INFERENCE_DIR / "detections.csv",
                 C.INFERENCE_DIR / "detections.gpkg", layer="detections")
    write_points(plants, C.INFERENCE_DIR / "corrected_coordinates.csv",
                 C.INFERENCE_DIR / "corrected_coordinates.gpkg",
                 lon_col="corrected_lon", lat_col="corrected_lat", layer="plants")

    print("\n=== Summary ===")
    print(f"  Plants corrected : {len(plants)}")
    print(f"  Unique objects   : {len(objects)}")
    print(f"  Raw detections   : {len(dets)}")
    if "offset_m" in plants:
        print(f"  Median move      : {plants['offset_m'].median():.1f} m")
    for cls, n in objects["class"].value_counts().items():
        print(f"    {cls:<18}: {n}")
    print(f"\nOutputs in: {C.INFERENCE_DIR}")


if __name__ == "__main__":
    main()
