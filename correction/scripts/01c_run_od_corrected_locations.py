"""
01c_run_od_corrected_locations.py
====================================
Runs object detection on the CORRECTED (true) location for every Stage 2
training plant -- a small, targeted companion to 01b_run_object_detection.py,
which only ever ran OD against each plant's REPORTED (often wrong) location.

Why this exists (design settled 2026-08-21): Stage 2b, the not-yet-built
OD-aware final-candidate-selection model, needs a contrastive training pair
per corrections-bin plant:
  - OD features for the WRONG parcel -- already exist, from 01b's normal
    run using LATITUDE/LONGITUDE (the reported location).
  - OD features for the RIGHT parcel -- this script, using
    Corrected_X/Corrected_Y instead.

Reuses ALL of 01b_run_object_detection.py's tile/fetch/NMS/parcel-cap
machinery via direct import from that file (importlib, since a filename
starting with a digit isn't a valid Python module name for a normal `import`
statement). Duplicating that logic here instead would risk it drifting out
of sync with 01b's already-hard-won bug fixes (retry misclassification,
multi-quad fallback, parcel bbox capping, real-polygon tile filtering, etc.)
-- all of that applies exactly as much to a corrected-location parcel as a
reported one, so there is no reason to re-derive it.

Output: same four-table schema as 01b (tiles/detections/objects/plants),
written to config.py's OD_OUTPUT_DIR_CORRECTED -- a SEPARATE root from
OD_OUTPUT_DIR, so no existing reader of the reported-location OD output
ever accidentally unions reported and corrected-location rows for the same
CWNS_ID together.

Usage:
    python 01c_run_od_corrected_locations.py --workers 32
    python 01c_run_od_corrected_locations.py --limit 10 --no-resume
"""
import argparse
import importlib.util
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from shapely import from_wkb, Point
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

# ---- Dynamically import 01b's machinery (its filename starts with a digit,
# so a normal `import 01b_run_object_detection` is a SyntaxError) ----
_spec = importlib.util.spec_from_file_location(
    "od_lib", Path(__file__).resolve().parent / "01b_run_object_detection.py")
od_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(od_lib)   # __name__ != "__main__" here, so od_lib's
                                     # own main() is NOT triggered by this import


def load_correction_plants(limit: int = None) -> pd.DataFrame:
    """CWNS_ID + STATE_CODE + LONGITUDE/LATITUDE (renamed from
    Corrected_X/Corrected_Y) for every plant in the 'corrections' training
    layer -- these are the known-right-answer locations that need OD run on
    them. Deliberately reuses the exact column names (LONGITUDE, LATITUDE,
    STATE_CODE, CWNS_ID) that od_lib.find_containing_parcels() and
    prepare_plant() already expect, so this plant list can be fed straight
    into 01b's existing per-state loop machinery unmodified."""
    corrections = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CORRECTIONS)
    corrections["CWNS_ID"] = corrections["CWNS_ID"].astype(str)
    df = corrections[["CWNS_ID", "Corrected_X", "Corrected_Y"]].dropna(
        subset=["Corrected_X", "Corrected_Y"])
    df = df.rename(columns={"Corrected_X": "LONGITUDE", "Corrected_Y": "LATITUDE"})

    # corrections layer doesn't carry STATE_CODE itself (see
    # build_training_bins.py) -- pull it from CWNS.
    loc = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt", dtype={"CWNS_ID": str}, encoding="latin1")
    loc = loc[["CWNS_ID", "STATE_CODE"]].drop_duplicates(subset="CWNS_ID")
    df = df.merge(loc, on="CWNS_ID", how="left")

    n_before = len(df)
    df = df.dropna(subset=["STATE_CODE"])
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} corrections with no resolvable STATE_CODE")

    df = df.drop_duplicates(subset="CWNS_ID").reset_index(drop=True)
    if limit:
        df = df.head(limit)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                     help="process only the first N corrections (testing)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--models-dir", default=str(C.OD_MODEL_DIR))
    ap.add_argument("--conf", type=float, default=C.CONF_THRESHOLD)
    ap.add_argument("--iou", type=float, default=C.IOU_THRESHOLD)
    ap.add_argument("--nms-dist", type=float, default=C.NMS_DISTANCE_M)
    ap.add_argument("--workers", type=int, default=C.NAIP_WORKERS,
                     help="concurrent NAIP fetch threads.")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    C.ensure_dirs()
    out_root = C.OD_OUTPUT_DIR_CORRECTED

    print("=== 01c_run_od_corrected_locations.py ===")
    print(f"Device: {args.device}  |  fetch workers: {args.workers}  |  "
          f"output: {out_root}")

    weights_path = od_lib.resolve_weights(args.weights, Path(args.models_dir))
    model = od_lib.YOLO(str(weights_path))
    class_names = {int(k): v for k, v in model.names.items()}
    print(f"Model: {weights_path.name}  |  Classes: {class_names}\n")

    print("Connecting to Planetary Computer STAC API...")
    catalog = pystac_client.Client.open(C.STAC_URL, modifier=planetary_computer.sign_inplace)
    print("Connected.\n")

    plants = load_correction_plants(limit=args.limit)
    print(f"Correction (true-location) plants loaded: {len(plants)}")

    if not args.no_resume:
        all_done = set()
        for state_dir in (out_root / "plants").glob("state=*"):
            all_done |= od_lib.load_processed_ids(state_dir)
        if all_done:
            before = len(plants)
            plants = plants[~plants["CWNS_ID"].isin(all_done)]
            print(f"Resuming: skipping {before - len(plants)} already-processed corrections")

    if len(plants) == 0:
        print("Nothing to do.")
        return

    totals = dict(plants=0, kept=0, deleted=0, failed=0, no_parcel=0,
                  parcels_capped=0, fetch_time=0.0, infer_time=0.0)
    t_start = time.time()

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; SET enable_geoparquet_conversion = false;")

    with rasterio.Env(**od_lib.GDAL_ENV):
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for state_abbr, state_plants in plants.groupby("STATE_CODE"):
                print(f"\n--- State {state_abbr}: {len(state_plants)} corrections ---")

                plants_state_dir = out_root / "plants" / f"state={state_abbr}"
                state_plants_todo = state_plants
                if not args.no_resume:
                    done_ids = od_lib.load_processed_ids(plants_state_dir)
                    if done_ids:
                        before = len(state_plants_todo)
                        state_plants_todo = state_plants_todo[
                            ~state_plants_todo["CWNS_ID"].isin(done_ids)]
                        print(f"  Resuming: skipping {before - len(state_plants_todo)} "
                              f"already-processed corrections in this state")

                if len(state_plants_todo) == 0:
                    print("  Nothing to do for this state.")
                    continue

                parcel_matches = od_lib.find_containing_parcels(con, state_abbr, state_plants_todo)
                matched_ids = set(parcel_matches["CWNS_ID"])
                unmatched = len(state_plants_todo) - len(matched_ids)
                if unmatched:
                    print(f"  No containing parcel found for {unmatched} corrected "
                          f"locations -- skipped")
                    totals["no_parcel"] += unmatched

                n_before = len(parcel_matches)
                parcel_matches = parcel_matches.sort_values("ll_uuid").drop_duplicates(
                    subset="CWNS_ID", keep="first")
                if len(parcel_matches) < n_before:
                    print(f"  {n_before - len(parcel_matches)} corrections matched >1 "
                          f"parcel (boundary edge case) -- kept one match each")

                merged = state_plants_todo.merge(parcel_matches, on="CWNS_ID", how="inner")

                tiles_buf, dets_buf, objs_buf, plants_buf = [], [], [], []
                FLUSH_EVERY = 200

                def _flush_all():
                    od_lib.flush_parquet(tiles_buf, out_root, "tiles", state_abbr)
                    od_lib.flush_parquet(dets_buf, out_root, "detections", state_abbr)
                    od_lib.flush_parquet(objs_buf, out_root, "objects", state_abbr)
                    od_lib.flush_parquet(plants_buf, out_root, "plants", state_abbr)

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
                            tasks[cwns_id] = od_lib.prepare_plant(
                                cwns_id, pgeom_5070,
                                float(row["LONGITUDE"]), float(row["LATITUDE"]), catalog)
                        except Exception as e:
                            print(f"    WARNING: prepare_plant failed for {cwns_id} "
                                  f"after retries ({e}) -- skipping, will retry on next run")
                            totals["failed"] += 1
                            continue

                        if not area_exceeded and len(tasks[cwns_id]["grid"]) > C.MAX_TILES_PER_PLANT:
                            cap_reason = "tile_count"
                            parcel_capped = True
                            totals["parcels_capped"] += 1
                            print(f"    {cwns_id}: {len(tasks[cwns_id]['grid'])} tiles even "
                                  f"after real-polygon filtering -- exceeds "
                                  f"MAX_TILES_PER_PLANT={C.MAX_TILES_PER_PLANT}, falling back "
                                  f"to point footprint")
                            pt_5070 = gpd.GeoSeries(
                                [Point(float(row["LONGITUDE"]), float(row["LATITUDE"]))],
                                crs=C.EXPORT_CRS
                            ).to_crs(C.PROJECTED_CRS).iloc[0]
                            hw = C.CAPPED_PARCEL_FALLBACK_HALFWIDTH_M
                            fallback_geom = box(pt_5070.x - hw, pt_5070.y - hw,
                                                 pt_5070.x + hw, pt_5070.y + hw)
                            tasks[cwns_id] = od_lib.prepare_plant(
                                cwns_id, fallback_geom,
                                float(row["LONGITUDE"]), float(row["LATITUDE"]), catalog)

                        area_by_id[cwns_id] = (float(parcel_area_m2), parcel_capped,
                                                float(bbox_max_dim_m), cap_reason,
                                                len(tasks[cwns_id]["grid"]))

                    od_lib.run_batch(tasks, state_abbr, model, class_names, args, executor, catalog)

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
                    print(f"    ...{totals['plants']} corrections done "
                          f"({elapsed/60:.1f} min elapsed, batch of {len(batch)})")

                    _flush_all()
                    print(f"    flushed {len(plants_buf)} corrections (resume point advanced)")
                    tiles_buf, dets_buf, objs_buf, plants_buf = [], [], [], []

                print(f"  State {state_abbr} done")

    con.close()

    elapsed = time.time() - t_start
    print("\n=== Summary ===")
    print(f"  Corrections processed : {totals['plants']}")
    print(f"  No parcel found        : {totals['no_parcel']}")
    print(f"  Parcels capped         : {totals['parcels_capped']}")
    print(f"  Tiles w/ detections    : {totals['kept']}")
    print(f"  Tiles empty            : {totals['deleted']}")
    print(f"  Fetch failures         : {totals['failed']}")
    print(f"  Elapsed                : {elapsed/60:.1f} min")
    print(f"\nOutputs (Parquet, partitioned by state) in: {out_root}")


if __name__ == "__main__":
    main()
