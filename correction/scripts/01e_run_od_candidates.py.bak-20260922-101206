"""
01e_run_od_candidates.py
========================
Runs object detection on Stage 2a's TOP-K CANDIDATE PARCELS, so Stage 2b can
re-rank them. This is the step 05_run_inference.py's docstring defers as
"the not-yet-written OD-on-arbitrary-top-K step".

WHY THIS IS WORTH BUILDING NOW
    Stage 2b's measured OD importance is near zero, which looks like evidence
    that OD does not matter. It is not evidence about the question that
    matters, because of what Stage 2b was trained on: 06_build_stage2b_
    training.py pairs the REPORTED parcel (01b, label 0) against the
    CORRECTED parcel (01c, label 1). Those two are usually nothing alike --
    a residential lot versus a 40-acre municipal parcel on a river. Parcel
    attributes separate them trivially, so OD has no residual variance left
    to explain and its importance collapses.

    The deployment question is different: among the top ~20 candidates
    Stage 2a ALREADY ranked highly, every one is a plausible municipal
    parcel of roughly the right size and land cover. There the parcel
    features are near-tied and OD is the only feature family that can look
    at the actual infrastructure. Nothing in the current training data
    speaks to that case. This script generates it.

WHAT IT DOES
    Reads data/inference/stage2_candidates.parquet, resolves each candidate
    parcel's geometry from the Regrid mirror, and runs 01b's detection
    machinery on it -- same tiling, same geographic NMS, same in-parcel
    flagging, same select_plant_features() contract that 02 and 06 already
    consume.

    Output rows are keyed on (CWNS_ID, ll_uuid), not CWNS_ID alone: one
    plant now has up to --top-k candidate parcels rather than one reported
    location.

REUSE, NOT REIMPLEMENTATION
    01b's prepare_plant/run_batch/finalize_plant treat their plant key as an
    opaque string, so this passes "{CWNS_ID}::{ll_uuid}" and splits it back
    out afterwards. Same importlib pattern 01c already uses. Nothing about
    01b's detection behaviour is duplicated or altered here -- if it changes
    there, it changes here.

    One real difference from 01b: the NAIP item is resolved from the
    CANDIDATE PARCEL's centroid, not the plant's reported point. A candidate
    can sit kilometres from the reported location (that is the entire point
    of Stage 2a), and resolving the item from the reported point would fetch
    the wrong quad and fail every tile.

SCOPE IT SMALL FIRST
    --holdout-only (the default) restricts to the frozen holdout's
    corrections bin -- 50 plants x up to 20 candidates is about 1,000 tiles,
    comparable to 01c's national run and a few minutes of wall clock. Those
    plants have known truth and were excluded from every model, so they are
    the honest test of whether re-ranking with OD moves recall@1 at all.
    Only widen to --all once that answers yes.

Usage:
    python 01e_run_od_candidates.py --dry-run
    python 01e_run_od_candidates.py                       # holdout corrections
    python 01e_run_od_candidates.py --holdout-bins corrections,correct
    python 01e_run_od_candidates.py --all --states OH,MS  # everything, scoped
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
from shapely import from_wkb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

# ---- 01b's machinery, imported not copied (filename starts with a digit,
# so a normal import is a SyntaxError). Same pattern as 01c. ----
_spec = importlib.util.spec_from_file_location(
    "od_lib", Path(__file__).resolve().parent / "01b_run_object_detection.py")
od_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(od_lib)

KEY_SEP = "::"


def load_candidates(top_k: int) -> pd.DataFrame:
    path = C.DATA_DIR / "inference" / "stage2_candidates.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found.\n"
              f"  Run 05_run_inference.py (or the array + merge_05_shards.py) first.")
        sys.exit(2)
    df = pd.read_parquet(path, columns=["CWNS_ID", "ll_uuid", "STATE_CODE",
                                        "stage2_prob_correct"])
    df["CWNS_ID"] = df["CWNS_ID"].astype(str)
    df = df.sort_values(["CWNS_ID", "stage2_prob_correct"], ascending=[True, False])
    return df.groupby("CWNS_ID", group_keys=False).head(top_k).reset_index(drop=True)


def restrict_to_holdout(cands: pd.DataFrame, bins: list[str]) -> pd.DataFrame:
    """Keep only candidates belonging to holdout plants in the named bins.

    The corrections bin is the headline slice: known-misplaced plants with a
    known true parcel, excluded from every model by the anti-joins. Recall@1
    on those is the number this whole experiment exists to move."""
    from holdout import MANIFEST_PATH
    if not MANIFEST_PATH.exists():
        print(f"ERROR: {MANIFEST_PATH} not found. Run 09_build_holdout.py first, "
              f"or pass --all.")
        sys.exit(2)
    man = pd.read_parquet(MANIFEST_PATH)
    man["CWNS_ID"] = man["CWNS_ID"].astype(str)
    keep = man[man["bin"].isin(bins)]
    print(f"  Holdout manifest: {len(man)} plants; "
          f"{len(keep)} in bin(s) {bins}")
    out = cands[cands["CWNS_ID"].isin(set(keep["CWNS_ID"]))].copy()
    print(f"  Candidates for those plants: {len(out)} "
          f"across {out['CWNS_ID'].nunique()} plant(s)")
    return out


def load_parcel_geoms(con, state: str, uuids: list[str]) -> pd.DataFrame:
    """Candidate parcel geometries by ll_uuid, projected to PROJECTED_CRS.

    Direct ST_Intersects-free lookup by id -- these uuids came out of Stage
    2a, which sourced them from this same mirror, so an id miss means the
    Regrid vintage moved under us rather than a geometry problem."""
    con.register("want", pd.DataFrame({"ll_uuid": uuids}))
    try:
        res = con.execute(f"""
            SELECT p.{C.PARCEL_ID_FIELD} AS ll_uuid,
                   p.{C.PARCEL_WKB_FIELD} AS parcel_wkb
            FROM read_parquet('{C.PARCEL_BASE.as_posix()}/state={state}/*.parquet') p
            JOIN want ON want.ll_uuid = p.{C.PARCEL_ID_FIELD}
        """).df()
    except Exception as e:
        print(f"    [{state}] parcel geometry lookup failed: {str(e)[:200]}")
        return pd.DataFrame(columns=["ll_uuid", "parcel_wkb"])
    finally:
        con.unregister("want")
    return res.drop_duplicates(subset="ll_uuid")


def processed_keys(out_root: Path, state: str) -> set:
    """(CWNS_ID, ll_uuid) pairs already written, for resume."""
    d = out_root / "candidates" / f"state={state}"
    if not d.exists():
        return set()
    done = set()
    for f in d.glob("*.parquet"):
        try:
            df = pd.read_parquet(f, columns=["CWNS_ID", "ll_uuid"])
            done |= set(zip(df["CWNS_ID"].astype(str), df["ll_uuid"].astype(str)))
        except Exception:
            continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=20,
                    help="candidates per plant to run OD on (default 20, the "
                         "same cap 05 stores)")
    ap.add_argument("--all", action="store_true",
                    help="every plant in stage2_candidates, not just holdout. "
                         "This is the national job -- ~240k tiles. Do the "
                         "holdout first.")
    ap.add_argument("--holdout-bins", default="corrections",
                    help="comma-separated holdout bins (default: corrections)")
    ap.add_argument("--states", default=None, help="comma-separated filter")
    ap.add_argument("--batch-size", type=int, default=200,
                    help="candidate parcels prepared per batch")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--nms-dist", type=float, default=15.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the work without fetching any imagery")
    args = ap.parse_args()

    out_root = C.DATA_DIR / "od_features_candidates"
    print("=== 01e_run_od_candidates.py ===")
    print(f"output: {out_root}")

    print("\nLoading Stage 2a candidates...")
    cands = load_candidates(args.top_k)
    print(f"  {len(cands)} candidate(s) across {cands['CWNS_ID'].nunique()} plant(s) "
          f"(top {args.top_k} each)")

    if not args.all:
        bins = [b.strip() for b in args.holdout_bins.split(",")]
        cands = restrict_to_holdout(cands, bins)
    else:
        print("  --all: no holdout restriction")

    if args.states:
        keep = [s.strip() for s in args.states.split(",")]
        cands = cands[cands["STATE_CODE"].isin(keep)]
        print(f"  Restricted to {keep}: {len(cands)} candidate(s)")

    if args.limit:
        cands = cands.head(args.limit)
        print(f"  --limit: {len(cands)} candidate(s)")

    if cands.empty:
        print("\nNothing to do.")
        return

    print("\n--- work plan ---")
    for st, grp in cands.groupby("STATE_CODE"):
        print(f"  {st}: {len(grp)} candidate parcel(s), "
              f"{grp['CWNS_ID'].nunique()} plant(s)")
    print(f"  TOTAL: {len(cands)} candidate parcel(s)")
    print(f"  Each is tiled by 01b's generate_tile_grid -- most parcels are one "
          f"tile, large ones more.")

    if args.dry_run:
        print("\n--dry-run: no imagery fetched.")
        return

    weights = od_lib.resolve_weights(args.weights, C.OD_MODEL_DIR)
    print(f"\nWeights: {weights}")
    from ultralytics import YOLO
    model = YOLO(str(weights))
    class_names = {int(k): v for k, v in model.names.items()}
    print(f"Classes: {class_names}")
    unknown = set(class_names.values()) - set(C.CLASSES)
    if unknown:
        print(f"  WARNING: model emits {sorted(unknown)}, absent from C.CLASSES. "
              f"select_plant_features iterates C.CLASSES, so those detections "
              f"land in od_n_objects with no per-class feature.")

    print("\nConnecting to Planetary Computer STAC API...")
    catalog = od_lib.pystac_client.Client.open(
        od_lib.STAC_URL, modifier=od_lib.planetary_computer.sign_inplace) \
        if hasattr(od_lib, "STAC_URL") else None
    if catalog is None:
        import planetary_computer
        import pystac_client
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace)
    print("Connected.")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; SET enable_geoparquet_conversion = false;")

    t_start = time.time()
    n_done = n_skipped = n_nogeom = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for state, state_cands in cands.groupby("STATE_CODE"):
            print(f"\n--- State {state}: {len(state_cands)} candidate(s) ---")

            done = set() if args.no_resume else processed_keys(out_root, state)
            if done:
                before = len(state_cands)
                state_cands = state_cands[[
                    (c, u) not in done for c, u in
                    zip(state_cands["CWNS_ID"], state_cands["ll_uuid"])]]
                n_skipped += before - len(state_cands)
                print(f"  Resume: skipping {before - len(state_cands)} already done")
            if state_cands.empty:
                continue

            geoms = load_parcel_geoms(con, state, state_cands["ll_uuid"].tolist())
            merged = state_cands.merge(geoms, on="ll_uuid", how="left")
            missing = merged["parcel_wkb"].isna().sum()
            if missing:
                print(f"  {missing} candidate(s) have no geometry in the current "
                      f"Regrid vintage -- skipped")
                n_nogeom += int(missing)
                merged = merged[merged["parcel_wkb"].notna()]

            rows, objs = [], []
            for start in range(0, len(merged), args.batch_size):
                batch = merged.iloc[start:start + args.batch_size]
                tasks = {}
                for _, r in batch.iterrows():
                    pgeom_4326 = from_wkb(bytes(r["parcel_wkb"]))
                    pgeom_5070 = gpd.GeoSeries([pgeom_4326], crs=C.EXPORT_CRS) \
                        .to_crs(C.PROJECTED_CRS).iloc[0]
                    # NAIP item from the CANDIDATE's own centroid -- a candidate
                    # can be kilometres from the reported point, and resolving
                    # from the plant's reported location would fetch the wrong
                    # quad and fail every tile.
                    ctr = pgeom_4326.centroid
                    key = f"{r['CWNS_ID']}{KEY_SEP}{r['ll_uuid']}"
                    try:
                        tasks[key] = od_lib.prepare_plant(
                            key, pgeom_5070, float(ctr.x), float(ctr.y), catalog)
                    except Exception as e:
                        print(f"    {key}: prepare failed -- {str(e)[:120]}")

                if not tasks:
                    continue
                od_lib.run_batch(tasks, state, model, class_names, args,
                                 executor, catalog)
                for key, task in tasks.items():
                    if task.get("item") is None:
                        continue
                    row = od_lib.finalize_plant(task, state)
                    cwns, uuid = key.split(KEY_SEP, 1)
                    row["CWNS_ID"], row["ll_uuid"] = cwns, uuid
                    rows.append(row)
                    for o in task.get("objects", []):
                        o["CWNS_ID"], o["ll_uuid"] = cwns, uuid
                        objs.append(o)

                n_done += len(tasks)
                print(f"    ...{n_done} candidate(s) done "
                      f"({(time.time() - t_start) / 60:.1f} min elapsed)")

            if rows:
                od_lib.flush_parquet(rows, out_root, "candidates", state)
                if objs:
                    od_lib.flush_parquet(objs, out_root, "objects", state)
                print(f"  Flushed {len(rows)} candidate row(s)")

    con.close()

    print("\n=== Summary ===")
    print(f"  Candidates processed : {n_done}")
    print(f"  Skipped (resume)     : {n_skipped}")
    print(f"  No parcel geometry   : {n_nogeom}")
    print(f"  Elapsed              : {(time.time() - t_start) / 60:.1f} min")
    print(f"\nOutputs in: {out_root}")
    print("Next: score these with stage2b_rf_model.joblib and compare rank-1 "
          "accuracy against Stage 2a alone, on the holdout's known truth.")


if __name__ == "__main__":
    main()
