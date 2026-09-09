"""
01d_nlcd_topup.py
==================
Computes NLCD zonal stats for a SPECIFIC list of parcels and appends them to
01a's per-state output, rather than re-running 01a over a whole state.

Why this exists: 01a computes zonal stats only for parcels whose H3 res-9
cell falls inside the union of k-ring disks around the REPORTED points in a
state. A corrected (true) parcel outside every plant's disk therefore never
gets a row -- and since 02 PART 2 builds 10_parcel_features directly from
01a's output, 06's inner join silently drops that plant's positive row.
As of 2026-08-24 that cost 46 of 312 corrections (15%) -- non-randomly, the
large-error cases, which are exactly the ones the model most needs to learn.

Design decision -- WHERE this writes:
    Appends to  data/nlcd_features/nlcd_{STATE}_k{K}.parquet
    NOT to      data/features/10_parcel_features.parquet

    Appending at the 01a layer means the next 02 run derives everything else
    for these parcels through the normal code path: Regrid attributes, LBCS
    reclass, owner regex flags, has_ww_keyword, OSM wastewater tags, county
    data-quality aggregates. Appending NLCD columns straight into
    10_parcel_features would leave every one of those NULL, and a parcel with
    a null has_ww_keyword/lbcs_activity is worse than useless as a positive
    training row -- the model would learn "nulls mean right answer".

    So the sequence is:  01d  ->  re-run 02  ->  re-run 06  ->  re-run 07.

The original per-state parquet is backed up to *.pre-topup-{timestamp}.parquet
before any write, since this mutates an input that other stages already read.

Input: data/diagnostics/missing_parcels_topup.csv (written by 08), or an
explicit --uuids list. Run 08 first.

Usage:
    python 01d_nlcd_topup.py                        # from 08's CSV
    python 01d_nlcd_topup.py --dry-run              # report only, write nothing
    python 01d_nlcd_topup.py --uuids-file mine.csv  # custom state,ll_uuid CSV
"""
import argparse
import importlib.util
import shutil
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
from shapely import from_wkb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

# ---- Reuse 01a's zonal-stats implementation verbatim. Re-deriving the
# coverage-fraction / >=50% threshold logic here would risk these parcels
# getting subtly different NLCD semantics from every other parcel in the
# table -- which would show up as a spurious signal separating exactly the
# corrected parcels from everything else. ----
_spec = importlib.util.spec_from_file_location(
    "extract_lib", Path(__file__).resolve().parent / "01a_extract_parcels.py")
extract_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract_lib)

NLCD_SCHEMA = ["ll_uuid", "h3_index_9", "state", "total_pixels",
               "water_pixels", "has_water", "dominant_class", "dominant_count"]


def fetch_parcels_by_uuid(con, state: str, uuids: list[str],
                          chunk_size: int = 500) -> pd.DataFrame:
    """Same chunked-IN-list pattern as 01a's fetch_parcels_for_h3_cells,
    keyed on ll_uuid instead of h3_index_9."""
    frames = []
    for i in range(0, len(uuids), chunk_size):
        chunk = uuids[i:i + chunk_size]
        uuid_list = ", ".join("'" + str(u).replace("'", "''") + "'" for u in chunk)
        try:
            df = con.execute(f"""
                SELECT {C.PARCEL_ID_FIELD} AS ll_uuid, h3_index_9, state,
                       {C.PARCEL_WKB_FIELD} AS wkb_geometry
                FROM read_parquet('{C.PARCEL_BASE.as_posix()}/state={state}/*.parquet')
                WHERE {C.PARCEL_ID_FIELD} IN ({uuid_list})
            """).df()
            frames.append(df)
        except Exception as e:
            print(f"    [{state}] chunk fetch failed: {e}")
    if not frames:
        return pd.DataFrame(columns=["ll_uuid", "h3_index_9", "state", "wkb_geometry"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset="ll_uuid")


def process_state(con, state: str, uuids: list[str], k_rings: int,
                  dry_run: bool) -> dict:
    print(f"\n--- {state}: {len(uuids)} parcels requested ---")
    out_path = C.NLCD_OUTPUT_DIR / f"nlcd_{state}_k{k_rings}.parquet"
    if not out_path.exists():
        print(f"  ERROR: {out_path.name} does not exist -- run 01a for {state} "
              f"first. A top-up cannot bootstrap a state's output file.")
        return dict(state=state, requested=len(uuids), skipped=True)

    existing = pd.read_parquet(out_path)
    already = set(existing["ll_uuid"])
    todo = [u for u in uuids if u not in already]
    print(f"  Already present: {len(uuids) - len(todo)}  |  to compute: {len(todo)}")
    if not todo:
        return dict(state=state, requested=len(uuids), added=0, missing_geom=0)

    parcels = fetch_parcels_by_uuid(con, state, todo)
    n_missing_geom = len(todo) - len(parcels)
    if n_missing_geom:
        print(f"  WARNING: {n_missing_geom} ll_uuids not found in the Regrid store "
              f"for {state} -- these are genuine parcel-data gaps, not fixable here.")
    if not len(parcels):
        return dict(state=state, requested=len(uuids), added=0,
                    missing_geom=n_missing_geom)

    geoms = from_wkb([bytes(g) for g in parcels["wkb_geometry"].to_numpy()])
    parcel_sf = gpd.GeoDataFrame(
        parcels[["ll_uuid", "h3_index_9", "state"]], geometry=geoms, crs=C.EXPORT_CRS)

    print(f"  Extracting NLCD stats for {len(parcel_sf)} parcels...")
    import rasterio
    with rasterio.open(C.NLCD_PATH) as src:
        nlcd_crs = src.crs
    stats = extract_lib.extract_nlcd_stats(parcel_sf.to_crs(nlcd_crs))

    new_rows = pd.concat(
        [parcel_sf[["ll_uuid", "h3_index_9", "state"]].reset_index(drop=True),
         stats.reset_index(drop=True)], axis=1)
    new_rows["has_water"] = new_rows["water_pixels"] > 0
    new_rows = new_rows[NLCD_SCHEMA]

    n_zero = int(new_rows["total_pixels"].eq(0).sum())
    print(f"  Computed: {len(new_rows)} rows  "
          f"({int(new_rows['water_pixels'].gt(0).sum())} with water, {n_zero} with 0 pixels)")
    if n_zero:
        print(f"  NOTE: {n_zero} parcels got 0 NLCD pixels -- slivers below the "
              f"50% coverage threshold. They will carry null-ish land cover; "
              f"check them before trusting them as positives.")

    if dry_run:
        print("  --dry-run: not writing.")
        return dict(state=state, requested=len(uuids), added=len(new_rows),
                    missing_geom=n_missing_geom, dry_run=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = out_path.with_suffix(f".pre-topup-{stamp}.parquet")
    shutil.copy2(out_path, backup)
    print(f"  Backed up original -> {backup.name}")

    # Align dtypes to the existing file so the concat doesn't silently
    # widen an int column to float via NaN.
    for col in NLCD_SCHEMA:
        if col in existing.columns and col in new_rows.columns:
            try:
                new_rows[col] = new_rows[col].astype(existing[col].dtype)
            except (TypeError, ValueError):
                pass

    combined = pd.concat([existing, new_rows], ignore_index=True) \
                 .drop_duplicates(subset="ll_uuid", keep="first")
    combined.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"  Written: {out_path.name}  ({len(existing)} -> {len(combined)} rows)")

    return dict(state=state, requested=len(uuids), added=len(combined) - len(existing),
                missing_geom=n_missing_geom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uuids-file", type=str, default=None,
                    help="CSV with columns state,ll_uuid "
                         "(default: data/diagnostics/missing_parcels_topup.csv from 08)")
    ap.add_argument("--k-rings", type=int, default=C.K_RINGS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    C.ensure_dirs()
    print("=== 01d_nlcd_topup.py ===\n")

    src = Path(args.uuids_file) if args.uuids_file else \
        C.DATA_DIR / "diagnostics" / "missing_parcels_topup.csv"
    if not src.exists():
        print(f"ERROR: {src} not found. Run 08_diagnose_candidate_coverage.py first, "
              f"or pass --uuids-file.")
        return
    todo = pd.read_csv(src, dtype=str).dropna(subset=["state", "ll_uuid"]).drop_duplicates()
    print(f"Parcels to top up: {len(todo)} across {todo['state'].nunique()} states "
          f"(source: {src.name})\n")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; SET enable_geoparquet_conversion = false;")

    results = []
    for state, grp in todo.groupby("state"):
        results.append(process_state(con, state, grp["ll_uuid"].tolist(),
                                     args.k_rings, args.dry_run))
    con.close()

    summary = pd.DataFrame(results)
    print(f"\n=== Summary ===\n{summary.to_string(index=False)}")
    total_added = int(summary.get("added", pd.Series(dtype=int)).fillna(0).sum())
    print(f"\nParcels added: {total_added}")
    if total_added and not args.dry_run:
        print("\nNEXT STEPS (in order -- 01d alone changes nothing downstream):")
        print("  1. python 02_feature_engineering.py     # rebuilds 10_parcel_features")
        print("  2. python 06_build_stage2b_training.py  # should now report fewer drops")
        print("  3. python 07_train_stage2b.py")
        print("\nRe-run 08 afterwards: ok_usable should have risen by roughly the "
              "number above, and anything still missing is a real parcel-data gap.")

    print("\n=== complete ===")


if __name__ == "__main__":
    main()
