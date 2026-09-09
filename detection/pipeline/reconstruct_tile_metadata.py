"""
reconstruct_tile_metadata.py
==============================
tile_metadata.csv only covers round 2 (48 plants). The 264 existing labels
in annotation/ls_export/labels/ are almost entirely round 1 (71 plants, only
1 overlapping) -- confirmed via diagnose_label_matching.py, 2026-08-28.
Round 1's own metadata file could not be located.

This reconstructs equivalent metadata rows from first principles, since
tile geometry is fully deterministic: same parcel geometry in, same
generate_tile_grid()/tiles_to_wgs84() algorithm, same result. Reuses those
two functions directly from 02_extract_tiles.py via importlib rather than
reimplementing them, so there's no risk of subtly drifting from the
original tiling logic.

For each label file:
  1. Parse (CWNS_ID, ll_uuid, row, col) from the filename via
     03_prepare_dataset.py's parse_tile_stem() (also reused, not reimplemented).
  2. Derive state from CWNS_ID's 2-digit FIPS prefix.
  3. Look up that parcel's real geometry from the Regrid store (glob across
     the whole state's partition, since geoid/county isn't recoverable from
     the filename alone -- fine at this scale, ~264 lookups, not a
     nationwide job).
  4. Regenerate the tile grid for that parcel and pick out the specific
     (row, col) tile the filename refers to.
  5. Write a metadata row with real bbox_xmin/ymin/xmax/ymax, matching
     tile_metadata.csv's own schema exactly.

Also copies each label file to a CLEAN filename ({tile_id}_rgb.txt) in the
output folder -- so convert_tiles_to_500m.py needs no further changes; it
already expects clean filenames, which is what label_app.R writes for new
labels. The mangled-filename handling stays contained to this one script.

CAVEAT: if Regrid's parcel data has changed since the original extraction
(re-survey, boundary correction, a parcel split/merged), the regenerated
grid could differ from what was actually tiled and shown to you at
labeling time. Flagged per-parcel below if a lookup fails outright; not
detectable if the parcel still exists but its boundary shifted slightly.

Usage:
    python reconstruct_tile_metadata.py --out reconstructed
"""
import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
from shapely import wkt as shapely_wkt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

# Standard 2-digit FIPS -> USPS state code. CWNS_ID's first 2 digits are
# this code (confirmed directly: CWNS 29001011003 has st=MO in the existing
# metadata, and 29 is Missouri's real FIPS code).
FIPS_TO_STATE = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY", "60": "AS", "66": "GU", "69": "MP",
    "72": "PR", "78": "VI",
}

# Reuse 02_extract_tiles.py's own tiling functions -- deterministic, and
# reimplementing them risks a subtle mismatch from the real thing.
_spec02 = importlib.util.spec_from_file_location(
    "extract", Path(__file__).resolve().parent / "02_extract_tiles.py")
extract = importlib.util.module_from_spec(_spec02)
_spec02.loader.exec_module(extract)

# Reuse 03_prepare_dataset.py's filename parser too.
_spec03 = importlib.util.spec_from_file_location(
    "prep", Path(__file__).resolve().parent / "03_prepare_dataset.py")
prep = importlib.util.module_from_spec(_spec03)
_spec03.loader.exec_module(prep)


def get_parcel_geometry(con, state: str, ll_uuid: str):
    """Globs the whole state's Regrid partition -- geoid isn't recoverable
    from the label filename alone, and 264 lookups at this scale is fine
    without narrowing to a specific county file.

    Decodes WKB -> WKT INSIDE DuckDB via ST_AsText(ST_GeomFromWKB(...)),
    not client-side in Python. Every other script in this project that reads
    this column (parcels.py, 01a_extract_parcels.py, 02_feature_engineering.py)
    wraps it in ST_GeomFromWKB() before it ever leaves SQL -- none of them
    pull the raw BLOB into pandas and decode it with shapely.from_wkb()
    client-side, which is what an earlier version of this function did.
    Confirmed 2026-08-28: that approach failed on ALL 264 real parcels
    across multiple different states with 'Expected bytes or string, got
    int' -- too uniform a failure to be a per-row data anomaly, and it
    diverges from the one pattern already proven to work everywhere else in
    this codebase. WKT text is unambiguous coming back through duckdb's
    pandas conversion (always a plain str), unlike a raw BLOB column."""
    glob_pattern = f"{C.PARCEL_BASE.as_posix()}/state={state}/*.parquet"
    try:
        df = con.execute(f"""
            SELECT ST_AsText(ST_GeomFromWKB({C.PARCEL_WKB_FIELD})) AS wkt
            FROM read_parquet('{glob_pattern}')
            WHERE {C.PARCEL_ID_FIELD} = '{ll_uuid}'
            LIMIT 1
        """).df()
    except Exception as e:
        print(f"    Regrid lookup failed for state={state}: {e}")
        return None
    if df.empty:
        return None

    raw = df["wkt"].iloc[0]
    if not isinstance(raw, str) or not raw:
        print(f"    WARNING: ll_uuid {ll_uuid} in state={state} returned "
              f"{type(raw).__name__} ({raw!r}) instead of WKT text -- skipping "
              f"this parcel.")
        return None

    try:
        return shapely_wkt.loads(raw)
    except Exception as e:
        print(f"    WARNING: ll_uuid {ll_uuid} in state={state} -- WKT parse "
              f"failed ({e}) -- skipping this parcel.")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="reconstructed",
                     help="output folder (relative to this script) -- gets "
                          "data/tile_metadata.csv and annotation/ls_export/labels/ "
                          "with clean filenames, ready for convert_tiles_to_500m.py "
                          "--source-root")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_metadata = out_root / "data" / "tile_metadata.csv"
    out_labels = out_root / "annotation" / "ls_export" / "labels"
    out_metadata.parent.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    print("=== reconstruct_tile_metadata.py ===")

    label_files = sorted((C.ANNOTATION_DIR / "labels").glob("*.txt"))
    print(f"Found {len(label_files)} label file(s) to reconstruct metadata for")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; SET enable_geoparquet_conversion = false;")

    geom_cache = {}  # (state, ll_uuid) -> shapely geometry, avoid re-querying
    rows = []
    n_ok, n_no_state, n_no_parcel, n_grid_mismatch, n_unexpected = 0, 0, 0, 0, 0

    for f in label_files:
      try:
        parsed = prep.parse_tile_stem(f.name)
        if parsed is None:
            print(f"  {f.name}: could not parse filename -- skipped")
            continue
        cwns_id, tile_stem = parsed
        # tile_stem = {cwns_id}_{ll_uuid}_r{row}_c{col} -- slice by known
        # length rather than a naive split("_"), since ll_uuid itself
        # contains no underscores but this keeps the logic explicit either way
        parts = tile_stem.split("_")
        ll_uuid = parts[1]
        rc = tile_stem[len(cwns_id) + 1 + len(ll_uuid) + 1:]  # "r01_c01"
        row_str, col_str = rc.split("_")
        target_row, target_col = int(row_str[1:]), int(col_str[1:])

        fips = cwns_id[:2]
        state = FIPS_TO_STATE.get(fips)
        if state is None:
            n_no_state += 1
            print(f"  {f.name}: CWNS {cwns_id} -- unrecognized FIPS prefix '{fips}', skipped")
            continue

        cache_key = (state, ll_uuid)
        if cache_key not in geom_cache:
            geom_cache[cache_key] = get_parcel_geometry(con, state, ll_uuid)
        geom = geom_cache[cache_key]
        if geom is None:
            n_no_parcel += 1
            print(f"  {f.name}: CWNS {cwns_id}, ll_uuid {ll_uuid} not found in "
                  f"Regrid state={state} -- skipped (parcel store may have changed "
                  f"since original extraction)")
            continue

        parcel_gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs=C.EXPORT_CRS).to_crs(C.PROJECTED_CRS)
        grid = extract.generate_tile_grid(parcel_gdf.total_bounds)
        match = [t for t in grid if t["row"] == target_row and t["col"] == target_col]
        if not match:
            n_grid_mismatch += 1
            print(f"  {f.name}: regenerated grid has no r{target_row:02d}_c{target_col:02d} "
                  f"(grid has {len(grid)} tile(s)) -- parcel geometry may have changed "
                  f"since original extraction, skipped")
            continue

        bboxes, centers = extract.tiles_to_wgs84(match)
        bbox = bboxes[0]
        ctr = centers[0]
        new_tile_id = tile_stem

        rows.append({
            "tile_id": new_tile_id, "source": "reconstructed", "CWNS_ID": cwns_id,
            "TRI_FACILITY_ID": "", "ll_uuid_primary": ll_uuid, "ll_uuid_alternates": "",
            "st": state, "geoid": "", "tile_row": target_row, "tile_col": target_col,
            "ctr_lon": ctr[0], "ctr_lat": ctr[1],
            "bbox_xmin": bbox[0], "bbox_ymin": bbox[1], "bbox_xmax": bbox[2], "bbox_ymax": bbox[3],
            "source_res_m": "", "target_res_m": C.TARGET_RES_M, "image_px": C.IMAGE_PX,
            "acq_date": "", "rgb_path": "", "ndwi_path": "", "label": "",
        })

        shutil.copy2(f, out_labels / f"{new_tile_id}_rgb.txt")
        n_ok += 1

      except Exception as e:
        # Two different unexpected failure classes already hit real rows in
        # this dataset (duckdb extension version mismatch, a glob
        # schema-unification issue returning an int instead of a WKB blob) --
        # a third is plausible across the remaining files. One bad row must
        # not kill the whole batch a third time.
        n_unexpected += 1
        print(f"  {f.name}: UNEXPECTED ERROR ({type(e).__name__}: {e}) -- skipped")

    con.close()

    pd.DataFrame(rows).reindex(columns=C.METADATA_COLUMNS).to_csv(out_metadata, index=False)

    print(f"\n=== Summary ===")
    print(f"  Reconstructed successfully : {n_ok} / {len(label_files)}")
    print(f"  Unrecognized state prefix  : {n_no_state}")
    print(f"  Parcel not found in Regrid : {n_no_parcel}")
    print(f"  Grid row/col mismatch      : {n_grid_mismatch}")
    print(f"  Unexpected errors          : {n_unexpected}")
    print(f"\nWritten: {out_metadata}")
    print(f"Written: {out_labels} ({n_ok} clean-named label file(s))")
    print(f"\nNEXT: python convert_tiles_to_500m.py --source-root \"{out_root.resolve()}\"")


if __name__ == "__main__":
    main()