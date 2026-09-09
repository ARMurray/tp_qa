"""
01a_extract_parcels.py
=======================
Port of extract_parcels.R. For every reported CWNS treatment-plant location:
  1. Expand an H3 k-ring search window (default k=18, ~5.4km radius) around the
     reported point.
  2. Pull every parcel whose H3 res-9 cell falls in that window (DuckDB
     spatial query against the Regrid parquet store).
  3. Extract NLCD land cover stats per parcel (coverage-fraction-weighted,
     >=50% pixel coverage threshold -- matches exactextractr's default
     behavior in the R version) via the `exactextract` Python package,
     which is the same author's Python port of exactextractr, so results
     should match the R pipeline's semantics exactly.

This is Stage 1/2 CANDIDATE PARCEL geometry -- separate from and independent
of 01b's object detection, which only needs the SINGLE parcel a reported
point falls on. This script pulls the full k=18 candidate neighborhood,
which Stage 2 (location correction search) needs to rank alternatives.

Output: one Parquet file per state, matching the R version's naming:
    data/nlcd_features/nlcd_{STATE_CODE}_k{K_RINGS}.parquet
    columns: ll_uuid, h3_index_9, state, total_pixels, water_pixels,
             has_water, dominant_class, dominant_count, used_centroid_fallback

NOTE (added 2026-08-20): small/narrow parcels relative to NLCD's 30m grid can
have ZERO pixels reach the >=50% coverage threshold used for the main zonal
stats (this is the R pipeline's exact behavior too, not a bug -- observed on
~46% of Ohio's k=18 candidate parcels in the first real HPC run). Rather than
leave those parcels with no land-cover signal at all, a fallback samples
whichever single NLCD pixel contains the parcel's CENTROID, no coverage
threshold. used_centroid_fallback=True marks every row where this happened,
so a real coverage-weighted stat is never silently indistinguishable from a
single-pixel approximation downstream.

Requires: duckdb, h3 (v4 API), exactextract, rasterio, geopandas, shapely, pandas

Usage:
    python 01a_extract_parcels.py [--states OH,PA] [--k-rings 18] [--no-resume]
"""
import argparse
import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely import from_wkb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


def load_training_plant_ids() -> set[str]:
    """Union of CWNS_IDs across both label layers -- Stage 1 needs 'classes'
    (correct/incorrect), Stage 2 needs 'corrections' (a subset of those with
    a known right answer). Restricting to only one would silently starve
    the other model of training rows it's supposed to have."""
    import geopandas as gpd
    if not C.TRAINING_GPKG.exists():
        raise FileNotFoundError(
            f"{C.TRAINING_GPKG} not found. Copy training_locations.gpkg in from "
            f"the Location_Correction repo first, or pass --full-universe to "
            f"skip this restriction (not recommended before a Stage 1 model exists)."
        )
    classes = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CLASSES)
    corrections = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CORRECTIONS)
    ids = set(classes["CWNS_ID"].astype(str)) | set(corrections["CWNS_ID"].astype(str))
    print(f"  Training plant universe: {len(classes)} classes + {len(corrections)} "
          f"corrections -> {len(ids)} unique CWNS_IDs")
    return ids

def load_correction_seed_points(states: list[str] | None) -> pd.DataFrame:
    """Corrected (known-true) coordinates, used as ADDITIONAL seed points for
    the H3 search window.

    01a originally seeded the window from reported points only. Because a
    correction can be further from its reported point than k rings (mean
    correction distance is ~5 km against a k=18 radius of ~5.4 km -- see
    08_diagnose_candidate_coverage.py), the true parcel for a large-error plant
    was frequently outside the window and therefore absent from 10_parcel_features
    entirely. That cost 46 of 312 corrections as of 2026-08, non-randomly the
    largest errors, i.e. exactly the cases the model most needs to learn from.

    Seeding from the corrected points too guarantees the true parcel is always in
    the candidate pool for any plant whose answer we already know. This changes
    only which parcels get NLCD stats computed -- it does not touch labels, and
    it does not tell the model where the answer is.

    Returns empty (harmless) if the corrections layer is missing, so
    --full-universe runs are unaffected."""
    if not C.TRAINING_GPKG.exists():
        return pd.DataFrame(columns=["CWNS_ID", "STATE_CODE", "LATITUDE", "LONGITUDE"])
    corr = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CORRECTIONS)
    corr["CWNS_ID"] = corr["CWNS_ID"].astype(str)
    corr = corr.dropna(subset=["Corrected_X", "Corrected_Y"])
    if not len(corr):
        return pd.DataFrame(columns=["CWNS_ID", "STATE_CODE", "LATITUDE", "LONGITUDE"])

    # STATE_CODE comes from CWNS, not the corrections layer -- a corrected point
    # near a state line still belongs to its plant's state partition in Regrid.
    loc = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt",
                      dtype={"CWNS_ID": str}, encoding="latin1")
    loc = loc[["CWNS_ID", "STATE_CODE"]].drop_duplicates(subset="CWNS_ID")
    corr = corr.merge(loc, on="CWNS_ID", how="left").dropna(subset=["STATE_CODE"])

    out = corr[["CWNS_ID", "STATE_CODE", "Corrected_Y", "Corrected_X"]].rename(
        columns={"Corrected_Y": "LATITUDE", "Corrected_X": "LONGITUDE"})
    out["LATITUDE"] = pd.to_numeric(out["LATITUDE"], errors="coerce")
    out["LONGITUDE"] = pd.to_numeric(out["LONGITUDE"], errors="coerce")
    out = out.dropna(subset=["LATITUDE", "LONGITUDE"])
    if states:
        out = out[out["STATE_CODE"].isin(states)]
    print(f"  Correction seed points: {len(out)}")
    return out.reset_index(drop=True)

def load_treatment_plants(states: list[str] | None, training_only: bool = True) -> pd.DataFrame:
    """Same filter as 01b: CWNS_ID, STATE_CODE, LATITUDE, LONGITUDE for
    treatment plants only."""
    facility_types = pd.read_csv(C.CWNS_DIR / "FACILITY_TYPES.txt", dtype=str, encoding="latin1")
    treatment_ids = set(
        facility_types.loc[facility_types["FACILITY_TYPE"] == "Treatment Plant", "CWNS_ID"]
    )
    loc = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt", dtype={"CWNS_ID": str}, encoding="latin1")
    loc = loc[loc["CWNS_ID"].isin(treatment_ids)]
    # Force real numeric dtype -- if pandas inferred LATITUDE/LONGITUDE as
    # object (happens if even one row has non-numeric/blank content), the
    # column stays Python strings for every row, not just the bad ones, and
    # a plain dropna(subset=...) only catches actual NaN -- it lets malformed
    # non-numeric strings straight through into h3.latlng_to_cell, which
    # requires real floats and errors with a confusing TypeError far from
    # the actual bad-data source. Coercing explicitly turns bad rows into
    # real NaN so dropna actually catches them here, at the source.
    loc["LATITUDE"] = pd.to_numeric(loc["LATITUDE"], errors="coerce")
    loc["LONGITUDE"] = pd.to_numeric(loc["LONGITUDE"], errors="coerce")
    n_before = len(loc)
    loc = loc.dropna(subset=["LATITUDE", "LONGITUDE"])
    if len(loc) < n_before:
        print(f"  Dropped {n_before - len(loc)} rows with non-numeric/missing LATITUDE or LONGITUDE")
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


def build_h3_search_window(plants_state: pd.DataFrame, k_rings: int) -> set[str]:
    """h3-py v4 API: latlng_to_cell / grid_disk (renamed from the R h3
    package's geo_to_h3 / k_ring -- same underlying algorithm)."""
    cells = plants_state.apply(
        lambda r: h3.latlng_to_cell(r["LATITUDE"], r["LONGITUDE"], 9), axis=1
    )
    if k_rings <= 0:
        return set(cells)
    all_cells = set()
    for c in cells.unique():
        all_cells |= set(h3.grid_disk(c, k_rings))
    return all_cells


def fetch_parcels_for_h3_cells(con: duckdb.DuckDBPyConnection, state: str,
                                h3_cells: set[str], chunk_size: int = 500) -> pd.DataFrame:
    """Chunked to keep the IN-list a reasonable size per query -- mirrors
    the R version's 500-per-chunk split, though DuckDB here runs the chunks
    sequentially rather than via furrr parallel workers (simpler, and this
    is a local single-machine run rather than an HPC array job)."""
    h3_list = list(h3_cells)
    chunks = [h3_list[i:i + chunk_size] for i in range(0, len(h3_list), chunk_size)]
    results = []
    for chunk in chunks:
        h3_filter = ", ".join(f"'{c}'" for c in chunk)
        try:
            df = con.execute(f"""
                SELECT ll_uuid, h3_index_9, state, wkb_geometry
                FROM read_parquet('{C.PARCEL_BASE.as_posix()}/state={state}/*.parquet')
                WHERE h3_index_9 IN ({h3_filter})
            """).df()
            results.append(df)
        except Exception as e:
            print(f"    Chunk fetch failed: {e}")
    if not results:
        return pd.DataFrame(columns=["ll_uuid", "h3_index_9", "state", "wkb_geometry"])
    return pd.concat(results, ignore_index=True).drop_duplicates(subset="ll_uuid")


def extract_nlcd_stats(parcels: gpd.GeoDataFrame) -> pd.DataFrame:
    """Coverage-fraction-weighted zonal stats, >=50% pixel coverage --
    matches extract_water_metrics()'s threshold in the R version exactly.
    `exactextract` (Python) is the same author's port of exactextractr (R),
    so the coverage-fraction semantics should carry over without needing to
    re-derive the algorithm."""
    from exactextract import exact_extract

    results = exact_extract(
        str(C.NLCD_PATH), parcels,
        ops=["values", "coverage"],
        include_cols=[],
        output="pandas",
    )

    rows = []
    for _, row in results.iterrows():
        vals = np.asarray(row["values"])
        cov = np.asarray(row["coverage"])
        mask = (cov >= 0.5) & ~np.isnan(vals)
        vals = vals[mask].astype(int)

        total_pixels = len(vals)
        water_pixels = int(np.sum(vals == 11))
        if total_pixels > 0:
            uniq, counts = np.unique(vals, return_counts=True)
            dominant_class = int(uniq[np.argmax(counts)])
            dominant_count = int(counts.max())
        else:
            dominant_class = None
            dominant_count = None

        rows.append(dict(
            total_pixels=total_pixels, water_pixels=water_pixels,
            dominant_class=dominant_class, dominant_count=dominant_count,
        ))
    return pd.DataFrame(rows)


def fallback_centroid_sample(parcel_proj: gpd.GeoDataFrame, results: pd.DataFrame,
                              nlcd_path: Path) -> pd.DataFrame:
    """For parcels where NO pixel reached the >=50% coverage threshold
    (small/narrow parcels relative to NLCD's 30m grid -- ~46% of OH's
    candidate parcels in the first real run, 2026-08-20), fall back to
    sampling whichever single pixel contains the parcel's centroid, no
    coverage threshold at all. Cheap (point sampling, not zonal stats) and
    gives every parcel SOME land-cover signal instead of a hard NaN/None
    that a downstream model can't use.

    IMPORTANT: parcel_proj must already be in the NLCD raster's CRS, not
    EXPORT_CRS/WGS84. rasterio's .sample() does NOT reproject coordinates --
    passing lon/lat against a projected raster silently samples the wrong
    location entirely (every point misses the raster's actual extent and
    comes back nodata). Caught 2026-08-20: passing the un-reprojected
    parcel_sf here made ALL fallback samples come back None.

    Rows that used this fallback are flagged in used_centroid_fallback so
    02_feature_engineering.py / model training can tell a real
    coverage-weighted stat from a single-pixel approximation if that
    distinction ever matters -- never silently blend the two."""
    import rasterio

    zero_mask = (results["total_pixels"] == 0).to_numpy()
    results["used_centroid_fallback"] = False
    n_zero = int(zero_mask.sum())
    if n_zero == 0:
        return results

    print(f"  {n_zero} parcels had 0 pixels at >=50% coverage -- "
          f"falling back to nearest-pixel centroid sample for these")

    centroids = parcel_proj.loc[zero_mask, "geometry"].centroid
    coords = [(pt.x, pt.y) for pt in centroids]

    with rasterio.open(nlcd_path) as src:
        nodata = src.nodata
        sampled = [v[0] for v in src.sample(coords)]

    # np.nan, not None -- dominant_class already has NaN semantics from the
    # zero-pixel rows in the main pass (mixed int/None -> pandas inferred
    # float64), and bulk-assigning a Python None into a float64 column
    # trips pandas' strict dtype checking (LossySetitemError).
    values = [float(v) if (nodata is None or v != nodata) else np.nan for v in sampled]
    idx = results.index[zero_mask]
    results.loc[idx, "dominant_class"] = values
    results.loc[idx, "dominant_count"] = [1 if not np.isnan(v) else 0 for v in values]
    results.loc[idx, "total_pixels"] = [1 if not np.isnan(v) else 0 for v in values]
    results.loc[idx, "water_pixels"] = [1 if v == 11 else 0 for v in values]
    results.loc[idx, "used_centroid_fallback"] = True

    n_still_zero = sum(np.isnan(v) for v in values)
    if n_still_zero:
        print(f"    {n_still_zero} of those still have no value -- centroid landed "
              f"on nodata (likely outside the NLCD raster's extent entirely)")

    return results


def process_state(con: duckdb.DuckDBPyConnection, state: str, plants_state: pd.DataFrame,
                   k_rings: int, out_path: Path) -> None:
    print(f"\n--- State {state}: {len(plants_state)} plants ---")

    h3_cells = build_h3_search_window(plants_state, k_rings)
    print(f"  Unique H3 cells to query: {len(h3_cells)}")

    parcel_geoms = fetch_parcels_for_h3_cells(con, state, h3_cells)
    print(f"  Parcels retrieved: {len(parcel_geoms)}")
    if len(parcel_geoms) == 0:
        print(f"  WARNING: no parcels found for state {state} -- skipping")
        return

    geoms = from_wkb([bytes(g) for g in parcel_geoms["wkb_geometry"].to_numpy()])
    parcel_sf = gpd.GeoDataFrame(
        parcel_geoms[["ll_uuid", "h3_index_9", "state"]], geometry=geoms, crs=C.EXPORT_CRS
    )

    # Drop parcels that are almost certainly Regrid digitization artifacts
    # (an entire town as one polygon, not a real parcel boundary) rather than
    # legitimate candidate locations -- see config.py's MAX_PARCEL_AREA_M2
    # for the reasoning. Area must be computed in a projected CRS (meters),
    # not EXPORT_CRS/WGS84 degrees, where .area is meaningless.
    #
    # NOTE (2026-08-21): an earlier version of this also dropped parcels by
    # bounding-box extent, mirroring 01b's tile-grid safety net. That was
    # wrong here and got reverted: exactextract computes zonal stats on the
    # ACTUAL polygon shape, not its bounding box, so a long/thin/diagonal
    # parcel doesn't cost anything extra here the way it does for 01b's
    # rectangular tiling -- and a real, legitimately-shaped elongated
    # treatment plant (confirmed by visual inspection of a KY facility
    # strung out along a road) is a perfectly valid Stage 2 candidate.
    # Area alone is still a fine filter for the actual problem (whole-town
    # digitization artifacts), which don't get any more forgivable just
    # because they're compact-shaped.
    area_m2 = parcel_sf.geometry.to_crs(C.PROJECTED_CRS).area
    oversized_mask = area_m2 > C.MAX_PARCEL_AREA_M2
    n_oversized = int(oversized_mask.sum())
    if n_oversized:
        print(f"  Dropping {n_oversized} candidate parcels over "
              f"{C.MAX_PARCEL_AREA_M2/1e6:.1f} km2 (largest: "
              f"{area_m2[oversized_mask].max()/1e6:.1f} km2) -- treated as "
              f"digitization artifacts, not real candidate locations")
        parcel_sf = parcel_sf.loc[~oversized_mask].reset_index(drop=True)

    print("  Extracting NLCD stats...")
    import rasterio
    with rasterio.open(C.NLCD_PATH) as src:
        nlcd_crs = src.crs
    parcel_proj = parcel_sf.to_crs(nlcd_crs)

    nlcd_stats = extract_nlcd_stats(parcel_proj)

    nlcd_results = pd.concat(
        [parcel_sf[["ll_uuid", "h3_index_9", "state"]].reset_index(drop=True),
         nlcd_stats.reset_index(drop=True)],
        axis=1,
    )
    n_zero_before = int((nlcd_results["total_pixels"] == 0).sum())
    if n_zero_before:
        nlcd_results = fallback_centroid_sample(parcel_proj, nlcd_results, C.NLCD_PATH)

    nlcd_results["has_water"] = nlcd_results["water_pixels"] > 0
    if "used_centroid_fallback" not in nlcd_results.columns:
        nlcd_results["used_centroid_fallback"] = False
    nlcd_results = nlcd_results[["ll_uuid", "h3_index_9", "state", "total_pixels",
                                   "water_pixels", "has_water", "dominant_class",
                                   "dominant_count", "used_centroid_fallback"]]

    print(f"  Results: {len(nlcd_results)} parcels, "
          f"{nlcd_results['water_pixels'].gt(0).sum()} with water, "
          f"{nlcd_results['used_centroid_fallback'].sum()} used centroid fallback, "
          f"{nlcd_results['total_pixels'].eq(0).sum()} with 0 pixels (real gaps, after fallback)")

    nlcd_results.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"  Written: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=str, default=None, help="comma-separated STATE_CODE filter, e.g. OH,PA")
    ap.add_argument("--k-rings", type=int, default=C.K_RINGS)
    ap.add_argument("--no-resume", action="store_true", help="reprocess states even if output already exists")
    ap.add_argument("--no-correction-seeds", action="store_true",
                     help="do NOT seed the search window from corrected coordinates "
                          "(reproduces pre-2026-08-24 behavior; use only to "
                          "regenerate a historical run for comparison)")
    ap.add_argument("--full-universe", action="store_true",
                     help="process ALL treatment plants, not just those in training_locations.gpkg "
                          "(use once a trained Stage 1 model exists to decide who needs Stage 2 -- "
                          "not recommended during training/development)")
    args = ap.parse_args()

    C.ensure_dirs()
    states_filter = [s.strip() for s in args.states.split(",")] if args.states else None

    print("=== 01a_extract_parcels.py ===")
    print(f"K rings: {args.k_rings}\n")

    plants = load_treatment_plants(states_filter, training_only=not args.full_universe)
    print(f"Treatment plants loaded: {len(plants)}")

    # Seed the search window from corrected points as well. These are extra
    # window CENTERS only -- they add rows to `plants` purely so the k-ring
    # union covers the known-true parcel. A plant can therefore appear twice
    # here (reported + corrected); build_h3_search_window() unions cells and
    # fetch_parcels_for_h3_cells() dedupes on ll_uuid, so nothing double-counts.
    if not args.full_universe and not args.no_correction_seeds:
        seeds = load_correction_seed_points(states_filter)
        if len(seeds):
            plants = pd.concat([plants, seeds], ignore_index=True)
            print(f"Seed points after adding corrections: {len(plants)}")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; SET enable_geoparquet_conversion = false;")

    for state, plants_state in plants.groupby("STATE_CODE"):
        out_path = C.NLCD_OUTPUT_DIR / f"nlcd_{state}_k{args.k_rings}.parquet"
        if out_path.exists() and not args.no_resume:
            print(f"\n--- State {state}: output already exists, skipping ({out_path}) ---")
            continue
        process_state(con, state, plants_state, args.k_rings, out_path)

    con.close()
    print("\n=== 01a_extract_parcels.py complete ===")
    print(f"Outputs in: {C.NLCD_OUTPUT_DIR}")


if __name__ == "__main__":
    main()