"""
01_sample_sites.py
==================
Builds a labeling sample. Three possible site sources, selected with
--source, written as layers into data/samples/training_sample_round2.gpkg:

    plants  -> correct-location treatment plants, stratified region x pop
    review  -> rank-1 false positives from the last review round, i.e. the
               model's own confident mistakes (see --review-csv)
    tri     -> TRI industrial hard negatives, stratified region x NAICS
    all     -> all three

02_extract_tiles.py reads whichever layers exist. Nothing else feeds it.

SIZING
    --n controls the plant/review draw, --n-tri the TRI draw. Defaults are
    deliberately small: tile counts multiply (a parcel wider than one 500m
    tile yields 9 tiles, not 1), so a draw of 20 sites is usually 40-80
    tiles to review. Run 02 with --dry-run to see the real number before
    fetching anything.

    PER_REGION_CAP in config.py is a flat 50, which never binds on a small
    draw -- all 20 sites could land in one region. --n therefore scales the
    cap proportionally unless you pass --flat-region-cap.

REVIEW SOURCE
    Reads the CSV written by review_app/analysis/find_false_top_picks.py --
    each row is a parcel the model ranked ABOVE the correct answer, i.e. a
    confirmed visual false positive. Exactly the "visually confusable"
    negative category that script's own docstring notes this pipeline was
    missing.

    rank1_ll_uuid is the authoritative locator, NOT rank1_lat/rank1_lon.
    For polygon parcels those coordinates come from _flatten_coords(), a
    vertex mean that the source script explicitly flags as "not a real
    centroid" -- on an L-shaped or crescent parcel it can fall outside the
    parcel entirely. 02_extract_tiles.py therefore resolves the geometry by
    uuid against the Regrid mirror and takes a true centroid. Rows whose
    uuid isn't in the county parquet are SKIPPED with a message, not tiled
    from the approximate point -- a tile centered off-parcel is worse than
    no tile, since it costs review time and teaches the model nothing. The
    lat/lon is still carried through for the county sjoin and for eyeballing
    the sample in QGIS.

    geoid is not in the CSV and is recovered by county spatial join.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

# config.py lives in THIS directory (detection/pipeline/), not the parent.
# The previous `parents[1]` worked only because Python puts the script's own
# directory on sys.path anyway -- but it inserted detection/ at position 0,
# so a config.py appearing at the repo root would silently have won.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

SAMPLE_LAYER_REVIEW = "review_fp"

DEFAULT_N_SITES = 25
DEFAULT_N_TRI = 10

# Pinned to find_false_top_picks.py's actual output header (confirmed
# 2026-09-03), with fallbacks kept in case that script's columns move.
# Note 'state_code' is already a POSTAL code, not FIPS -- it comes from the
# review app's plants table and is what parcels.get_parcel_context() passes
# to the same Regrid mirror PARCEL_BASE points at. No conversion needed.
# There is no geoid column; it's recovered by county sjoin below.
REVIEW_COL_CANDIDATES = {
    "lat": ["rank1_lat", "lat", "latitude", "y", "cand_lat"],
    "lon": ["rank1_lon", "lon", "longitude", "x", "cand_lon"],
    "cwns": ["cwns_id", "CWNS_ID", "cwnsid"],
    "uuid": ["rank1_ll_uuid", "ll_uuid", "ll_uuid_primary", "parcel_id"],
    "geoid": ["geoid", "GEOID", "county_geoid"],
    "state": ["state_code", "st", "STATE_CODE", "state", "STATE_ABBR"],
}


# ---------------------------------------------------------------------------
def assign_region(state_series: pd.Series) -> pd.Series:
    return state_series.map(C.STATE_TO_REGION).fillna("Other")


def resolve_col(df: pd.DataFrame, key: str) -> str | None:
    for cand in REVIEW_COL_CANDIDATES[key]:
        if cand in df.columns:
            return cand
    return None


def region_cap(n: int, flat: bool) -> int:
    """Scale the per-region cap to the draw size so a small --n stays spread.
    1.5x slack over an even split: enough to let a region with more good
    candidates take extra, not enough for one region to take everything."""
    if flat:
        return C.PER_REGION_CAP
    n_regions = max(1, len(C.REGIONS))
    return max(1, math.ceil(n / n_regions * 1.5))


def load_population() -> pd.DataFrame:
    cols = ["CWNS_ID", "TOTAL_RES_POPULATION_2022"]
    p1 = pd.read_csv(C.POP_FILE_1, usecols=lambda c: c in cols)
    p2 = pd.read_csv(C.POP_FILE_2, usecols=lambda c: c in cols)
    pop = pd.concat([p1, p2], ignore_index=True).drop_duplicates()
    pop["CWNS_ID"] = pop["CWNS_ID"].astype(str)
    return pop


def load_counties() -> gpd.GeoDataFrame:
    if C.COUNTIES_GPKG:
        counties = gpd.read_file(C.COUNTIES_GPKG)
    else:
        from pygris import counties as pygris_counties
        counties = pygris_counties(cb=True, cache=True)
    return counties[["GEOID", "geometry"]].to_crs(C.EXPORT_CRS)


def attach_county_geoid(pts: gpd.GeoDataFrame,
                        state_col: str = "STATE_CODE") -> gpd.GeoDataFrame:
    counties = load_counties()
    joined = gpd.sjoin(pts, counties, how="inner", predicate="intersects")
    joined = joined.drop(columns=["index_right"]).rename(columns={"GEOID": "geoid"})
    if state_col in joined.columns:
        joined = joined.rename(columns={state_col: "st"})
    return joined


# ===========================================================================
# Plants
# ===========================================================================
def load_correct_plants() -> gpd.GeoDataFrame:
    """Correct-only point set, using the right X/Y columns for each subset."""
    gdf = gpd.read_file(C.UPDATES_GDB, layer=C.CWNS_LAYER)
    df = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
    df["CWNS_ID"] = df["CWNS_ID"].astype(str)

    parcel = df[df["How_Corrected"] == "Parcel"].copy()
    parcel["X"], parcel["Y"] = parcel["Corrected_X"], parcel["Corrected_Y"]

    orig = df[df["Original_Correct"] == "Yes"].copy()
    orig["X"], orig["Y"] = orig["Original_X"], orig["Original_Y"]

    keep = ["CWNS_ID", "STATE_CODE", "X", "Y"]
    both = pd.concat([parcel[keep], orig[keep]], ignore_index=True)
    both = both.dropna(subset=["X", "Y"]).drop_duplicates("CWNS_ID")
    both["class"] = "correct"

    return gpd.GeoDataFrame(
        both, geometry=gpd.points_from_xy(both["X"], both["Y"]), crs=C.EXPORT_CRS)


def sample_plants(plants: gpd.GeoDataFrame, pop: pd.DataFrame,
                  n: int, flat_cap: bool) -> gpd.GeoDataFrame:
    """Stratified weighted draw: region x pop bin, capped per region then total."""
    g = plants.merge(pop, on="CWNS_ID", how="left")
    g["region"] = assign_region(g["st"])
    g["pop_bin"] = pd.cut(
        g["TOTAL_RES_POPULATION_2022"],
        bins=C.POP_BIN_BREAKS, labels=C.POP_BIN_LABELS, include_lowest=True)
    g = g[(g["region"] != "Other") & g["pop_bin"].notna()].copy()
    g["sample_weight"] = g["pop_bin"].map(dict(C.POP_BIN_WEIGHTS)).astype(float)

    # Iterate groups explicitly: newer pandas drops grouping columns inside
    # .apply(), which would strip 'region'/'pop_bin' from the result.
    parts = [
        cell.sample(frac=1, weights=cell["sample_weight"],
                    random_state=C.RANDOM_SEED)
        for _, cell in g.groupby(["region", "pop_bin"], observed=True)
    ]
    shuffled = pd.concat(parts).sort_values("sample_weight", ascending=False)

    cap = region_cap(n, flat_cap)
    balanced = pd.concat([
        reg.head(cap) for _, reg in shuffled.groupby("region", observed=True)
    ]).head(n)
    return gpd.GeoDataFrame(balanced, geometry="geometry", crs=plants.crs)


# ===========================================================================
# Review false positives
# ===========================================================================
def load_review_fps(csv_path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)
    lat_c, lon_c = resolve_col(df, "lat"), resolve_col(df, "lon")
    if lat_c is None or lon_c is None:
        print(f"ERROR: could not find lat/lon columns in {csv_path.name}.")
        print(f"  looked for lat: {REVIEW_COL_CANDIDATES['lat']}")
        print(f"  looked for lon: {REVIEW_COL_CANDIDATES['lon']}")
        print(f"  actual header: {list(df.columns)}")
        print("  Add the right name to REVIEW_COL_CANDIDATES and re-run.")
        sys.exit(1)

    print(f"  using lat='{lat_c}', lon='{lon_c}'")
    df = df.dropna(subset=[lat_c, lon_c]).copy()

    out = pd.DataFrame({
        "CWNS_ID": (df[resolve_col(df, "cwns")].astype(str)
                    if resolve_col(df, "cwns") else ""),
        "class": "review_fp",
    })
    for key, name in [("uuid", "ll_uuid"), ("geoid", "geoid"), ("state", "st")]:
        col = resolve_col(df, key)
        out[name] = df[col].astype(str).values if col else None
        if col:
            print(f"  using {name}='{col}'")

    gdf = gpd.GeoDataFrame(
        out, geometry=gpd.points_from_xy(df[lon_c], df[lat_c]), crs=C.EXPORT_CRS)
    return gdf.drop_duplicates(subset=["CWNS_ID", "ll_uuid", "geometry"])


def enrich_review_fps(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Recover geoid/st by spatial join when the CSV didn't carry them.
    ll_uuid is left alone -- 02 resolves it from the parcel layer, which is
    the same source 01b used, so joining here would just duplicate that."""
    if gdf["geoid"].notna().all() and gdf["st"].notna().all():
        return gdf
    print("  geoid/st missing -- recovering by county spatial join")
    counties = load_counties()
    joined = gpd.sjoin(gdf.drop(columns=["geoid"]), counties,
                       how="left", predicate="intersects")
    joined = joined.drop(columns=["index_right"]).rename(columns={"GEOID": "geoid"})
    if joined["st"].isna().any():
        joined["st"] = joined["st"].fillna(joined["geoid"].astype(str).str[:2])
        print("  NOTE: 'st' filled from GEOID state FIPS, not postal code. "
              "02 uses it for the parcel path (state=XX); verify it matches "
              "PARCEL_BASE's layout before a full run.")
    return joined.drop_duplicates(subset=["CWNS_ID", "ll_uuid", "geometry"])


def sample_review(gdf: gpd.GeoDataFrame, n: int) -> gpd.GeoDataFrame:
    """Plain random draw. These are already a filtered, ranked set -- the
    stratification that matters was applied upstream when they were scored,
    and re-stratifying here would fight it."""
    if len(gdf) <= n:
        return gdf
    return gdf.sample(n=n, random_state=C.RANDOM_SEED)


# ===========================================================================
# TRI
# ===========================================================================
def sample_tri(n: int) -> gpd.GeoDataFrame:
    tri = gpd.read_file(C.TRI_SOURCE_GPKG, layer=C.TRI_SOURCE_LAYER).to_crs(C.EXPORT_CRS)
    state_col = "STATE_ABBR" if "STATE_ABBR" in tri.columns else "STATE_CODE"
    tri["region"] = assign_region(tri[state_col])
    tri = tri[tri["region"] != "Other"].copy()

    if "ONSITE_WATER" in tri.columns:
        tri = tri[tri["ONSITE_WATER"] == 0]

    per_cell = pd.concat([
        cell.sample(n=min(len(cell), 4), random_state=C.RANDOM_SEED)
        for _, cell in tri.groupby(["region", "INDUSTRY_CODE"])
    ])
    k = min(len(per_cell), n)
    return gpd.GeoDataFrame(
        per_cell.sample(n=k, random_state=C.RANDOM_SEED),
        geometry="geometry", crs=tri.crs)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["plants", "review", "tri", "all"],
                    default="plants",
                    help="which site source(s) to sample (default: plants)")
    ap.add_argument("--n", type=int, default=DEFAULT_N_SITES,
                    help=f"sites to draw from plants and/or review "
                         f"(default: {DEFAULT_N_SITES})")
    ap.add_argument("--n-tri", type=int, default=DEFAULT_N_TRI,
                    help=f"TRI hard negatives to draw (default: {DEFAULT_N_TRI})")
    ap.add_argument("--review-csv", type=Path,
                    default=Path(r"C:\Users\AMURRA02\tp_qa\review_app\analysis"
                                 r"\false_top_picks.csv"),
                    help="output of find_false_top_picks.py")
    ap.add_argument("--flat-region-cap", action="store_true",
                    help=f"use config's flat PER_REGION_CAP ({C.PER_REGION_CAP}) "
                         f"instead of scaling it to --n")
    args = ap.parse_args()

    print("=== 01_sample_sites.py ===")
    print(f"source: {args.source}   n: {args.n}   n_tri: {args.n_tri}\n")
    C.ensure_dirs()

    want_plants = args.source in ("plants", "all")
    want_review = args.source in ("review", "all")
    want_tri = args.source in ("tri", "all")
    wrote = []

    if want_plants:
        print("Loading correct-location plants...")
        plants = load_correct_plants()
        print(f"  {len(plants)} correct plants")
        plants = attach_county_geoid(plants).drop_duplicates("CWNS_ID")
        print(f"  {len(plants)} with county")

        cap = region_cap(args.n, args.flat_region_cap)
        print(f"Sampling plants (region x population, per-region cap {cap})...")
        plants_sample = sample_plants(plants, load_population(),
                                      args.n, args.flat_region_cap)
        print(f"  Selected {len(plants_sample)} plants")
        print(pd.crosstab(plants_sample["region"], plants_sample["pop_bin"]))

        keep = ["CWNS_ID", "class", "st", "geoid",
                "TOTAL_RES_POPULATION_2022", "region", "pop_bin", "geometry"]
        out = plants_sample[[c for c in keep if c in plants_sample.columns]]
        out.to_file(C.SAMPLE_GPKG, layer=C.SAMPLE_LAYER_PLANTS, driver="GPKG")
        wrote.append((C.SAMPLE_LAYER_PLANTS, len(out)))

    if want_review:
        print(f"\nLoading review false positives from {args.review_csv.name}...")
        if not args.review_csv.exists():
            print(f"ERROR: {args.review_csv} not found. Run "
                  f"find_false_top_picks.py first, or pass --review-csv.")
            sys.exit(1)
        fps = enrich_review_fps(load_review_fps(args.review_csv))
        print(f"  {len(fps)} candidate false positive(s)")
        fps_sample = sample_review(fps, args.n)
        print(f"  Selected {len(fps_sample)}")

        keep = ["CWNS_ID", "class", "st", "geoid", "ll_uuid", "geometry"]
        out = fps_sample[[c for c in keep if c in fps_sample.columns]]
        out.to_file(C.SAMPLE_GPKG, layer=SAMPLE_LAYER_REVIEW, driver="GPKG")
        wrote.append((SAMPLE_LAYER_REVIEW, len(out)))

    if want_tri:
        print("\nSampling TRI hard negatives (region x NAICS)...")
        tri_sample = sample_tri(args.n_tri)
        print(f"  Selected {len(tri_sample)} TRI facilities")
        print(tri_sample["region"].value_counts())

        keep = [C.TRI_ID_FIELD, "INDUSTRY_CODE", "region", "geometry"]
        out = tri_sample[[c for c in keep if c in tri_sample.columns]]
        out.to_file(C.SAMPLE_GPKG, layer=C.SAMPLE_LAYER_TRI, driver="GPKG")
        wrote.append((C.SAMPLE_LAYER_TRI, len(out)))

    print(f"\nWrote {C.SAMPLE_GPKG}")
    for layer, k in wrote:
        print(f"  layer '{layer}': {k}")
    print("\nNext: 02_extract_tiles.py --dry-run  (check the tile count first)")


if __name__ == "__main__":
    main()
