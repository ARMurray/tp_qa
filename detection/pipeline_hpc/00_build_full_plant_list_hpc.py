"""
00_build_full_plant_list.py
============================
Prepares the full plant universe for the HPC production run from the gpkg
you built in R (data/plants.gpkg, layer 'treatment', columns: CWNS_ID,
geometry). The geometry is the best-available location per plant (includes
manual corrections where they exist) -- that's what tiles get centered on.

This is a NEW, SEPARATE list from training_sample_round2.gpkg. Keep both --
01_sample_sites.py still drives future annotation rounds; this one drives
the full production inference run across all ~17,877 plants.

Adds two things 07_run_state_pipeline.py needs that aren't in the raw R
export:
  - st      : USPS state abbreviation, derived from the CWNS_ID FIPS prefix
              (no spatial join needed for this one -- CWNS_ID already encodes it)
  - geoid   : county FIPS, via spatial join to counties (needed to locate
              the parcel parquet files, same as 01_sample_sites.py does)

Also VALIDATES that CWNS_ID survived as text. If R/GDAL wrote it as a
numeric type, any state whose FIPS code starts with '0' (01 AL, 02 AK, 04 AZ,
05 AR, 06 CA, 08 CO, 09 CT) would have silently lost its leading zero, which
would break every prefix-based state filter downstream. This script fails
loudly rather than silently mis-filing those states' plants.
"""
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from state_fips import fips_to_abbr, FIPS_TO_ABBR

OUTPUT_LAYER = "all_plants"


def validate_cwns_ids(cwns_ids: pd.Series):
    """Catch the leading-zero-truncation failure mode immediately, not later
    as a silently-wrong state filter."""
    lengths = cwns_ids.str.len().value_counts()
    if len(lengths) > 1:
        print(f"  WARNING: CWNS_ID lengths are inconsistent -- possible truncation:\n{lengths}")

    prefixes = cwns_ids.str[:2]
    bad = sorted(set(prefixes) - set(FIPS_TO_ABBR.keys()))
    if bad:
        n_bad = prefixes.isin(bad).sum()
        raise SystemExit(
            f"CWNS_ID validation FAILED: {n_bad} plants have a 2-character prefix that "
            f"isn't a known state FIPS code: {bad}\n"
            f"This almost always means CWNS_ID was stored as a number somewhere along the "
            f"way and lost a leading zero (e.g. Alabama '01...' became '1...'). "
            f"Check the dtype of CWNS_ID in R before st_write, and re-export as character."
        )
    print(f"  CWNS_ID validation OK -- all {len(cwns_ids)} prefixes match a known state FIPS code")


def attach_county_geoid(plants: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Same approach as 01_sample_sites.py -- needed to locate parcel parquet files."""
    if C.COUNTIES_GPKG:
        counties = gpd.read_file(C.COUNTIES_GPKG)
    else:
        from pygris import counties as pygris_counties
        counties = pygris_counties(cb=True, cache=True)
    counties = counties[["GEOID", "geometry"]].to_crs(C.EXPORT_CRS)
    counties["GEOID"] = counties["GEOID"].astype(str).str.zfill(5)
    joined = gpd.sjoin(plants, counties, how="inner", predicate="intersects")
    joined = joined.drop(columns=["index_right"]).rename(columns={"GEOID": "geoid"})
    return joined.drop_duplicates("CWNS_ID")


def main():
    print("=== 00_build_full_plant_list.py ===\n")

    plants = gpd.read_file(C.PLANTS_RAW_GPKG, layer=C.PLANTS_RAW_LAYER)
    print(f"Loaded {len(plants)} plants from {C.PLANTS_RAW_GPKG} (layer '{C.PLANTS_RAW_LAYER}')")
    print(f"  Source CRS: {plants.crs}")

    plants["CWNS_ID"] = plants["CWNS_ID"].astype(str)
    validate_cwns_ids(plants["CWNS_ID"])

    plants = plants.to_crs(C.EXPORT_CRS)
    plants["state_fips"] = plants["CWNS_ID"].str[:2]
    plants["st"] = plants["state_fips"].map(fips_to_abbr)

    print("\nAttaching county GEOID (needed for parcel lookup)...")
    plants = attach_county_geoid(plants)
    print(f"  {len(plants)} plants with county\n")

    print(plants["state_fips"].value_counts().sort_index())

    keep_cols = ["CWNS_ID", "st", "geoid", "state_fips", "geometry"]
    out = plants[keep_cols]

    C.ALL_PLANTS_GPKG.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(C.ALL_PLANTS_GPKG, layer=OUTPUT_LAYER, driver="GPKG")
    print(f"\nWrote {C.ALL_PLANTS_GPKG} (layer '{OUTPUT_LAYER}')")
    print("\nNext: 07_run_state_pipeline.py --state <FIPS>")


if __name__ == "__main__":
    main()