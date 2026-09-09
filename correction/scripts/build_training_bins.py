"""
build_training_bins.py
========================
Builds the training_locations.gpkg that 01b_run_object_detection.py's
load_training_plant_ids() and 02_feature_engineering.py's
build_stage1_training()/build_stage2_training() already expect -- 'classes'
and 'corrections' layers, same column contract as before. This replaces
however that file was built previously; going forward, THIS script + the
master Updates.gdb (specifically the dated CWNS_Locations_YYYYMMDD layer
produced by update_master_locations.R) is the single source of truth for
what's correct/incorrect/corrected.

Bin definitions (confirmed 2026-08-20 session):
    Correct     : Verified == "Yes" & Original_Correct == "Yes"
                  -> geometry = Original_X/Original_Y
    Incorrect   : Verified == "Yes" & Original_Correct == "No"
                  -> geometry = Original_X/Original_Y (NOT the "best available"
                     geometry column on the source file, which for these rows
                     may already be the corrected point -- using it here would
                     silently mislabel a wrong point as right)
    Corrections : subset of Incorrect that also has Corrected == "Yes" and a
                  non-null Corrected_X/Corrected_Y -- i.e. we know the actual
                  right answer, not just that the original was wrong
    Unverified  : Verified == "No" -- NOT used for any training label. Written
                  to a third layer for reference / future full-universe
                  inference use, not currently consumed by any existing script.

Duplicate_Parcel_Flag is deliberately IGNORED here -- confirmed 2026-08-20
that it was computed without restricting the OSM/parcel intersect to
treatment-plant-type parcels specifically, so it's not a reliable signal.
Worth revisiting/recomputing correctly at some point, but out of scope here.

CRS note: Original_X/Y and Corrected_X/Y are plain numeric columns (not the
feature geometry), carried through unchanged from the CWNS source -- treated
as NAD83 (EPSG:4269) to match what 02_feature_engineering.py's
build_stage2_training() already assumes when it reprojects them.

ENVIRONMENT: plain venv (geopandas + pyogrio can read OpenFileGDB layers
directly -- no arcgispro-py3-clone / GDAL MrSID dependency needed for this,
unlike the imagery-reading scripts). Works identically on the local Windows
machine or after uploading Updates.gdb to the HPC.

Usage:
    python build_training_bins.py --gdb "C:/Users/AMURRA02/OneDrive - Environmental Protection Agency (EPA)/Github/Location_Correction/data/Updates.gdb" --layer CWNS_Locations_20260820 --out training_locations.gpkg
"""
import argparse
import sys

import geopandas as gpd
import pandas as pd

REQUIRED_COLUMNS = [
    "CWNS_ID", "Verified", "Original_Correct", "Corrected",
    "Original_X", "Original_Y", "Corrected_X", "Corrected_Y",
]

REPORTED_CRS = "EPSG:4269"  # NAD83 -- matches existing build_stage2_training() assumption


def load_master(gdb_path: str, layer: str) -> gpd.GeoDataFrame:
    print(f"Loading {layer} from {gdb_path} ...")
    gdf = gpd.read_file(gdb_path, layer=layer)
    print(f"  Loaded {len(gdf)} rows")

    missing = [c for c in REQUIRED_COLUMNS if c not in gdf.columns]
    if missing:
        raise ValueError(
            f"Missing expected column(s) in {layer}: {missing}. "
            f"Check the layer name/version -- did the column names change?"
        )

    gdf["CWNS_ID"] = gdf["CWNS_ID"].astype(str)
    n_dupe = gdf["CWNS_ID"].duplicated().sum()
    if n_dupe:
        raise ValueError(
            f"{n_dupe} duplicate CWNS_ID values found -- this pipeline assumes "
            f"CWNS_ID is a unique key. Investigate before proceeding."
        )

    # Verified is expected to be fully populated (defaulted to "No" for
    # unreviewed rows per 2026-08-20 confirmation) -- but check anyway rather
    # than silently treating a real gap as "No".
    n_null_verified = gdf["Verified"].isna().sum()
    if n_null_verified:
        print(f"  WARNING: {n_null_verified} rows have null Verified (expected "
              f"fully populated with 'Yes'/'No') -- treating null as unverified, "
              f"but this is worth checking.")

    return gdf


def make_point_gdf(df: pd.DataFrame, x_col: str, y_col: str, extra_cols: list[str]) -> gpd.GeoDataFrame:
    sub = df[["CWNS_ID"] + extra_cols + [x_col, y_col]].dropna(subset=[x_col, y_col])
    # Keep x_col/y_col as PLAIN columns too, not just consumed into geometry --
    # 02_feature_engineering.py's build_stage2_training() reads Original_X/
    # Original_Y directly as attribute columns off the 'corrections' layer
    # (it builds two separate point sets -- original/wrong and corrected/true
    # -- from plain X/Y pairs, not from the geometry column). Dropping them
    # from the output here (as an earlier version of this function did) broke
    # that with a KeyError. Harmless extra columns on 'classes'/'unverified'.
    return gpd.GeoDataFrame(
        sub[["CWNS_ID"] + extra_cols + [x_col, y_col]],
        geometry=gpd.points_from_xy(sub[x_col], sub[y_col]),
        crs=REPORTED_CRS,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdb", required=True, help="path to Updates.gdb")
    ap.add_argument("--layer", required=True, help="dated CWNS_Locations layer name, e.g. CWNS_Locations_20260820")
    ap.add_argument("--out", default="training_locations.gpkg",
                     help="output gpkg path. Point config.py's TRAINING_GPKG at this file "
                          "(or overwrite the existing one at that path) once you're happy "
                          "with the diagnostic counts below.")
    ap.add_argument("--review-gpkg", default=None,
                     help="path to review_derived_locations.gpkg from "
                          "11_ingest_review_log.py (2026-08-27+). Unions its classes/"
                          "corrections rows in on top of the Updates.gdb-derived ones. "
                          "Updates.gdb wins on any CWNS_ID appearing in both -- it's the "
                          "stated single source of truth; a review verdict on an "
                          "already-labeled plant is informational, not an automatic "
                          "override. Every dropped conflict is printed explicitly.")
    args = ap.parse_args()

    gdf = load_master(args.gdb, args.layer)
    verified = gdf[gdf["Verified"].fillna("No") == "Yes"].copy()
    unverified = gdf[gdf["Verified"].fillna("No") != "Yes"].copy()
    print(f"\nVerified: {len(verified)}  |  Unverified: {len(unverified)}")

    # ---- Correct bin ----
    correct_rows = verified[verified["Original_Correct"] == "Yes"].copy()
    correct_rows["class"] = "Correct"

    # ---- Incorrect bin ----
    incorrect_rows = verified[verified["Original_Correct"] == "No"].copy()
    incorrect_rows["class"] = "Incorrect"

    print(f"Correct bin candidates  : {len(correct_rows)}")
    print(f"Incorrect bin candidates: {len(incorrect_rows)}")

    classes_src = pd.concat([correct_rows, incorrect_rows], ignore_index=True)
    classes_gdf = make_point_gdf(classes_src, "Original_X", "Original_Y", extra_cols=["class"])
    n_dropped_classes = len(classes_src) - len(classes_gdf)
    if n_dropped_classes:
        print(f"  NOTE: dropped {n_dropped_classes} rows from 'classes' with null "
              f"Original_X/Original_Y (can't build a point without them)")
    print(f"'classes' layer final size: {len(classes_gdf)} "
          f"({(classes_gdf['class'] == 'Correct').sum()} Correct, "
          f"{(classes_gdf['class'] == 'Incorrect').sum()} Incorrect)")

    # ---- Corrections bin (subset of Incorrect with a known right answer) ----
    corrections_src = incorrect_rows[
        (incorrect_rows["Corrected"] == "Yes")
        & incorrect_rows["Corrected_X"].notna()
        & incorrect_rows["Corrected_Y"].notna()
    ].copy()
    corrections_gdf = make_point_gdf(
        corrections_src, "Original_X", "Original_Y",
        extra_cols=["Corrected_X", "Corrected_Y"],
    )
    n_dropped_corrections = len(corrections_src) - len(corrections_gdf)
    if n_dropped_corrections:
        print(f"  NOTE: dropped {n_dropped_corrections} rows from 'corrections' with "
              f"null Original_X/Original_Y")
    print(f"'corrections' layer final size: {len(corrections_gdf)}")

    # ---- Unverified pool (reference only, not consumed by existing scripts yet) ----
    unverified_gdf = make_point_gdf(unverified, "Original_X", "Original_Y", extra_cols=[])
    print(f"'unverified' layer size: {len(unverified_gdf)}")

    # ---- Sanity checks before writing ----
    overlap = set(classes_gdf.loc[classes_gdf["class"] == "Correct", "CWNS_ID"]) & \
              set(classes_gdf.loc[classes_gdf["class"] == "Incorrect", "CWNS_ID"])
    assert not overlap, f"{len(overlap)} CWNS_ID(s) appear as BOTH Correct and Incorrect -- data bug, investigate"

    not_in_classes = set(corrections_gdf["CWNS_ID"]) - set(classes_gdf["CWNS_ID"])
    if not_in_classes:
        print(f"  WARNING: {len(not_in_classes)} 'corrections' CWNS_IDs are not present "
              f"in 'classes' at all -- check the Original_Correct/Corrected/Verified "
              f"combination for these rows, this shouldn't normally happen given "
              f"corrections is built as a subset of the Incorrect bin.")

    # ---- Union with review-loop-derived rows, if provided ----
    if args.review_gpkg:
        print(f"\nUnioning review-derived rows from {args.review_gpkg} ...")
        import geopandas as gpd

        def union_layer(base_gdf, layer_name):
            try:
                review_gdf = gpd.read_file(args.review_gpkg, layer=layer_name)
            except Exception as e:
                print(f"  {layer_name}: could not read from review gpkg ({e}) -- skipping")
                return base_gdf
            review_gdf["CWNS_ID"] = review_gdf["CWNS_ID"].astype(str)

            conflict_ids = set(review_gdf["CWNS_ID"]) & set(base_gdf["CWNS_ID"])
            if conflict_ids:
                print(f"  {layer_name}: {len(conflict_ids)} CWNS_ID(s) already present "
                      f"from Updates.gdb -- Updates.gdb wins, review verdict dropped for: "
                      f"{sorted(conflict_ids)}")
                review_gdf = review_gdf[~review_gdf["CWNS_ID"].isin(conflict_ids)]

            if len(review_gdf) == 0:
                return base_gdf
            combined = pd.concat([base_gdf, review_gdf], ignore_index=True)
            combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=base_gdf.crs)
            print(f"  {layer_name}: +{len(review_gdf)} review-derived row(s) "
                  f"({len(base_gdf)} -> {len(combined)})")
            return combined

        classes_gdf = union_layer(classes_gdf, "classes")
        corrections_gdf = union_layer(corrections_gdf, "corrections")

        # Re-run the same sanity check as above, now that review rows are in --
        # a review-derived row could theoretically create a Correct/Incorrect
        # collision if something upstream is broken, and this is the one place
        # that would go undetected otherwise.
        overlap = set(classes_gdf.loc[classes_gdf["class"] == "Correct", "CWNS_ID"]) & \
                  set(classes_gdf.loc[classes_gdf["class"] == "Incorrect", "CWNS_ID"])
        assert not overlap, (
            f"{len(overlap)} CWNS_ID(s) appear as BOTH Correct and Incorrect after "
            f"unioning review data -- investigate before writing")

    # ---- Write ----
    print(f"\nWriting {args.out} ...")
    classes_gdf.to_file(args.out, layer="classes", driver="GPKG")
    corrections_gdf.to_file(args.out, layer="corrections", driver="GPKG")
    unverified_gdf.to_file(args.out, layer="unverified", driver="GPKG")

    print("\n=== Done ===")
    print(f"  classes     : {len(classes_gdf)} rows "
          f"({(classes_gdf['class']=='Correct').sum()} Correct / "
          f"{(classes_gdf['class']=='Incorrect').sum()} Incorrect)")
    print(f"  corrections : {len(corrections_gdf)} rows")
    print(f"  unverified  : {len(unverified_gdf)} rows")
    print(f"\nOutput: {args.out}")
    print("Point config.py's / config_hpc.py's TRAINING_GPKG at this file "
          "(with TRAINING_LAYER_CLASSES='classes', TRAINING_LAYER_CORRECTIONS="
          "'corrections') once you're satisfied with these counts.")


if __name__ == "__main__":
    main()