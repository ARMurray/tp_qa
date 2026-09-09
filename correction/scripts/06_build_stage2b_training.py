"""
06_build_stage2b_training.py
==============================
Builds the contrastive training table for Stage 2b (the not-yet-trained
OD-aware final-candidate-selection model, design settled 2026-08-21).

For each corrections-bin plant, assembles TWO rows:
  - label=0 (wrong): the REPORTED location's parcel, with OD features from
    01b's existing run (data/od_features/) and non-OD features from the
    existing 10_parcel_features.parquet (already computed for Stage 1/2a).
  - label=1 (right): the CORRECTED (true) location's parcel, with OD
    features from 01c's run (data/od_features_corrected/) and non-OD
    features from the SAME 10_parcel_features.parquet if that parcel
    happens to already be in it (it was built from 01a's k=18-ring search
    around the REPORTED point, so a corrected parcel nearby is often
    already there) -- or dropped, with a clear count reported, if not.
    Building a targeted NLCD top-up for missing corrected parcels is
    deliberately NOT done here until we see whether that's actually a
    meaningful loss on real data, rather than pre-building a speculative
    mini-pipeline for a problem that might affect only a handful of plants.

Both OD runs are keyed by CWNS_ID (each is "this plant's OD result at ONE
specific location"), so joining OD features needs no ll_uuid at all. The
non-OD (NLCD/Regrid) side does need ll_uuid, which OD's plant output never
carried (see 01c's docstring) -- so this script does its OWN fresh
point-in-parcel lookup for both the reported and corrected coordinates,
purely to get ll_uuid for that join.

Output: data/features/16_stage2b_training.parquet
    One row per (plant, location-type) pair, label 0/1, CWNS_ID, plus all
    OD + non-OD feature columns (same names as Stage 1/2a's feature tables,
    so model_utils.py's build_preprocessor()/spatial_cluster_folds() work
    unchanged on this table).

Usage:
    python 06_build_stage2b_training.py
    python 06_build_stage2b_training.py --allow-no-holdout   # pre-holdout only
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from holdout import exclude_holdout

# Reuse 02's add_name_matching rather than reimplementing it -- filename starts
# with a digit so a normal `import` isn't available; same importlib pattern 05
# already uses. Drift between two copies of this logic would put Stage 2b's
# name-match features quietly out of step with Stage 1/2a's, which is the exact
# failure mode 05's docstring warns about for point_in_parcel_lookup.
_spec = importlib.util.spec_from_file_location(
    "feat_eng", Path(__file__).resolve().parent / "02_feature_engineering.py")
feat_eng = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(feat_eng)

add_name_matching = feat_eng.add_name_matching

# The four plant-level census columns add_name_matching reads. This table is
# built from parcel + OD features only, so they have to be brought in
# explicitly. Deliberately NOT the whole of 05_plant_features.parquet: that
# would also drag in LATITUDE/LONGITUDE, which the neg/pos frames already set
# per-row from Original_X/Y vs Corrected_X/Y, and the collision would suffix
# both sides into LATITUDE_x/LATITUDE_y (the same merge trap 02's
# build_stage1_training documents). is_rural comes along because
# add_name_matching's place_match branch reads it, and it is already a live
# feature in Stage 1 and Stage 2a, so this keeps the three stages consistent.
NAME_MATCH_GEO_COLS = ["subdivision", "place", "county", "is_rural"]


def read_od_plants_union(od_root: Path) -> pd.DataFrame:
    """Read and union every part-file under {od_root}/plants/, deduplicated
    by CWNS_ID (keep most-recently-written row). Same pattern as
    02_feature_engineering.py's OD read -- see that file for why this can't
    just be pd.read_parquet(directory) (pyarrow whole-dir schema-merge
    crash) or "keep newest part-file per partition" (silently drops rows
    from a resumable, multi-flush pipeline like 01b/01c's)."""
    plants_dir = od_root / "plants"
    if not plants_dir.exists() or not any(plants_dir.rglob("*.parquet")):
        return pd.DataFrame()

    part_files_by_partition = {}
    for f in plants_dir.rglob("part-*.parquet"):
        part_files_by_partition.setdefault(f.parent, []).append(f)

    frames = []
    for files in part_files_by_partition.values():
        for f in sorted(files, key=lambda f: f.stat().st_mtime):
            df_part = pd.read_parquet(f)
            for col in df_part.select_dtypes(include=["category"]).columns:
                df_part[col] = df_part[col].astype(str)
            frames.append(df_part)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(combined):
        combined["CWNS_ID"] = combined["CWNS_ID"].astype(str)
        combined = combined.sort_values("processed_at").drop_duplicates(
            subset="CWNS_ID", keep="last")
    return combined


def lookup_ll_uuid(con, state: str, cwns_ids, lons, lats) -> pd.DataFrame:
    """Fresh point-in-parcel lookup, same ST_Intersects pattern used
    throughout this pipeline. Returns CWNS_ID -> ll_uuid, one row per match
    (a point exactly on a shared boundary can match >1 parcel -- caller
    should dedupe if that matters)."""
    pts = pd.DataFrame({"CWNS_ID": cwns_ids, "LONGITUDE": lons, "LATITUDE": lats})
    con.register("pts", pts)
    try:
        result = con.execute(f"""
            SELECT pts.CWNS_ID, p.{C.PARCEL_ID_FIELD} AS ll_uuid
            FROM read_parquet('{C.PARCEL_BASE.as_posix()}/state={state}/*.parquet') p
            JOIN pts ON ST_Intersects(
                ST_GeomFromWKB(p.{C.PARCEL_WKB_FIELD}),
                ST_Point(pts.LONGITUDE, pts.LATITUDE)
            )
        """).df()
    except Exception as e:
        print(f"    Point-in-parcel lookup failed for {state}: {e}")
        return pd.DataFrame(columns=["CWNS_ID", "ll_uuid"])
    finally:
        con.unregister("pts")
    result["CWNS_ID"] = result["CWNS_ID"].astype(str)
    return result.drop_duplicates(subset="CWNS_ID", keep="first")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-no-holdout", action="store_true",
                    help="proceed even if the holdout manifest is missing. Only "
                         "for deliberate pre-holdout runs -- normally a missing "
                         "manifest should stop the job.")
    args = ap.parse_args()

    C.ensure_dirs()
    print("=== 06_build_stage2b_training.py ===\n")

    # ---- Load corrections-bin plants: both coordinate sets ----
    corrections = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CORRECTIONS)
    corrections["CWNS_ID"] = corrections["CWNS_ID"].astype(str)
    corrections = corrections.dropna(
        subset=["Original_X", "Original_Y", "Corrected_X", "Corrected_Y"])
    loc = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt", dtype={"CWNS_ID": str}, encoding="latin1")
    loc = loc[["CWNS_ID", "STATE_CODE"]].drop_duplicates(subset="CWNS_ID")
    corrections = corrections.merge(loc, on="CWNS_ID", how="left").dropna(subset=["STATE_CODE"])
    print(f"Corrections-bin plants: {len(corrections)}")

    # ---- ll_uuid lookup for both coordinate sets, per state ----
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; SET enable_geoparquet_conversion = false;")

    reported_matches, corrected_matches = [], []
    for state, grp in corrections.groupby("STATE_CODE"):
        reported_matches.append(lookup_ll_uuid(
            con, state, grp["CWNS_ID"], grp["Original_X"], grp["Original_Y"]))
        corrected_matches.append(lookup_ll_uuid(
            con, state, grp["CWNS_ID"], grp["Corrected_X"], grp["Corrected_Y"]))
    con.close()

    reported_ll = pd.concat(reported_matches, ignore_index=True) if reported_matches else pd.DataFrame()
    corrected_ll = pd.concat(corrected_matches, ignore_index=True) if corrected_matches else pd.DataFrame()
    reported_ll = reported_ll.rename(columns={"ll_uuid": "reported_ll_uuid"})
    corrected_ll = corrected_ll.rename(columns={"ll_uuid": "corrected_ll_uuid"})
    print(f"Reported-location parcel matches : {len(reported_ll)} / {len(corrections)}")
    print(f"Corrected-location parcel matches: {len(corrected_ll)} / {len(corrections)}")

    # ---- Non-OD (NLCD/Regrid) features, keyed by ll_uuid ----
    parcel_features_path = C.FEATURES_OUTPUT_DIR / "10_parcel_features.parquet"
    if not parcel_features_path.exists():
        print(f"ERROR: {parcel_features_path} not found -- run 02_feature_engineering.py first.")
        return
    parcel_features = pd.read_parquet(parcel_features_path)
    known_uuids = set(parcel_features["ll_uuid"])

    n_reported_found = reported_ll["reported_ll_uuid"].isin(known_uuids).sum()
    n_corrected_found = corrected_ll["corrected_ll_uuid"].isin(known_uuids).sum()
    print(f"\nReported parcels already in 10_parcel_features.parquet: "
          f"{n_reported_found} / {len(reported_ll)}")
    print(f"Corrected parcels already in 10_parcel_features.parquet: "
          f"{n_corrected_found} / {len(corrected_ll)}")
    if n_corrected_found < len(corrected_ll):
        print(f"  NOTE: {len(corrected_ll) - n_corrected_found} corrected parcels have no "
              f"precomputed NLCD/Regrid features (outside 01a's original k={C.K_RINGS}-ring "
              f"search around the reported point) -- DROPPED from this training set for now. "
              f"Not building a targeted NLCD top-up until this count is known to matter.")

    # ---- OD features, keyed by CWNS_ID (one run per location-type) ----
    od_reported = read_od_plants_union(C.OD_OUTPUT_DIR)
    od_corrected = read_od_plants_union(C.OD_OUTPUT_DIR_CORRECTED)
    print(f"\nOD (reported) plants available : {len(od_reported)}")
    print(f"OD (corrected) plants available: {len(od_corrected)}")

    # Drop columns that don't belong in a feature table / would leak the
    # answer or collide across the reported+corrected merge.
    od_drop_cols = ["state", "orig_lon", "orig_lat", "n_objects_total",
                     "n_objects_in_parcel", "processed_at"]
    od_reported = od_reported.drop(columns=od_drop_cols, errors="ignore")
    od_corrected = od_corrected.drop(columns=od_drop_cols, errors="ignore")

    # ---- Assemble negative (reported/wrong) rows ----
    neg = reported_ll.merge(parcel_features, left_on="reported_ll_uuid", right_on="ll_uuid", how="inner")
    neg = neg.merge(od_reported, on="CWNS_ID", how="left")
    neg = neg.merge(corrections[["CWNS_ID", "Original_X", "Original_Y"]], on="CWNS_ID", how="left")
    neg = neg.rename(columns={"Original_X": "LONGITUDE", "Original_Y": "LATITUDE"})
    neg["label"] = 0

    # ---- Assemble positive (corrected/right) rows ----
    pos = corrected_ll.merge(parcel_features, left_on="corrected_ll_uuid", right_on="ll_uuid", how="inner")
    pos = pos.merge(od_corrected, on="CWNS_ID", how="left")
    pos = pos.merge(corrections[["CWNS_ID", "Corrected_X", "Corrected_Y"]], on="CWNS_ID", how="left")
    pos = pos.rename(columns={"Corrected_X": "LONGITUDE", "Corrected_Y": "LATITUDE"})
    pos["label"] = 1

    neg = neg.drop(columns=["reported_ll_uuid"], errors="ignore")
    pos = pos.drop(columns=["corrected_ll_uuid"], errors="ignore")

    stage2b = pd.concat([neg, pos], ignore_index=True)

    # ---- Name matching (2026-08-25) ----
    # Stage 1 and Stage 2a both run their assembled frames through
    # add_name_matching, which turns the raw parcel `owner` text into
    # is_municipal / owner_water / sd_match / place_match / county_match /
    # any_geo_match and then DROPS `owner` itself. This script never called it,
    # with two consequences: Stage 2b had none of those six features, and the
    # raw owner string survived into 16_stage2b_training.parquet, where 07's
    # build_preprocessor() one-hot encoded it into 226 levels -- 74% of the
    # deployed model's entire categorical feature space, almost all of them
    # novel at inference. That is the same trap the raw geography dummies were
    # (master reference S3), on a far higher-cardinality column.
    plant_features_path = C.FEATURES_OUTPUT_DIR / "05_plant_features.parquet"
    if not plant_features_path.exists():
        print(f"ERROR: {plant_features_path} not found -- run 02_feature_engineering.py first.")
        return
    plant_geo = pd.read_parquet(plant_features_path)[["CWNS_ID"] + NAME_MATCH_GEO_COLS]
    plant_geo["CWNS_ID"] = plant_geo["CWNS_ID"].astype(str)
    n_before = len(stage2b)
    stage2b = stage2b.merge(plant_geo.drop_duplicates(subset="CWNS_ID"),
                            on="CWNS_ID", how="left")
    assert len(stage2b) == n_before, \
        "census-geography join changed row count -- duplicate CWNS_IDs in plant_features"
    n_no_geo = int(stage2b["county"].isna().sum())
    if n_no_geo:
        print(f"  NOTE: {n_no_geo} / {len(stage2b)} rows have no census geography "
              f"(plant absent from 05_plant_features.parquet for this run's scope) "
              f"-- their geo-match features fall back to the keyword-only patterns.")
    stage2b = add_name_matching(stage2b)
    print(f"  Name-match features added; raw 'owner' column consumed and dropped.")

    # x_5070/y_5070 -- needed by model_utils.py's spatial_cluster_folds(),
    # same pattern as Stage 1/2a. Each row's location depends on its label
    # (reported point for label=0, corrected point for label=1), which is
    # exactly what LONGITUDE/LATITUDE above already carries per-row.
    pts_5070 = gpd.GeoSeries(
        gpd.points_from_xy(stage2b["LONGITUDE"], stage2b["LATITUDE"]), crs=C.EXPORT_CRS
    ).to_crs(C.PROJECTED_CRS)
    stage2b["x_5070"] = pts_5070.x
    stage2b["y_5070"] = pts_5070.y

    # od_ran/od_has_detection can be NaN for plants where the OD run itself
    # produced no row at all (e.g. no parcel found) -- same fillna pattern
    # as build_stage1_training/build_stage2_training use.
    if "od_ran" in stage2b.columns:
        stage2b["od_ran"] = stage2b["od_ran"].fillna(False)
    if "od_has_detection" in stage2b.columns:
        stage2b["od_has_detection"] = stage2b["od_has_detection"].fillna(False)
    for c in stage2b.columns:
        if c.startswith("od_has_"):
            stage2b[c] = stage2b[c].fillna(False)
        elif c.startswith("od_n_"):
            stage2b[c] = stage2b[c].fillna(0)

    print(f"\n=== Stage 2b training table ===")
    print(f"Total rows  : {len(stage2b)}")
    print(f"Label=0 (wrong)  : {(stage2b['label'] == 0).sum()}")
    print(f"Label=1 (right)  : {(stage2b['label'] == 1).sum()}")
    n_plants_both = len(set(neg["CWNS_ID"]) & set(pos["CWNS_ID"]))
    print(f"Plants with BOTH a positive and negative row (usable pairs): {n_plants_both}")

    # Exclude holdout plants before writing, so 07 receives a table that is
    # already clean. 07 excludes again -- belt and braces, and the second call
    # is a no-op that costs nothing.
    stage2b = exclude_holdout(stage2b, "stage2b-build",
                              allow_missing=args.allow_no_holdout)

    out_path = C.FEATURES_OUTPUT_DIR / "16_stage2b_training.parquet"
    stage2b.to_parquet(out_path, index=False)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()