"""
diagnose_scoreable_dtypes.py
=============================
05_run_inference.py's crash is runtime-constructed (point_in_parcel_lookup's
live DuckDB output, or the OD merge) -- not present in any already-written
parquet file, so diagnose_bytearray_columns.py came back clean. This
reproduces run_stage1()'s exact steps on a small plant sample and inspects
the actual `scoreable` frame right before scoring, which is what crashed.

Usage:
    python diagnose_scoreable_dtypes.py --states OH --limit 20
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import h3
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

_spec = importlib.util.spec_from_file_location(
    "feat_eng", Path(__file__).resolve().parent / "02_feature_engineering.py")
feat_eng = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(feat_eng)
point_in_parcel_lookup = feat_eng.point_in_parcel_lookup
add_name_matching = feat_eng.add_name_matching


def report_dtypes(df: pd.DataFrame, label: str):
    print(f"\n--- {label}: {len(df)} rows, {len(df.columns)} cols ---")
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        types_seen = non_null.map(type).value_counts()
        if len(types_seen) > 1 or types_seen.index[0] not in (str, bool, int, float,
                                                                pd.Timestamp, type(None)) \
                and not str(types_seen.index[0]).startswith(("<class 'numpy"), ):
            print(f"  SUSPECT column '{col}': dtype={df[col].dtype}, "
                  f"python types={dict(types_seen)}")
            sample = non_null.iloc[0]
            print(f"    sample value: {sample!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=str, required=True)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    states = [s.strip() for s in args.states.split(",")]

    print("=== diagnose_scoreable_dtypes.py ===")

    facility_types = pd.read_csv(C.CWNS_DIR / "FACILITY_TYPES.txt", dtype=str, encoding="latin1")
    treatment_ids = set(facility_types.loc[facility_types["FACILITY_TYPE"] == "Treatment Plant", "CWNS_ID"])
    loc = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt", dtype={"CWNS_ID": str}, encoding="latin1")
    loc = loc[loc["CWNS_ID"].isin(treatment_ids)]
    loc["LATITUDE"] = pd.to_numeric(loc["LATITUDE"], errors="coerce")
    loc["LONGITUDE"] = pd.to_numeric(loc["LONGITUDE"], errors="coerce")
    loc = loc.dropna(subset=["LATITUDE", "LONGITUDE"]).drop_duplicates(subset="CWNS_ID")
    loc = loc[loc["STATE_CODE"].isin(states)].head(args.limit).reset_index(drop=True)
    loc["h3_res9"] = loc.apply(lambda r: h3.latlng_to_cell(r["LATITUDE"], r["LONGITUDE"], 9), axis=1)
    print(f"Sample: {len(loc)} plants")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; SET enable_geoparquet_conversion = false;")

    reported = []
    for state, grp in loc.groupby("STATE_CODE"):
        pts = pd.DataFrame({
            "CWNS_ID": grp["CWNS_ID"], "h3_res9": grp["h3_res9"],
            "geom_wkb": gpd.points_from_xy(grp["LONGITUDE"], grp["LATITUDE"]).map(lambda g: g.wkb),
        })
        matched = point_in_parcel_lookup(con, state, pts)
        report_dtypes(matched, f"point_in_parcel_lookup raw output ({state})")
        if len(matched):
            reported.append(matched)
    reported_parcels = pd.concat(reported, ignore_index=True) if reported else pd.DataFrame()
    reported_parcels = reported_parcels.drop_duplicates(subset="CWNS_ID").rename(columns={"ll_uuid": "reported_ll_uuid"})
    report_dtypes(reported_parcels, "reported_parcels (deduped, renamed)")

    out = loc.merge(reported_parcels, on="CWNS_ID", how="left")
    report_dtypes(out, "after merge onto plant list")

    scoreable = out[out["reported_ll_uuid"].notna()].copy()
    plant_features = pd.read_parquet(C.FEATURES_OUTPUT_DIR / "05_plant_features.parquet")
    parcel_features = pd.read_parquet(C.FEATURES_OUTPUT_DIR / "10_parcel_features.parquet")
    scoreable = scoreable.merge(
        plant_features.drop(columns=["STATE_CODE", "LATITUDE", "LONGITUDE"], errors="ignore"),
        on="CWNS_ID", how="left"
    ).merge(
        parcel_features.rename(columns={"ll_uuid": "reported_ll_uuid"}).drop(columns=["state"], errors="ignore"),
        on="reported_ll_uuid", how="left")
    report_dtypes(scoreable, "after plant_features + parcel_features merge")

    scoreable = add_name_matching(scoreable)
    report_dtypes(scoreable, "after add_name_matching")

    con.close()
    print("\n=== complete ===")


if __name__ == "__main__":
    main()
