"""
diagnose_stage1_attrition.py
==============================
Two questions, both raised 2026-08-21 after the first real Stage 1 training
run:

1. Source labels have 316 Verified=Yes/Original_Correct=No (Incorrect)
   rows, but only 89 survived into 14_stage1_training.parquet. Where did
   the other 227 go? Hypothesis: Incorrect points are disproportionately
   likely to fall on NO parcel at all (being wrong often means being
   nowhere near any real parcel), so they get filtered out by
   point_in_parcel_lookup at a much higher rate than Correct points. This
   checks that directly rather than assuming it.

2. osm_ww dominated Stage 1 feature importance by a huge margin (65x the
   next feature). If that's partly because Correct-labeled training rows
   skew heavily toward OSM-tagged parcels (unsurprising, since Has_OSM was
   literally used to AUTO-SET some Original_Correct='Yes' labels in the
   first place -- see update_master_locations.R), the model may be
   partially reconstructing its own label-generation rule rather than
   learning an independent signal. Reports the OSM split within the
   SURVIVING Correct training rows specifically.

Usage:
    python diagnose_stage1_attrition.py
"""
import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def main():
    print("=== diagnose_stage1_attrition.py ===\n")

    # ---- Source labels ----
    classes = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CLASSES).to_crs(4326)
    classes["CWNS_ID"] = classes["CWNS_ID"].astype(str)
    correct_ids = set(classes.loc[classes["class"] == "Correct", "CWNS_ID"])
    incorrect_ids = set(classes.loc[classes["class"] == "Incorrect", "CWNS_ID"])
    print(f"Source labels: {len(correct_ids)} Correct, {len(incorrect_ids)} Incorrect")

    # ---- What actually survived into Stage 1 training ----
    s1_path = C.FEATURES_OUTPUT_DIR / "14_stage1_training.parquet"
    if not s1_path.exists():
        print(f"\n{s1_path} not found -- run 02_feature_engineering.py first.")
        return
    s1 = pd.read_parquet(s1_path)
    s1["CWNS_ID"] = s1["CWNS_ID"].astype(str)
    survived_correct = set(s1.loc[s1["class"] == "Correct", "CWNS_ID"])
    survived_incorrect = set(s1.loc[s1["class"] == "Incorrect", "CWNS_ID"])
    print(f"Survived into training: {len(survived_correct)} Correct, {len(survived_incorrect)} Incorrect")
    print(f"  Correct survival rate  : {len(survived_correct)/len(correct_ids)*100:.1f}%")
    print(f"  Incorrect survival rate: {len(survived_incorrect)/len(incorrect_ids)*100:.1f}%")

    # ---- Question 1: why did the dropped Incorrect rows drop? ----
    dropped_incorrect = incorrect_ids - survived_incorrect
    print(f"\n--- Dropped Incorrect rows: {len(dropped_incorrect)} ---")
    print("Checking whether the reported point truly matches NO parcel at all "
          "(vs some other reason for dropping)...")

    dropped_rows = classes[classes["CWNS_ID"].isin(dropped_incorrect)].copy()
    dropped_rows["STATE_CODE"] = dropped_rows.get("STATE_CODE")
    if "STATE_CODE" not in dropped_rows.columns or dropped_rows["STATE_CODE"].isna().all():
        # classes layer may not carry STATE_CODE directly -- derive via CWNS
        loc = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt", dtype={"CWNS_ID": str}, encoding="latin1")
        loc = loc[["CWNS_ID", "STATE_CODE"]].drop_duplicates(subset="CWNS_ID")
        dropped_rows = dropped_rows.drop(columns=["STATE_CODE"], errors="ignore").merge(
            loc, on="CWNS_ID", how="left")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; SET enable_geoparquet_conversion = false;")

    no_parcel_at_all = 0
    found_some_parcel = 0
    no_state_coverage = 0
    checked = 0
    for state, grp in dropped_rows.groupby("STATE_CODE"):
        state_dir = C.PARCEL_BASE / f"state={state}"
        if not state_dir.exists():
            no_state_coverage += len(grp)
            continue
        pts = pd.DataFrame({
            "CWNS_ID": grp["CWNS_ID"],
            "geom_wkb": grp.geometry.apply(lambda g: g.wkb),
        })
        con.register("pts", pts)
        try:
            matched = con.execute(f"""
                SELECT DISTINCT pts.CWNS_ID
                FROM pts
                JOIN read_parquet('{state_dir.as_posix()}/*.parquet') p
                  ON ST_Intersects(ST_GeomFromWKB(p.wkb_geometry), ST_GeomFromWKB(pts.geom_wkb))
            """).df()
        except Exception as e:
            print(f"  {state}: lookup failed ({e})")
            matched = pd.DataFrame(columns=["CWNS_ID"])
        finally:
            con.unregister("pts")
        n_matched = len(matched)
        found_some_parcel += n_matched
        no_parcel_at_all += (len(grp) - n_matched)
        checked += len(grp)

    print(f"  Checked: {checked} / {len(dropped_incorrect)} dropped rows "
          f"({no_state_coverage} had no local parcel coverage for their state at all)")
    print(f"  Truly match NO parcel : {no_parcel_at_all} "
          f"({no_parcel_at_all/max(checked,1)*100:.0f}% of checked)")
    print(f"  DID match a parcel    : {found_some_parcel} "
          f"({found_some_parcel/max(checked,1)*100:.0f}% of checked) "
          f"-- these were dropped for a DIFFERENT reason, worth investigating further")

    # ---- Question 2: OSM representation within surviving Correct rows ----
    print(f"\n--- OSM representation within the {len(survived_correct)} surviving Correct rows ---")
    if "osm_ww" in s1.columns:
        correct_rows = s1[s1["class"] == "Correct"]
        osm_counts = correct_rows["osm_ww"].value_counts(dropna=False)
        print(osm_counts.to_string())
        n_no_osm = int((correct_rows["osm_ww"] == False).sum())
        print(f"\n  Correct examples WITHOUT an OSM tag: {n_no_osm} "
              f"({n_no_osm/len(correct_rows)*100:.0f}% of surviving Correct rows)")
        if n_no_osm < 50:
            print("  WARNING: fewer than 50 non-OSM Correct examples -- osm_ww's "
                  "dominance in feature importance may reflect the training set's "
                  "own composition more than a generalizable signal.")
    else:
        print("  'osm_ww' column not found in 14_stage1_training.parquet")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()