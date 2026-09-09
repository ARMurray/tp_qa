"""
diagnose_01b_output.py
========================
Standalone check on 01b's plants/ output for one state:
  1. Summary stats (od_ran / od_has_detection / od_n_objects distribution)
  2. Which training-universe CWNS_IDs for that state are MISSING from the
     output entirely (these are the "no containing parcel found" plants --
     01b doesn't currently write a row for them, only a console count, so
     this reconstructs which ones by diffing the expected vs actual CWNS_IDs)

Usage:
    python diagnose_01b_output.py --state OH
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def load_training_plant_ids() -> set:
    import geopandas as gpd
    classes = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CLASSES)
    corrections = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CORRECTIONS)
    return set(classes["CWNS_ID"].astype(str)) | set(corrections["CWNS_ID"].astype(str))


def load_expected_plants(state: str) -> pd.DataFrame:
    """Same restriction logic as 01b_run_object_detection.py's
    load_treatment_plants() -- kept in sync manually, not imported, so this
    script has zero risk of accidentally mutating pipeline state."""
    facility_types = pd.read_csv(C.CWNS_DIR / "FACILITY_TYPES.txt", dtype=str, encoding="latin1")
    treatment_ids = set(
        facility_types.loc[facility_types["FACILITY_TYPE"] == "Treatment Plant", "CWNS_ID"]
    )
    loc = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt", dtype={"CWNS_ID": str}, encoding="latin1")
    loc = loc[loc["CWNS_ID"].isin(treatment_ids)]
    loc["LATITUDE"] = pd.to_numeric(loc["LATITUDE"], errors="coerce")
    loc["LONGITUDE"] = pd.to_numeric(loc["LONGITUDE"], errors="coerce")
    loc = loc.dropna(subset=["LATITUDE", "LONGITUDE"]).drop_duplicates(subset="CWNS_ID")
    loc = loc[["CWNS_ID", "STATE_CODE", "LATITUDE", "LONGITUDE"]]

    training_ids = load_training_plant_ids()
    loc = loc[loc["CWNS_ID"].isin(training_ids)]
    loc = loc[loc["STATE_CODE"] == state]
    return loc.reset_index(drop=True)


def read_partitioned_parquet_union(dir_path: Path, verbose: bool = True) -> pd.DataFrame:
    """Read every part-file in a partition dir and UNION them in pandas
    (see 02_feature_engineering.py's OD read for why -- same schema-drift
    crash avoided the same way). Prints a per-file plant count and overlap
    check so it's visible whether resume is actually skipping already-done
    plants across separate job submissions, or silently reprocessing them."""
    files = sorted(dir_path.glob("*.parquet"), key=lambda f: f.stat().st_mtime)
    if not files:
        return pd.DataFrame()

    frames = []
    seen_ids = set()
    if verbose:
        print(f"--- Per-part-file breakdown (resume sanity check) ---")
    for f in files:
        df = pd.read_parquet(f)
        ids = set(df["CWNS_ID"].astype(str))
        overlap = ids & seen_ids
        if verbose:
            print(f"  {f.name}: {len(df)} rows, {len(ids)} unique CWNS_IDs, "
                  f"{len(overlap)} already seen in an earlier part-file"
                  + ("  <-- resume did NOT skip these" if overlap else ""))
        seen_ids |= ids
        frames.append(df)
    if verbose:
        print()

    combined = pd.concat(frames, ignore_index=True)
    combined["CWNS_ID"] = combined["CWNS_ID"].astype(str)
    n_before = len(combined)
    combined = combined.drop_duplicates(subset="CWNS_ID", keep="last")
    if verbose and n_before > len(combined):
        print(f"Deduplicated: {n_before} raw rows -> {len(combined)} unique plants "
              f"(kept most recently written row per CWNS_ID)\n")
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="2-letter STATE_CODE, e.g. OH")
    args = ap.parse_args()
    state = args.state.upper()

    plants_dir = C.OD_OUTPUT_DIR / "plants" / f"state={state}"
    print(f"=== diagnose_01b_output.py: state={state} ===")
    print(f"Reading: {plants_dir}\n")

    if not plants_dir.exists() or not any(plants_dir.glob("*.parquet")):
        print("No output found for this state yet -- nothing to diagnose.")
        return

    n_files = len(list(plants_dir.glob("*.parquet")))
    df = read_partitioned_parquet_union(plants_dir)
    print(f"{len(df)} unique plant rows in output (from {n_files} part-files)\n")

    print("--- od_ran ---")
    print(df["od_ran"].value_counts(dropna=False).to_string())
    print("\n--- od_has_detection ---")
    print(df["od_has_detection"].value_counts(dropna=False).to_string())
    print("\n--- od_n_objects distribution ---")
    print(df["od_n_objects"].describe().to_string())

    if "od_dominant_class" in df.columns:
        print("\n--- od_dominant_class (plants with a detection) ---")
        print(df.loc[df["od_has_detection"] == True, "od_dominant_class"]
              .value_counts(dropna=False).to_string())

    # --- which expected plants are missing entirely (no parcel found) ---
    print("\n=== Checking for plants missing from output (no parcel found) ===")
    expected = load_expected_plants(state)
    expected_ids = set(expected["CWNS_ID"])
    present_ids = set(df["CWNS_ID"])
    missing_ids = expected_ids - present_ids

    print(f"Expected (training-universe) plants for {state}: {len(expected_ids)}")
    print(f"Present in plants/ output              : {len(present_ids)}")
    print(f"Missing (presumed 'no parcel found')    : {len(missing_ids)}")

    if missing_ids:
        missing = expected[expected["CWNS_ID"].isin(missing_ids)].sort_values("CWNS_ID")
        print("\nMissing CWNS_IDs (check these for parcel coverage gaps):")
        print(missing.to_string(index=False))

        # Rough clustering check -- if these all sit close together, that
        # points to a real coverage gap in the Regrid data rather than N
        # unrelated coincidences.
        if len(missing) > 1:
            lat_spread = missing["LATITUDE"].max() - missing["LATITUDE"].min()
            lon_spread = missing["LONGITUDE"].max() - missing["LONGITUDE"].min()
            print(f"\nSpread across missing points: {lat_spread:.3f} deg lat, "
                  f"{lon_spread:.3f} deg lon")
            if lat_spread < 0.5 and lon_spread < 0.5:
                print("  -> These are geographically close together (< ~50km spread) -- "
                      "worth checking if they share a county with a real parcel-data gap.")
            else:
                print("  -> These are geographically spread out -- more consistent with "
                      "independent edge cases (e.g. plants on federal/unparceled land) "
                      "than one shared data gap.")
    else:
        print("\nNone missing -- every expected plant is accounted for in the output.")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()