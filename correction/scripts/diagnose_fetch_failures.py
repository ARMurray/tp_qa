"""
diagnose_fetch_failures.py
============================
01b writes a fetch_error string for every tile with outcome='fetch_failed'.
This prints the actual distribution of those messages for one state, since
a generic failure COUNT doesn't say whether the cause is rate-limiting
(transient, retry-able) or something deterministic (no NAIP coverage, a
geometry/CRS edge case, etc -- which reruns will never fix).

Usage:
    python diagnose_fetch_failures.py --state KY
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    args = ap.parse_args()
    state = args.state.upper()

    tiles_dir = C.OD_OUTPUT_DIR / "tiles" / f"state={state}"
    if not tiles_dir.exists():
        print(f"No tiles output found for {state} at {tiles_dir}")
        return

    files = sorted(tiles_dir.glob("*.parquet"), key=lambda f: f.stat().st_mtime)
    print(f"Reading {len(files)} part-file(s) for {state}...")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    # Dedup by tile_id, keeping the most recent attempt -- without this, a
    # --no-resume rerun's part-file just gets unioned with the original
    # run's, DOUBLE (or more) counting the same tiles. Caught 2026-08-21:
    # KY showed "134 fetch_failed" here vs "67" in the run's own summary --
    # exactly 2x, from 2 part-files never being deduplicated.
    n_before = len(df)
    df = df.sort_values("processed_at").drop_duplicates(subset="tile_id", keep="last")
    if len(df) < n_before:
        print(f"Deduplicated: {n_before} raw tile rows -> {len(df)} unique tiles "
              f"(kept most recent attempt per tile_id)")

    failed = df[df["outcome"] == "fetch_failed"]
    print(f"\n{len(df)} total tile rows, {len(failed)} fetch_failed\n")

    if len(failed) == 0:
        print("No failures to diagnose.")
        return

    print("--- fetch_error message distribution ---")
    print(failed["fetch_error"].value_counts(dropna=False).to_string())

    print("\n--- per-CWNS_ID breakdown ---")
    per_plant = failed.groupby("CWNS_ID").size().sort_values(ascending=False)
    print(per_plant.to_string())

    # Cross-reference the plants table for parcel size/tile-grid context on
    # the affected plants -- tells us whether this is a genuinely large
    # parcel producing a big multi-tile grid (expected to have more
    # failure-surface) vs. something odd going on for a normal-sized parcel.
    plants_dir = C.OD_OUTPUT_DIR / "plants" / f"state={state}"
    if plants_dir.exists():
        plant_files = sorted(plants_dir.glob("*.parquet"), key=lambda f: f.stat().st_mtime)
        pframes = [pd.read_parquet(f) for f in plant_files]
        plants_df = pd.concat(pframes, ignore_index=True)
        plants_df["CWNS_ID"] = plants_df["CWNS_ID"].astype(str)
        plants_df = plants_df.sort_values("processed_at").drop_duplicates(
            subset="CWNS_ID", keep="last")

        print("\n--- affected plants: parcel context ---")
        for cwns_id, n_failed in per_plant.items():
            n_tiles_total = len(df[df["CWNS_ID"] == cwns_id])
            row = plants_df[plants_df["CWNS_ID"] == cwns_id]
            if len(row):
                area = row["parcel_area_m2"].iloc[0] if "parcel_area_m2" in row.columns else None
                capped = row["parcel_capped"].iloc[0] if "parcel_capped" in row.columns else None
                area_str = f"{area/1e6:.3f} km2" if area is not None else "unknown"
                print(f"  {cwns_id}: {n_failed}/{n_tiles_total} tiles failed, "
                      f"parcel_area={area_str}, parcel_capped={capped}")
            else:
                print(f"  {cwns_id}: {n_failed}/{n_tiles_total} tiles failed, "
                      f"NOT FOUND in plants table (odd -- worth a look)")

    print("\n--- sample errors (first 3, full detail) ---")
    for err in failed["fetch_error"].unique()[:3]:
        print(f"\n  {err}")


if __name__ == "__main__":
    main()