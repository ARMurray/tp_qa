"""
diagnose_bytearray_columns.py
===============================
05_run_inference.py crashed inside OneHotEncoder with
"TypeError: unhashable type: 'bytearray'" -- meaning some value in a
nominally-string feature column is actually raw bytes, not a Python str.
This checks every object-dtype column in 10_parcel_features.parquet (and
optionally 05_plant_features.parquet) for non-str/non-null values, and
prints sample offending rows so the real fix (decode at the DuckDB fetch
point in 02_feature_engineering.py's fetch_parcel_attrs) can target the
actual column instead of guessing.

Usage:
    python diagnose_bytearray_columns.py --states OH,MS,DE
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def check_file(path: Path, states: list[str]):
    print(f"\n=== {path.name} ===")
    if not path.exists():
        print("  not found")
        return
    df = pd.read_parquet(path)
    if "state" in df.columns:
        df = df[df["state"].isin(states)]
    elif "STATE_CODE" in df.columns:
        df = df[df["STATE_CODE"].isin(states)]
    print(f"  {len(df)} rows (scoped to {states})")

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"  object-dtype columns: {obj_cols}")

    any_bad = False
    for col in obj_cols:
        types_seen = df[col].dropna().map(type).value_counts()
        if len(types_seen) > 1 or (len(types_seen) == 1 and types_seen.index[0] is not str):
            any_bad = True
            print(f"\n  COLUMN '{col}' has non-str values:")
            print(f"    type distribution: {dict(types_seen)}")
            bad_mask = df[col].map(lambda v: v is not None and not isinstance(v, str) and pd.notna(v))
            bad_rows = df[bad_mask]
            print(f"    {len(bad_rows)} bad rows. Sample (up to 5):")
            id_col = "ll_uuid" if "ll_uuid" in df.columns else \
                     "CWNS_ID" if "CWNS_ID" in df.columns else None
            for _, r in bad_rows.head(5).iterrows():
                val = r[col]
                ident = r[id_col] if id_col else "?"
                print(f"      {id_col}={ident}  value={val!r}  type={type(val)}")
                if isinstance(val, (bytes, bytearray)):
                    try:
                        print(f"        decoded (utf-8, replace): {bytes(val).decode('utf-8', errors='replace')!r}")
                    except Exception as e:
                        print(f"        decode failed: {e}")

    if not any_bad:
        print("  No non-str values found in any object column.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=str, required=True)
    args = ap.parse_args()
    states = [s.strip() for s in args.states.split(",")]

    print("=== diagnose_bytearray_columns.py ===")
    check_file(C.FEATURES_OUTPUT_DIR / "10_parcel_features.parquet", states)
    check_file(C.FEATURES_OUTPUT_DIR / "05_plant_features.parquet", states)
    print("\n=== complete ===")


if __name__ == "__main__":
    main()
