"""
check_location_type.py
========================
Two of the pipeline's CWNS reads (PHYSICAL_LOCATION.txt, POPULATION_WASTEWATER.txt)
select a single row per CWNS_ID via drop_duplicates(subset="CWNS_ID") without
filtering on LOCATION_TYPE / POPULATION_TYPE / INFRASTRUCTURE_TYPE first --
whichever row appears first in the file silently wins. If a plant can have more
than one row of either table (e.g. physical vs mailing location), this could
mean the reported coordinate the whole pipeline evaluates -- or pop_served, a
live Stage 1/2a feature -- is sometimes taken from the wrong row.

This just reports whether that's a real risk or not. No fix here.

Usage:
    python check_location_type.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def check_table(path: Path, type_cols: list[str], id_col: str = "CWNS_ID"):
    print(f"\n=== {path.name} ===")
    if not path.exists():
        print(f"  MISSING: {path}")
        return
    df = pd.read_csv(path, dtype=str, encoding="latin1")
    print(f"  {len(df)} rows, {df[id_col].nunique()} distinct {id_col}")

    dupe_counts = df.groupby(id_col).size()
    n_dupes = int((dupe_counts > 1).sum())
    print(f"  {id_col}s with >1 row: {n_dupes} "
          f"({100 * n_dupes / df[id_col].nunique():.2f}%)")

    for col in type_cols:
        if col not in df.columns:
            print(f"  column '{col}' not present")
            continue
        print(f"\n  {col} value counts:")
        print(df[col].value_counts(dropna=False).to_string())

    if n_dupes:
        print(f"\n  Sample of {id_col}s with multiple rows (up to 5):")
        dupe_ids = dupe_counts[dupe_counts > 1].index[:5]
        for did in dupe_ids:
            sub = df[df[id_col] == did]
            print(f"\n  {id_col}={did} ({len(sub)} rows):")
            show_cols = [id_col] + [c for c in type_cols if c in df.columns]
            print(sub[show_cols].to_string(index=False))


def main():
    print("=== check_location_type.py ===")

    check_table(C.CWNS_DIR / "PHYSICAL_LOCATION.txt",
                type_cols=["LOCATION_TYPE"])

    check_table(C.CWNS_DIR / "POPULATION_WASTEWATER.txt",
                type_cols=["POPULATION_TYPE", "INFRASTRUCTURE_TYPE"])

    print("\n=== complete ===")
    print("\nIf either table shows real duplicate CWNS_IDs across genuinely")
    print("different type values, load_treatment_plants() (01a/01b/05) and")
    print("build_population_features() (02) need an explicit filter before")
    print("drop_duplicates(), not just drop_duplicates() alone.")


if __name__ == "__main__":
    main()
