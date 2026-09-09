"""
inspect_stage2_columns.py
===========================
Quick, targeted look at 15_stage2_training.parquet's actual saved columns,
to trace where a "STATE_CODE_x" suffix is coming from (found in Stage 2
model feature importance, 2026-08-21) -- static code tracing didn't
converge on the source, so checking the real data directly.

Usage:
    python inspect_stage2_columns.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def main():
    path = C.FEATURES_OUTPUT_DIR / "15_stage2_training.parquet"
    print(f"Reading columns from: {path}\n")
    # Read only the schema/first few rows -- this file can be large (millions
    # of rows), no need to load it all just to inspect column names.
    df = pd.read_parquet(path, columns=None).head(5)

    all_cols = list(df.columns)
    print(f"Total columns: {len(all_cols)}\n")

    state_related = [c for c in all_cols if "state" in c.lower()]
    print(f"--- Columns containing 'state' (case-insensitive) ---")
    for c in state_related:
        print(f"  {c!r}  dtype={df[c].dtype}  sample={df[c].iloc[0] if len(df) else None}")

    suffixed = [c for c in all_cols if c.endswith("_x") or c.endswith("_y")]
    print(f"\n--- ALL suffixed columns (any _x/_y, not just state) ---")
    for c in suffixed:
        print(f"  {c!r}  dtype={df[c].dtype}  sample={df[c].iloc[0] if len(df) else None}")

    print(f"\n--- Full column list ---")
    for c in all_cols:
        print(f"  {c}")


if __name__ == "__main__":
    main()
