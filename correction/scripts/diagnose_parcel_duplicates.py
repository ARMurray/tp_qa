"""
diagnose_parcel_duplicates.py
==============================
One-off diagnostic for the 5.47x row inflation seen in 02_feature_engineering.py's
"Parcel features combined" step on the OH/MS/DE full-universe pilot (2026-08-25):
26,652,201 output rows against 4,873,248 input candidate parcels from 01a.

fetch_parcel_attrs() in 02_feature_engineering.py queries PARCEL_BASE by ll_uuid
and does not deduplicate the result. If PARCEL_BASE has more than one row per
ll_uuid, the left-merge against nlcd (which IS deduplicated to one row per
ll_uuid) fans every duplicate out into extra output rows. This script checks
whether that's actually true, at what rate, and what the duplicate rows look
like, before committing to a fix.

Usage:
    python diagnose_parcel_duplicates.py --states OH,MS,DE
"""
import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def check_state(con, state: str) -> dict:
    path = f"{C.PARCEL_BASE.as_posix()}/state={state}/*.parquet"
    print(f"\n--- {state} ---")
    print(f"  Checking: {path}")

    total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]
    distinct = con.execute(
        f"SELECT COUNT(DISTINCT ll_uuid) FROM read_parquet('{path}')").fetchone()[0]
    ratio = total / distinct if distinct else float("nan")
    print(f"  total rows: {total}  distinct ll_uuid: {distinct}  ratio: {ratio:.3f}")

    dupes = con.execute(f"""
        SELECT ll_uuid, COUNT(*) n
        FROM read_parquet('{path}')
        GROUP BY ll_uuid
        HAVING COUNT(*) > 1
        ORDER BY n DESC
        LIMIT 5
    """).df()
    print(f"  Top duplicated ll_uuid values (up to 5):")
    print(dupes.to_string(index=False) if len(dupes) else "    none found")

    if len(dupes):
        uuid = dupes.iloc[0]["ll_uuid"]
        sample = con.execute(f"""
            SELECT * FROM read_parquet('{path}') WHERE ll_uuid = '{uuid}'
        """).df()
        print(f"\n  Full row content for the most-duplicated ll_uuid ({uuid}), "
              f"{len(sample)} rows, transposed so columns are readable:")
        print(sample.T.to_string())

    return dict(state=state, total=total, distinct=distinct, ratio=ratio,
                n_duplicated_uuids=len(con.execute(f"""
                    SELECT ll_uuid FROM read_parquet('{path}')
                    GROUP BY ll_uuid HAVING COUNT(*) > 1
                """).df()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=str, required=True,
                     help="comma-separated, e.g. --states OH,MS,DE")
    args = ap.parse_args()
    states = [s.strip() for s in args.states.split(",")]

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; SET enable_geoparquet_conversion = false;")

    print("=== diagnose_parcel_duplicates.py ===")
    print(f"PARCEL_BASE: {C.PARCEL_BASE}")

    summary = [check_state(con, st) for st in states]

    print("\n=== Summary across all states ===")
    print(pd.DataFrame(summary).to_string(index=False))
    con.close()
    print("\n=== complete ===")


if __name__ == "__main__":
    main()
