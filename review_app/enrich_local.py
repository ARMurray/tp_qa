"""
enrich_local.py
================
Backfills pop_served / surface_water_discharge / requires_npdes / any_reuse
on the local app.db `plants` table, reading directly from local CWNS text
files instead of relying on 05_plant_features.parquet (which is built on the
HPC with training_only=True by default, so it only covers the training
subset of plants -- not every plant a later review round might queue).

Does NOT touch reviewed/plant_verdict/etc -- same safety pattern as
queue_loader.py's --update-metadata: only backfills the specific descriptive
columns listed below, never verdict state.

Does NOT touch geography (subdivision/place/county/is_rural) -- that's a
spatial join against the census gdb, deliberately out of scope for this
script (needs geopandas + the gdb; see conversation for why it was skipped).

SETUP REQUIRED: fill in CWNS_DIR below to point at the local folder
containing POPULATION_WASTEWATER.txt and DISCHARGES.csv (same folder as
FACILITIES.txt, per config.py's FACILITIES_PATH).

Usage:
    cd review_app
    .\.venv\Scripts\python.exe enrich_local.py                # all plants missing pop_served
    .\.venv\Scripts\python.exe enrich_local.py --round 2       # just round 2
    .\.venv\Scripts\python.exe enrich_local.py --force         # overwrite even where already set
"""
import argparse
import sqlite3
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# SETUP: adjust this to match your local CWNS data folder (same one
# FACILITIES.txt lives in, per config.py's FACILITIES_PATH)
# ---------------------------------------------------------------------------
CWNS_DIR = Path(
    r"C:\Users\AMURRA02\OneDrive - Environmental Protection Agency (EPA)"
    r"\Github\Sewersheds\Data"
)
POPULATION_PATH = CWNS_DIR / "POPULATION_WASTEWATER.txt"
DISCHARGES_PATH = CWNS_DIR / "DISCHARGES.csv"

DB_PATH = Path("data/app.db")


def load_population(ids: set) -> pd.DataFrame:
    df = pd.read_csv(POPULATION_PATH, dtype={"CWNS_ID": str}, encoding="latin1")
    df = df[df["CWNS_ID"].isin(ids)]
    df = df[["CWNS_ID", "TOTAL_RES_POPULATION_2022"]].rename(
        columns={"TOTAL_RES_POPULATION_2022": "pop_served"})
    return df.drop_duplicates(subset="CWNS_ID")


def load_discharge(ids: set) -> pd.DataFrame:
    df = pd.read_csv(
        DISCHARGES_PATH,
        dtype={"CWNS_ID": str},
        encoding="latin1",
    )
    df = df[df["CWNS_ID"].isin(ids)]

    def agg(g):
        return pd.Series(dict(
            surface_water_discharge=bool((g["DISCHARGE_TYPE"] == "Outfall To Surface Waters").any()),
            requires_npdes=bool(g["DISCHARGE_TYPE"].isin([
                "Outfall To Surface Waters", "Ocean Discharge",
                "CSO Discharge", "Overland Flow With Discharge"]).any()),
            any_reuse=bool(g["DISCHARGE_TYPE"].str.startswith("Reuse:", na=False).any()),
        ))

    return df.groupby("CWNS_ID").apply(agg, include_groups=False).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=None,
                     help="only enrich this review round. Default: all rounds.")
    ap.add_argument("--force", action="store_true",
                     help="overwrite even plants that already have pop_served set. "
                          "Default: only backfill where pop_served IS NULL.")
    args = ap.parse_args()

    for p, label in [(POPULATION_PATH, "POPULATION_WASTEWATER.txt"), (DISCHARGES_PATH, "DISCHARGES.csv")]:
        if not p.exists():
            raise FileNotFoundError(
                f"{label} not found at {p}\n"
                f"Edit CWNS_DIR at the top of this script to point at the right folder."
            )
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH.resolve()} not found -- run this from review_app/")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    where = []
    params = []
    if args.round is not None:
        where.append("review_round = ?")
        params.append(args.round)
    if not args.force:
        where.append("pop_served IS NULL")
    where_clause = f"WHERE {' AND '.join(where)}" if where else ""

    plants = pd.read_sql(f"SELECT cwns_id FROM plants {where_clause}", conn, params=params)
    ids = set(plants["cwns_id"])
    print(f"Enriching {len(ids)} plant(s)"
          f"{f' (round {args.round})' if args.round else ''}"
          f"{' (--force: overwriting existing values)' if args.force else ''}")

    if not ids:
        print("Nothing to do.")
        conn.close()
        return

    pop = load_population(ids)
    disch = load_discharge(ids)
    print(f"  Population matched: {len(pop)}/{len(ids)}")
    print(f"  Discharge matched:  {len(disch)}/{len(ids)}")

    merged = pd.DataFrame({"CWNS_ID": list(ids)}) \
        .merge(pop, on="CWNS_ID", how="left") \
        .merge(disch, on="CWNS_ID", how="left")

    n_updated = 0
    for _, r in merged.iterrows():
        conn.execute("""
            UPDATE plants SET
                pop_served = ?,
                surface_water_discharge = ?,
                requires_npdes = ?,
                any_reuse = ?
            WHERE cwns_id = ?
        """, (
            None if pd.isna(r.get("pop_served")) else float(r["pop_served"]),
            None if pd.isna(r.get("surface_water_discharge")) else int(bool(r["surface_water_discharge"])),
            None if pd.isna(r.get("requires_npdes")) else int(bool(r["requires_npdes"])),
            None if pd.isna(r.get("any_reuse")) else int(bool(r["any_reuse"])),
            r["CWNS_ID"],
        ))
        n_updated += 1

    conn.commit()
    conn.close()
    print(f"\nUpdated {n_updated} plant(s) in {DB_PATH}")
    print("Note: geography (subdivision/place/county/is_rural) was NOT touched -- "
          "that needs a census gdb spatial join, deliberately out of scope here.")


if __name__ == "__main__":
    main()
