"""
check_parcel_coverage.py
=========================
Which states can this pipeline actually run on?

    python check_parcel_coverage.py

Compares three things that are easy to assume agree and often do not:

    the TRAINING UNIVERSE   states with labelled plants (training_locations.gpkg)
    the PARCEL STORE        states present under PARCEL_BASE
    the NLCD OUTPUT         states 01a has actually produced

WHY THIS EXISTS
    01b and 01c resolve a plant's parcel with

        read_parquet('{PARCEL_BASE}/state=XX/*.parquet') JOIN ST_Intersects(...)

    If state=XX is missing or empty, that query matches nothing and every
    plant in the state lands in the run's "No parcel found" total. The job
    still exits 0. The log still looks like a run. The only symptom is a
    number in a summary line, and the natural reading of it -- "these plants
    have bad coordinates" -- is wrong.

    Confirmed 2026-09-23: a run was widened from a pilot state list to all 52
    states of the training universe on the assumption that the parcel store
    covered them. The training universe is derived from the master locations
    file and says nothing about which Regrid states were ever downloaded.

WHAT TO DO WITH THE OUTPUT
    A training state with no parcel data cannot contribute anything to
    01a/01b/01c/01e/02, so the honest options are to download that state's
    parcels or to leave it out of the state list. Running it anyway is not
    harmful -- it wastes an array task and inflates a counter -- but it makes
    every downstream count quietly wrong about coverage rather than about
    data quality.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def parcel_store_states() -> dict:
    """{state: n_parquet_files} for every state=XX dir under PARCEL_BASE."""
    if not C.PARCEL_BASE.exists():
        print(f"  PARCEL_BASE does not exist at all: {C.PARCEL_BASE}")
        return {}
    out = {}
    for d in sorted(C.PARCEL_BASE.glob("state=*")):
        if d.is_dir():
            out[d.name.split("=", 1)[1]] = len(list(d.glob("*.parquet")))
    return out


def training_states() -> pd.Series:
    """Plants per state in the training universe (classes + corrections)."""
    import geopandas as gpd
    if not C.TRAINING_GPKG.exists():
        print(f"  {C.TRAINING_GPKG} not found -- run build_training_bins first.")
        return pd.Series(dtype=int)
    ids = set()
    for layer in (C.TRAINING_LAYER_CLASSES, C.TRAINING_LAYER_CORRECTIONS):
        try:
            ids |= set(gpd.read_file(C.TRAINING_GPKG, layer=layer)["CWNS_ID"].astype(str))
        except Exception as e:
            print(f"  could not read layer {layer}: {e}")
    loc = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt",
                      dtype={"CWNS_ID": str}, encoding="latin1")
    loc = loc.drop_duplicates("CWNS_ID")
    return (loc[loc["CWNS_ID"].isin(ids)]["STATE_CODE"]
            .value_counts().sort_index())


def nlcd_states() -> set:
    if not C.NLCD_OUTPUT_DIR.exists():
        return set()
    out = set()
    for f in C.NLCD_OUTPUT_DIR.glob(f"nlcd_*_k{C.K_RINGS}.parquet"):
        parts = f.stem.split("_")
        if len(parts) >= 2:
            out.add(parts[1])
    return out


def main():
    print("=== check_parcel_coverage.py ===")
    print(f"PARCEL_BASE : {C.PARCEL_BASE}")
    print(f"TRAINING    : {C.TRAINING_GPKG}")
    print(f"NLCD (01a)  : {C.NLCD_OUTPUT_DIR}\n")

    parcels = parcel_store_states()
    train = training_states()
    nlcd = nlcd_states()

    print(f"parcel store : {len(parcels)} state(s)")
    print(f"training universe: {len(train)} state(s), {int(train.sum())} plants")
    print(f"01a output   : {len(nlcd)} state(s)\n")

    empty = sorted(s for s, n in parcels.items() if n == 0)
    if empty:
        print(f"  state dirs present but EMPTY ({len(empty)}): {empty}\n")

    missing = [s for s in train.index if s not in parcels or parcels.get(s, 0) == 0]
    if missing:
        n_plants = int(train[missing].sum())
        print(f"*** {len(missing)} training state(s) have NO parcel data ***")
        print(f"    {n_plants} plants ({n_plants / train.sum() * 100:.0f}% of the "
              f"training universe) can never resolve a parcel.")
        print(f"    Every one of them lands in 01b/01c's 'No parcel found' total.")
        print()
        for s in missing:
            print(f"      {s}: {train[s]:>5} plants")
        print()
        usable = [s for s in train.index if s not in missing]
        print("    States that CAN run, space-separated for --export=STATES:")
        print(f'      "{" ".join(usable)}"')
        print(f"    --array=0-{len(usable) - 1}")
    else:
        print("  every training state has parcel data")

    no_nlcd = [s for s in train.index if s not in nlcd and s not in missing]
    if no_nlcd:
        print(f"\n  {len(no_nlcd)} state(s) have parcels but no 01a output yet "
              f"(run 01a for them before 02): {no_nlcd}")

    extra = sorted(set(parcels) - set(train.index))
    if extra:
        print(f"\n  {len(extra)} state(s) in the parcel store with no training "
              f"plants -- harmless, just unused: {extra[:10]}"
              f"{' ...' if len(extra) > 10 else ''}")


if __name__ == "__main__":
    main()
