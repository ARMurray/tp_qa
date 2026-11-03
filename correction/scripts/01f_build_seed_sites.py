"""
01f_build_seed_sites.py
=======================
Multi-source seeds -> seed sites, one state at a time (docs/PLAN_20261103_multi_source_seeds.md).

For each plant, every seed in plant_seeds.csv (CWNS point, CWNS geocode, ICIS-NPDES) is put on the parcel it
falls in, and seeds on the same parcel -- or on neighbouring parcels with one owner -- collapse into one
SEED SITE. Stage 1 judges each site once. See seed_sites.py for the rules.

Needs only the parcel store and plant_seeds.csv, so it can run any time after build_plant_seeds.py. It must
run BEFORE 01e --seed-sites (detection on each site's parcel) and 02 (Stage 1 table).

Output, per state:
    data/seed_sites/state=XX/seed_parcels.parquet    one row per seed: ll_uuid, match_kind, snap_m, site_id
    data/seed_sites/state=XX/seed_sites.parquet      one row per site

Plant universe is the same as 01a/01b: the labelled training plants, or every treatment plant with
--full-universe. Resume is per state (skips a state whose output exists), so a training-universe state needs
--no-resume before a full-universe run, exactly as for 01a.

Usage:
    python 01f_build_seed_sites.py --states OH,PA
    python 01f_build_seed_sites.py --states OH --full-universe --no-resume
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import seed_sites as SS

_spec = importlib.util.spec_from_file_location("extract_parcels", Path(__file__).resolve().parent / "01a_extract_parcels.py")
extract_parcels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract_parcels)


def state_dir(state: str) -> Path:
    return C.SEED_SITES_DIR / f"state={state}"


def process_state(con, state: str, plants_state: pd.DataFrame, seeds_all: pd.DataFrame) -> None:
    print(f"\n--- State {state}: {len(plants_state)} plants ---")
    seeds = seeds_all[seeds_all["CWNS_ID"].isin(set(plants_state["CWNS_ID"]))].copy()
    seeds["STATE_CODE"] = state
    print(f"  Seeds: {len(seeds)} for {seeds['CWNS_ID'].nunique()} plants "
          f"({len(plants_state) - seeds['CWNS_ID'].nunique()} plants have no seed)")
    if not len(seeds):
        return

    m = SS.match_seeds_to_parcels(con, state, seeds)
    print("  Seed -> parcel:", m["match_kind"].value_counts().to_dict())
    sp = seeds.merge(m, on="seed_id", how="left")
    parcels = SS.fetch_parcels(con, state, sp["ll_uuid"].dropna().unique())
    missing = set(sp["ll_uuid"].dropna()) - set(parcels.index)
    if missing:
        print(f"  WARNING: {len(missing)} matched parcel(s) not found when fetching geometry; their seeds are unparcelled")
        sp.loc[sp["ll_uuid"].isin(missing), ["ll_uuid", "match_kind"]] = [None, "none"]

    sp, sites = SS.build_sites(sp, parcels)
    d = state_dir(state)
    d.mkdir(parents=True, exist_ok=True)
    sp.drop(columns=["detail"], errors="ignore").to_parquet(d / "seed_parcels.parquet", index=False)
    sites.to_parquet(d / "seed_sites.parquet", index=False)
    per_plant = sites.groupby("CWNS_ID").size()
    print(f"  Sites: {len(sites)} for {sites['CWNS_ID'].nunique()} plants; "
          f"sites per plant {per_plant.value_counts().sort_index().to_dict()}")
    print(f"  Written: {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=str, default=None, help="comma-separated STATE_CODE filter")
    ap.add_argument("--full-universe", action="store_true",
                    help="every treatment plant, not just the labelled training plants")
    ap.add_argument("--no-resume", action="store_true", help="rebuild states whose output exists")
    args = ap.parse_args()

    C.ensure_dirs()
    states_filter = [s.strip() for s in args.states.split(",")] if args.states else None
    print("=== 01f_build_seed_sites.py ===")
    plants = extract_parcels.load_treatment_plants(states_filter, training_only=not args.full_universe)
    seeds_all = SS.load_seeds(plants["CWNS_ID"])
    print(f"Plants: {len(plants)}; seeds loaded: {len(seeds_all)}")

    con = C.duckdb_connect()
    for state, plants_state in plants.groupby("STATE_CODE"):
        if (state_dir(state) / "seed_sites.parquet").exists() and not args.no_resume:
            print(f"\n--- State {state}: output exists, skipping ({state_dir(state)}) ---")
            continue
        process_state(con, state, plants_state, seeds_all)
    con.close()
    print("\n=== 01f_build_seed_sites.py complete ===")
    print(f"Outputs in: {C.SEED_SITES_DIR}")


if __name__ == "__main__":
    main()
