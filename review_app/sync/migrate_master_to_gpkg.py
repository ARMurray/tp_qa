"""
migrate_master_to_gpkg.py
=========================
ONE-TIME migration: seed the master Updates.gpkg from the existing
Updates.gdb.

    python -m sync.migrate_master_to_gpkg --dry-run
    python -m sync.migrate_master_to_gpkg

WHY THIS EXISTS
---------------
The master moved from .gdb to .gpkg on 2026-09-21 so the review loop can
write back into the file it reads (a gdb can be read by geopandas/pyogrio
but not written). Config now points at Updates.gpkg -- but nothing created
that file, so the first run of update_master_locations.py on a machine
holding only the .gdb fails with "Master gpkg not found."

This script is that missing step. Run it once per machine that holds the
master. After it succeeds, the .gdb is no longer read by anything.

WHAT IT DOES
------------
Reads the newest dated CWNS_Locations_YYYYMMDD layer from the .gdb and
writes it, unchanged, as the first layer of the .gpkg. No rows are added,
dropped, or edited -- this is a container change, not a data change. The
row count and every column are asserted identical afterwards.

The layer name is normalized to YYYYMMDD if the source used the legacy
MMDDYYYY spelling, so that "newest layer" stays a sortable question from
here on.

DO NOT SEED FROM THE R SCRIPT'S OUTPUT
--------------------------------------
The retired pull_reviews.R prototype wrote a gpkg to a DIFFERENT path
(correction/data/training/Updates.gpkg) whose corrections are wrong:
Corrected_X/Y was set to the REPORTED coordinate for every
candidate_correct verdict, so every "correction" in it points at the
location it was supposed to be correcting. If that file exists, do not use
it as the seed and do not merge it in. Seed from the .gdb -- which the R
script only ever read -- and let close_round.py rebuild the verdicts
correctly from app.db.

This script refuses to read a source whose newest layer already contains
rows with Corrected_X == Original_X, as a guard against exactly that.
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from sync.update_master_locations import (LAYER_PREFIX, LAYER_DATE_FMT,
                                          REPORTED_CRS, parse_layer_date)

REQUIRED_COLUMNS = [
    "CWNS_ID", "Verified", "Original_Correct", "Corrected",
    "Original_X", "Original_Y", "Corrected_X", "Corrected_Y",
]


def newest_layer(path: Path) -> str:
    import pyogrio
    names = [str(n) for n in pyogrio.list_layers(path)[:, 0]]
    dated = [(parse_layer_date(n), n) for n in names]
    dated = [(d, n) for d, n in dated if d is not None]
    if not dated:
        raise SystemExit(
            f"No {LAYER_PREFIX}YYYYMMDD layers in {path}.\nFound: {names}")
    dated.sort()
    print(f"  layers found: {names}")
    return dated[-1][1]


def normalized_name(layer: str) -> str:
    """CWNS_Locations_MMDDYYYY -> CWNS_Locations_YYYYMMDD, else unchanged."""
    d = parse_layer_date(layer)
    if d is None:
        return layer
    return f"{LAYER_PREFIX}{d.strftime(LAYER_DATE_FMT)}"


def main():
    ap = argparse.ArgumentParser(
        description="One-time seed of the master gpkg from the existing gdb.")
    ap.add_argument("--gdb", default=None,
                    help="source .gdb. Defaults to MASTER_GPKG's sibling "
                         "Updates.gdb.")
    ap.add_argument("--layer", default=None,
                    help="source layer. Defaults to the newest dated one.")
    ap.add_argument("--out", default=None,
                    help="destination .gpkg. Defaults to config.MASTER_GPKG.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import geopandas as gpd

    out = Path(args.out) if args.out else C.MASTER_GPKG
    gdb = Path(args.gdb) if args.gdb else out.with_suffix(".gdb")

    print("=== migrate_master_to_gpkg.py ===")
    print(f"Source: {gdb}")
    print(f"Dest:   {out}")

    if not gdb.exists():
        raise SystemExit(
            f"\nSource not found: {gdb}\n"
            f"Pass --gdb with the real path to your Updates.gdb.")

    if out.exists():
        raise SystemExit(
            f"\n{out} already exists -- refusing to overwrite the master.\n"
            f"If this machine is already migrated, you want close_round.py, "
            f"not this script.\nIf the existing file is a bad seed (see this "
            f"script's docstring), move it aside first and re-run.")

    print("\nReading source ...")
    layer = args.layer or newest_layer(gdb)
    print(f"  using layer: {layer}")
    gdf = gpd.read_file(gdb, layer=layer)
    print(f"  {len(gdf)} rows, {len(gdf.columns)} columns")

    missing = [c for c in REQUIRED_COLUMNS if c not in gdf.columns]
    if missing:
        raise SystemExit(
            f"\nSource layer is missing expected column(s): {missing}\n"
            f"Wrong layer, or the schema changed. Nothing written.")

    # Guard against seeding from the R prototype's broken output.
    ox = pd.to_numeric(gdf["Original_X"], errors="coerce")
    cx = pd.to_numeric(gdf["Corrected_X"], errors="coerce")
    bad = ((cx.notna()) & (ox.notna()) & ((cx - ox).abs() < 1e-9)).sum()
    if bad:
        raise SystemExit(
            f"\n{bad} row(s) have Corrected_X == Original_X -- a 'correction' "
            f"pointing at the location it corrects.\nThis is the signature of "
            f"the retired pull_reviews.R bug; see this script's docstring.\n"
            f"Do not seed the master from this layer. Nothing written.")

    n_verified = (gdf["Verified"] == "Yes").sum()
    n_corrected = (gdf["Corrected"] == "Yes").sum()
    print(f"  verified: {n_verified} | corrected: {n_corrected}")

    dest_layer = normalized_name(layer)
    if dest_layer != layer:
        print(f"\n  normalizing layer name: {layer} -> {dest_layer}")
        print(f"  (YYYYMMDD so 'newest layer' stays sortable)")

    if gdf.crs is None:
        print(f"\n  WARNING: source has no CRS -- assuming {REPORTED_CRS}, "
              f"which is what the rest of the pipeline assumes for "
              f"Original_X/Y. Check this if coordinates look wrong later.")
        gdf = gdf.set_crs(REPORTED_CRS)

    if args.dry_run:
        print(f"\n--dry-run: would write layer {dest_layer} to {out}")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {dest_layer} -> {out} ...")
    gdf.to_file(out, layer=dest_layer, driver="GPKG")

    # Assert the container change changed nothing.
    check = gpd.read_file(out, layer=dest_layer)
    assert len(check) == len(gdf), (
        f"row count changed on write: {len(gdf)} -> {len(check)}")
    missing_after = set(gdf.columns) - set(check.columns)
    assert not missing_after, f"columns lost on write: {sorted(missing_after)}"
    print(f"  verified: {len(check)} rows, {len(check.columns)} columns, "
          f"nothing lost")

    print("\n=== migration complete ===")
    print(f"The .gdb is no longer read by anything. Keep it as a backup.")
    print(f"\nNEXT:")
    print(f"  cd review_app")
    print(f"  python -m sync.close_round --round 1 --skip-tiles --dry-run")


if __name__ == "__main__":
    main()
