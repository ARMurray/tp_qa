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
Picks a seed layer, optionally merges a newer layer's curation onto it, and
writes the result as the first layer of the .gpkg. Row count and columns are
asserted identical to the seed afterwards.

THE SEED IS NOT SIMPLY THE NEWEST LAYER
---------------------------------------
A newer layer can be a REGRESSION. Exporting from GIS through a select or a
join silently drops columns, and the result still looks like a normal, more
recent layer.

The real master this was written against had exactly that:
CWNS_Locations_09012026 was newer than CWNS_Locations_08202026 but had lost
the Verified column entirely, lost 2,104 of 2,178 Original_Correct == "Yes"
values, and carried 3 duplicate CWNS_IDs. Seeding from it would have thrown
away months of verification. Nothing downstream would have reported anything
worse than smaller training counts.

So pick_seed_layer() walks newest-first and takes the first layer that
carries the required schema, saying loudly which newer ones it skipped.

--reconcile-from merges a skipped layer's real curation back on top, under
one rule: a non-null value may overwrite the base, a NULL never may. That
asymmetry is what makes it safe against a layer that dropped columns. Every
change is printed.

LAYER NAMING
------------
Normalized to YYYYMMDD. The existing master uses MMDDYYYY, which sorts
wrongly as a string (09012026 vs 08202026 happens to work; 01012027 would
not). This is a deliberate convention change, not a bug fix -- the old R
script's mdy() parsing handled MMDDYYYY correctly.

DO NOT SEED FROM THE R SCRIPT'S OUTPUT
--------------------------------------
The retired pull_reviews.R prototype wrote a gpkg to a DIFFERENT path
(correction/data/training/Updates.gpkg) whose corrections are wrong:
Corrected_X/Y was set to the REPORTED coordinate for every
candidate_correct verdict, so every "correction" in it points at the
location it was supposed to be correcting. Seed from the .gdb -- which the R
script only ever read -- and let close_round.py rebuild the verdicts
correctly from app.db.

The guard for this is narrow on purpose: it blocks only on a row claiming
the original was WRONG while its correction points at that same location.
A row found already-correct with Corrected == Original is consistent, and
three Hawaii plants in the real master look exactly like that.
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


def dated_layers(path: Path) -> list[tuple]:
    """[(date, name)] for every dated CWNS_Locations layer, oldest first."""
    import pyogrio
    names = [str(n) for n in pyogrio.list_layers(path)[:, 0]]
    dated = sorted((d, n) for d, n in ((parse_layer_date(x), x) for x in names)
                   if d is not None)
    if not dated:
        raise SystemExit(
            f"No {LAYER_PREFIX} dated layers in {path}.\nFound: {names}")
    return dated


def pick_seed_layer(path: Path) -> str:
    """Newest dated layer that actually carries the required schema.

    NOT simply the newest (2026-09-21). A newer layer can be a REGRESSION --
    exported from GIS through a select or join that silently dropped columns.
    One real instance: CWNS_Locations_09012026 was newer than
    CWNS_Locations_08202026 but had lost the Verified column entirely and
    2,104 of 2,178 Original_Correct == "Yes" values, which are the verified-
    correct labels this whole project exists to accumulate.

    Seeding the master from that would have thrown away months of
    verification, and nothing downstream would have reported anything worse
    than smaller training counts.

    So: walk newest-first, take the first layer that passes the schema check,
    and say loudly which newer ones were skipped and why.
    """
    import geopandas as gpd

    dated = dated_layers(path)
    print(f"  dated layers: {[n for _, n in dated]}")
    skipped = []
    for _, name in reversed(dated):
        cols = set(gpd.read_file(path, layer=name, rows=1).columns)
        missing = [c for c in REQUIRED_COLUMNS if c not in cols]
        if missing:
            skipped.append((name, missing))
            continue
        if skipped:
            print()
            for s_name, s_missing in skipped:
                print(f"  SKIPPED newer layer {s_name}: missing {s_missing}")
            print(f"  --> seeding from {name} instead.")
            print(f"      If {skipped[0][0]} contains curation work you need, "
                  f"pass --reconcile-from {skipped[0][0]} to merge its "
                  f"non-null values on top.")
        return name

    raise SystemExit(
        f"\nNo layer in {path} has the required schema.\n"
        + "\n".join(f"  {n}: missing {m}" for n, m in skipped))


RECONCILE_COLUMNS = ["Original_Correct", "Corrected", "Corrected_X",
                     "Corrected_Y", "How_Corrected"]


def reconcile(base, other, other_name: str):
    """Merge a schema-poor newer layer's curation onto the base, safely.

    THE RULE: a non-null value in `other` may overwrite the base. A NULL in
    `other` may never overwrite a non-null base value.

    That single asymmetry is what makes this safe to run against a layer that
    dropped columns. The regressed layer this was written for had lost 2,104
    of 2,178 Original_Correct == "Yes" values; a symmetric merge would have
    propagated those nulls and destroyed the labels. This one cannot.

    Geometry moves with the row when the newer layer's geometry differs --
    that is usually the whole point, since a manual move in GIS is how
    corrections get made here.

    Every change is printed. Five rows differing is a sane outcome; five
    hundred means the two layers disagree structurally and you should stop
    and look rather than merge.
    """
    import geopandas as gpd

    base = base.copy()
    base["CWNS_ID"] = base["CWNS_ID"].astype(str)
    other = other.drop_duplicates("CWNS_ID").copy()
    other["CWNS_ID"] = other["CWNS_ID"].astype(str)

    usable = [c for c in RECONCILE_COLUMNS if c in other.columns]
    dropped = [c for c in RECONCILE_COLUMNS if c not in other.columns]
    if dropped:
        print(f"  {other_name} has no {dropped} -- those stay as the base has them")

    o = other.set_index("CWNS_ID")
    changes = []
    geom_moves = 0

    for pos, cwns_id in enumerate(base["CWNS_ID"]):
        if cwns_id not in o.index:
            continue
        src = o.loc[cwns_id]
        for col in usable:
            nv = src[col]
            if pd.isna(nv):
                continue                      # never overwrite with a null
            ov = base.iloc[pos][col]
            if pd.isna(ov) or str(ov) != str(nv):
                changes.append((cwns_id, col, ov, nv))
                base.iat[pos, base.columns.get_loc(col)] = nv
        sg, bg = src.geometry, base.iloc[pos].geometry
        if sg is not None and bg is not None and not sg.equals_exact(bg, 1e-9):
            base.iat[pos, base.columns.get_loc(base.geometry.name)] = sg
            geom_moves += 1

    print(f"\n  reconciled {len({c[0] for c in changes}) + geom_moves} row(s) "
          f"from {other_name}: {len(changes)} field change(s), "
          f"{geom_moves} geometry move(s)")
    for cwns_id, col, ov, nv in changes[:40]:
        print(f"    {cwns_id}  {col}: {ov!r} -> {nv!r}")
    if len(changes) > 40:
        print(f"    ... and {len(changes) - 40} more")
    return base


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
                    help="source layer. Defaults to the newest dated layer "
                         "that carries the required schema -- not simply the "
                         "newest, see pick_seed_layer().")
    ap.add_argument("--reconcile-from", default=None,
                    help="merge this layer's non-null values on top of the "
                         "seed. For a newer layer that has real curation but "
                         "dropped columns. Nulls never overwrite.")
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
    layer = args.layer or pick_seed_layer(gdb)
    print(f"  using layer: {layer}")
    gdf = gpd.read_file(gdb, layer=layer)
    print(f"  {len(gdf)} rows, {len(gdf.columns)} columns")

    missing = [c for c in REQUIRED_COLUMNS if c not in gdf.columns]
    if missing:
        raise SystemExit(
            f"\nSource layer is missing expected column(s): {missing}\n"
            f"Wrong layer, or the schema changed. Nothing written.")

    n_dupe = gdf["CWNS_ID"].astype(str).duplicated().sum()
    if n_dupe:
        raise SystemExit(
            f"\n{n_dupe} duplicate CWNS_ID(s) in {layer}. The pipeline assumes "
            f"CWNS_ID is a unique key and build_training_bins.py raises on "
            f"this too.\nUsually an accidental double-append in GIS. Fix the "
            f"source layer, or pick another with --layer. Nothing written.")

    # Guard against seeding from the R prototype's broken output.
    #
    # The check is NARROW on purpose (corrected 2026-09-21). An earlier
    # version flagged any row with Corrected_X == Original_X, which is a
    # false positive: a row that was reviewed, found already correct
    # (Original_Correct == "Yes"), and had Corrected_X/Y set equal to record
    # that, is perfectly consistent. Three Hawaii plants in the real master
    # look exactly like that.
    #
    # The actual contradiction is a row claiming the original was WRONG
    # (Original_Correct == "No") while its correction points at that same
    # wrong location. That is what pull_reviews.R produced, for every
    # candidate_correct verdict it touched.
    ox = pd.to_numeric(gdf["Original_X"], errors="coerce")
    oy = pd.to_numeric(gdf["Original_Y"], errors="coerce")
    cx = pd.to_numeric(gdf["Corrected_X"], errors="coerce")
    cy = pd.to_numeric(gdf["Corrected_Y"], errors="coerce")
    same = (cx.notna() & ox.notna()
            & (cx - ox).abs().lt(1e-9) & (cy - oy).abs().lt(1e-9))
    contradiction = same & (gdf["Original_Correct"] == "No") & (gdf["Corrected"] == "Yes")
    if contradiction.any():
        ids = gdf.loc[contradiction, "CWNS_ID"].astype(str).tolist()
        raise SystemExit(
            f"\n{contradiction.sum()} row(s) say the reported location was "
            f"WRONG (Original_Correct='No', Corrected='Yes') while the "
            f"correction points at that same location:\n  {ids[:20]}\n"
            f"That is the signature of the retired pull_reviews.R bug -- see "
            f"this script's docstring. Nothing written.")
    benign = int(same.sum())
    if benign:
        print(f"  note: {benign} row(s) have Corrected == Original, all with "
              f"Original_Correct == 'Yes' (reported location confirmed "
              f"correct). Consistent -- not blocking.")

    n_verified = (gdf["Verified"] == "Yes").sum()
    n_corrected = (gdf["Corrected"] == "Yes").sum()
    print(f"  verified: {n_verified} | corrected: {n_corrected}")

    if args.reconcile_from:
        print(f"\nReconciling from {args.reconcile_from} ...")
        other = gpd.read_file(gdb, layer=args.reconcile_from)
        gdf = reconcile(gdf, other, args.reconcile_from)
        n_ver = (gdf["Verified"] == "Yes").sum()
        n_oc = (gdf["Original_Correct"] == "Yes").sum()
        print(f"  after reconcile -- verified: {n_ver} | "
              f"Original_Correct=='Yes': {n_oc}")

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
