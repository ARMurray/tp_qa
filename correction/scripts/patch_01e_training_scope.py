"""
patch_01e_training_scope.py
===========================
Adds --training-corrections to 01e_run_od_candidates.py.

WHY
    01e currently offers two scopes: the holdout bins (default) or --all.
    Building re-rank training data needs a third: the corrections layer
    MINUS the holdout -- the ~266 plants with a known true parcel that the
    models are allowed to train on.

    Without it the only way to reach those plants is --all, which for a
    state like OH means fetching every flagged plant's 20 candidates
    (~20,000 parcels, hours) to obtain the 88 corrections' worth (~1,760).

Run once from $SCRIPTS:
    python patch_01e_training_scope.py --check
    python patch_01e_training_scope.py

Backs up first, verifies syntax, reverts itself if the result won't compile.
Idempotent -- re-running is a no-op.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "01e_run_od_candidates.py"

EDITS = [
    dict(
        name="1. restrict_to_training_corrections()",
        marker="def restrict_to_training_corrections",
        old='''def load_parcel_geoms(con, state: str, uuids: list[str]) -> pd.DataFrame:''',
        new='''def restrict_to_training_corrections(cands: pd.DataFrame) -> pd.DataFrame:
    """Candidates belonging to corrections-layer plants that are NOT in the
    holdout -- the plants a re-ranker is allowed to train on.

    The corrections layer is the only source of a known true parcel for
    training: classes-layer 'Correct' plants have their true location at the
    reported point, which Stage 2a excludes from the candidate pool by
    construction, so they can never contribute a positive row here.
    """
    import geopandas as gpd
    from holdout import MANIFEST_PATH

    corr = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CORRECTIONS)
    corr_ids = set(corr["CWNS_ID"].astype(str))
    print(f"  Corrections layer: {len(corr_ids)} plant(s)")

    if MANIFEST_PATH.exists():
        man = pd.read_parquet(MANIFEST_PATH)
        held = set(man["CWNS_ID"].astype(str))
        corr_ids -= held
        print(f"  Excluding {len(held)} holdout plant(s) -> {len(corr_ids)} trainable")
    else:
        print("  WARNING: no holdout manifest found. Proceeding with ALL "
              "corrections -- if a holdout is created later, anything trained "
              "on this data is contaminated.")

    out = cands[cands["CWNS_ID"].isin(corr_ids)].copy()
    print(f"  Candidates for those plants: {len(out)} "
          f"across {out['CWNS_ID'].nunique()} plant(s)")
    n_missing = len(corr_ids) - out["CWNS_ID"].nunique()
    if n_missing > 0:
        print(f"  NOTE: {n_missing} trainable correction(s) have no Stage 2a "
              f"candidates at all -- Stage 1 did not flag them, or K_RINGS "
              f"produced nothing. They cannot contribute a row either way.")
    return out


def load_parcel_geoms(con, state: str, uuids: list[str]) -> pd.DataFrame:''',
    ),
    dict(
        name="2. --training-corrections argument",
        marker='"--training-corrections"',
        old='''    ap.add_argument("--holdout-bins", default="corrections",''',
        new='''    ap.add_argument("--training-corrections", action="store_true",
                    help="corrections-layer plants MINUS the holdout -- the "
                         "scope for building re-rank training data. Mutually "
                         "exclusive with --all.")
    ap.add_argument("--holdout-bins", default="corrections",''',
    ),
    dict(
        name="3. dispatch on the new scope",
        marker="args.training_corrections:",
        old='''    if not args.all:
        bins = [b.strip() for b in args.holdout_bins.split(",")]
        cands = restrict_to_holdout(cands, bins)
    else:
        print("  --all: no holdout restriction")''',
        new='''    if args.all and args.training_corrections:
        print("ERROR: --all and --training-corrections are mutually exclusive.")
        sys.exit(2)
    if args.training_corrections:
        cands = restrict_to_training_corrections(cands)
    elif not args.all:
        bins = [b.strip() for b in args.holdout_bins.split(",")]
        cands = restrict_to_holdout(cands, bins)
    else:
        print("  --all: no scope restriction")''',
    ),
    dict(
        name="4. separate output root for the training scope",
        marker="od_features_candidates_train",
        old='''    out_root = C.DATA_DIR / "od_features_candidates"''',
        new='''    # Training-scope output goes to its own root. Keeping it apart from the
    # holdout's means a later --no-resume rebuild of one cannot quietly
    # discard the other, and makes it obvious at a glance which rows a model
    # was allowed to see.
    out_root = (C.DATA_DIR / "od_features_candidates_train"
                if args.training_corrections
                else C.DATA_DIR / "od_features_candidates")''',
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--file", default=str(TARGET))
    args = ap.parse_args()

    path = Path(args.file)
    print("=== patch_01e_training_scope.py ===")
    print(f"target: {path}\n")
    if not path.exists():
        print("ERROR: file not found.")
        sys.exit(2)

    src = path.read_text()
    applied, already, failed = [], [], []
    for e in EDITS:
        if e["marker"] in src:
            already.append(e["name"])
            continue
        n = src.count(e["old"])
        if n != 1:
            failed.append((e["name"], f"anchor found {n} times, expected 1"))
            continue
        src = src.replace(e["old"], e["new"])
        applied.append(e["name"])

    for n in already:
        print(f"  SKIP    {n} (already applied)")
    for n in applied:
        print(f"  {'WOULD  ' if args.check else 'APPLIED'} {n}")
    for n, why in failed:
        print(f"  FAILED  {n} -- {why}")

    if failed:
        print("\nAnchors did not match. Nothing written.")
        sys.exit(1)
    if not applied:
        print("\nNothing to do.")
        sys.exit(0)
    if args.check:
        print("\n--check: nothing written.")
        sys.exit(0)

    backup = path.with_suffix(f".py.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(path, backup)
    path.write_text(src)
    print(f"\nBackup: {backup.name}")

    import py_compile
    try:
        py_compile.compile(str(path), doraise=True)
        print("Syntax: OK")
    except py_compile.PyCompileError as err:
        print(f"Syntax: FAILED -- {err}")
        shutil.copy2(backup, path)
        print("Reverted. Nothing changed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
