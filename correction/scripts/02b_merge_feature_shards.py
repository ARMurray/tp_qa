"""
02b_merge_feature_shards.py
============================
Concatenates the per-state output of 02_feature_engineering.slurm (the array)
into the flat files every downstream script reads by name:

    data/feature_shards/state=XX/05_plant_features.parquet   ->
    data/feature_shards/state=XX/14_stage1_training.parquet  ->
    data/feature_shards/state=XX/15_stage2_training.parquet  ->
            data/features/05_plant_features.parquet
            data/features/14_stage1_training.parquet
            data/features/15_stage2_training.parquet

    data/features/10_parcel_features_by_state/state=XX.parquet ->
            data/features/10_parcel_features.parquet

WHY A MERGE STEP EXISTS AT ALL
    Same reasoning as merge_05_shards.py: 03/04/05/06/06b/10 all open these
    paths by name. Rather than teach six scripts about shards, the array
    writes shards and this recombines them into the layout they already
    expect. One less thing to keep in sync.

    Before this existed, running 02 as an array silently produced feature
    tables for ONE state -- whichever task happened to finish last -- because
    all 52 tasks wrote the same four filenames. Nothing errored. 03 and 04
    would train happily on a fiftieth of the data.

WHY CONCATENATION IS VALID
    Checked rather than assumed, because "just concatenate the shards" is only
    correct if no per-state result depends on any other state:

      - build_stage1_training groups by STATE_CODE and calls
        point_in_parcel_lookup(con, state, ...) per state; every merge after
        that is row-wise on CWNS_ID / ll_uuid.
      - build_stage2_training reads each state's own nlcd_{state}_k18 file for
        its candidate universe, then computes distance / label / ring per
        plant. A plant's candidates already come only from its own state's
        file, so sharding loses nothing that a single national job would have
        found.
      - add_name_matching is row-wise: no fit, no global statistic.
      - Nothing anywhere normalises, ranks or aggregates across plants.

    So the union of per-state tables is the same table a national run would
    have produced. If any of the above ever stops being true, this script
    stops being correct, and the failure will be silent.

10_parcel_features IS DIFFERENT
    It already had a per-state shard directory before the array existed
    (10_parcel_features_by_state/, a deliberate persistent CACHE), so there is
    no second copy under feature_shards/. This rebuilds the flat file from
    that cache, filtered to --states EXPLICITLY.

    Do not be tempted to glob the cache directory instead. It accumulates one
    file per state ever processed, across every run -- a 3-state run globbing
    it produced 26.65M rows against 4.87M real candidate parcels on
    2026-08-25, a 5.47x inflation, because 51 leftover files from an earlier
    nationwide run were sitting there. Those files are legitimate cached work;
    the fix is to filter, not to delete them.

WHAT IT CHECKS
    - every state asked for actually produced a shard. A silently missing
      state is the failure mode that matters here, because the merged file
      looks perfectly well formed without it.
    - no duplicate CWNS_IDs in the plant-keyed tables (05, 14). States are
      disjoint, so a duplicate means a task ran with the wrong state or a
      stale shard survived from an earlier run. 15 is candidate-keyed and
      legitimately has many rows per plant, so it is not checked this way.
    - columns present in some shards but not others, which would otherwise
      become silent NaN after the concat.
    - an EMPTY shard is not an error: a state with no corrections produces
      zero Stage 2 rows (AK, for one).

Usage:
    python 02b_merge_feature_shards.py --states "AK AL AR ... WY"
    python 02b_merge_feature_shards.py --states "OH PA" --dry-run
    python 02b_merge_feature_shards.py --all-present      # whatever is there
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

SHARD_ROOT = C.FEATURE_SHARD_DIR
OUT_DIR = C.FEATURES_OUTPUT_DIR
PARCEL_CACHE = OUT_DIR / "10_parcel_features_by_state"

# name -> the column that must be unique across the merged table, or None
SHARD_FILES = {
    "05_plant_features.parquet": "CWNS_ID",
    "14_stage1_training.parquet": "CWNS_ID",
    "15_stage2_training.parquet": None,   # one row per candidate parcel
}


def read_one(path: Path) -> pd.DataFrame:
    """Read a single shard, normalising dtypes that have broken concats here.

    Deliberately one file at a time rather than pd.read_parquet(directory):
    a whole-directory read makes pyarrow resolve ONE schema across every file
    up front, which blows up when a column that is all-null in one state
    (inferred as pyarrow 'null') meets a real string in another. Reading
    individually and letting pandas align on column names sidesteps it.
    """
    df = pd.read_parquet(path)
    for col in df.select_dtypes(include=["category"]).columns:
        df[col] = df[col].astype(str)
    return df


def discover_states() -> list[str]:
    if not SHARD_ROOT.exists():
        return []
    return sorted(d.name.split("=", 1)[1] for d in SHARD_ROOT.iterdir()
                  if d.is_dir() and d.name.startswith("state="))


def merge_file(name: str, unique_key: str | None, states: list[str]):
    """Build and validate one merged table. Returns (ok, dataframe).

    Deliberately does NOT write. Every table is validated before any is
    written -- see main(). A run that wrote 10_parcel_features and then
    refused on 14_stage1_training would leave the flat files describing two
    different runs, and nothing downstream would notice.
    """
    print(f"\n--- {name} ---")
    paths = [(st, SHARD_ROOT / f"state={st}" / name) for st in states]
    missing = [st for st, p in paths if not p.exists()]
    present = [(st, p) for st, p in paths if p.exists()]

    if missing:
        print(f"  MISSING shard for {len(missing)} state(s): {missing}")
        print("  Refusing to write a merged file that silently omits them.")
        print("  Re-run those array tasks, or drop them from --states if they")
        print("  genuinely have no data.")
        return False, None

    frames, empties, all_cols = [], [], []
    for st, p in present:
        df = read_one(p)
        all_cols.append((st, set(df.columns)))
        if df.empty:
            empties.append(st)
        else:
            frames.append(df)

    if empties:
        print(f"  {len(empties)} state(s) produced zero rows (fine -- e.g. a "
              f"state with no corrections has no Stage 2 candidates): {empties}")

    # Columns not present everywhere would become NaN after the concat, which
    # looks like real missing data downstream rather than a shape mismatch.
    union = set().union(*[c for _, c in all_cols]) if all_cols else set()
    ragged = {}
    for st, cols in all_cols:
        gap = union - cols
        if gap:
            ragged[st] = sorted(gap)
    if ragged:
        print(f"  WARNING: {len(ragged)} state(s) are missing columns other "
              f"states have. They will be NaN in the merged table:")
        for st, gap in list(ragged.items())[:5]:
            print(f"    {st}: {gap[:6]}{' ...' if len(gap) > 6 else ''}")

    if not frames:
        print("  every shard was empty -- writing an empty table with the "
              "shard schema so downstream reads still find the columns")
        merged = read_one(present[0][1]) if present else pd.DataFrame()
    else:
        merged = pd.concat(frames, ignore_index=True)

    print(f"  {len(present)} shard(s) -> {len(merged):,} rows, "
          f"{len(merged.columns)} columns")

    if unique_key and unique_key in merged.columns and len(merged):
        dupes = merged[unique_key].duplicated().sum()
        if dupes:
            ex = merged.loc[merged[unique_key].duplicated(keep=False), unique_key]
            print(f"  ERROR: {dupes} duplicate {unique_key} value(s). States are "
                  f"disjoint, so this means a task ran with the wrong state or a "
                  f"stale shard survived. Examples: {sorted(set(ex))[:5]}")
            return False, None

    return True, merged


def merge_parcel_features(states: list[str]):
    """Build and validate 10_parcel_features. Returns (ok, dataframe)."""
    name = "10_parcel_features.parquet"
    print(f"\n--- {name} (from {PARCEL_CACHE.name}) ---")
    if not PARCEL_CACHE.exists():
        print(f"  ERROR: {PARCEL_CACHE} does not exist -- did 02 run at all?")
        return False, None

    paths = [(st, PARCEL_CACHE / f"state={st}.parquet") for st in states]
    missing = [st for st, p in paths if not p.exists()]
    present = [(st, p) for st, p in paths if p.exists()]

    if missing:
        print(f"  {len(missing)} requested state(s) have no cached parcel "
              f"features: {missing}")
        print("  Usually means no 01a output for them, or zero candidate "
              "parcels. They will be absent from the combined table.")

    n_cached_total = len(list(PARCEL_CACHE.glob("state=*.parquet")))
    if n_cached_total > len(present):
        print(f"  cache holds {n_cached_total} state file(s) from earlier runs; "
              f"using only this run's {len(present)}, as it must")

    frames = [read_one(p) for _, p in present]
    if not frames:
        print("  ERROR: no parcel feature files at all for the requested states")
        return False, None
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset="ll_uuid")
    print(f"  {len(present)} state(s) -> {len(merged):,} unique parcels")
    return True, merged


def main():
    ap = argparse.ArgumentParser(
        description="Union 02's per-state shards into the flat feature tables.")
    ap.add_argument("--states", type=str, default=None,
                    help='states the array covered, space or comma separated, '
                         'e.g. "OH PA WI". Every one must have a shard.')
    ap.add_argument("--all-present", action="store_true",
                    help="merge whatever shards exist instead of naming them. "
                         "Convenient, but it cannot tell a state that failed "
                         "from one you never asked for -- prefer --states.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=== 02b_merge_feature_shards.py ===")
    print(f"Shards: {SHARD_ROOT}")
    print(f"Output: {OUT_DIR}")

    if args.all_present:
        states = discover_states()
        print(f"\n--all-present: found {len(states)} shard(s): {states}")
    elif args.states:
        states = [s for s in args.states.replace(",", " ").split() if s]
    else:
        raise SystemExit("Pass --states or --all-present.")

    if not states:
        raise SystemExit(f"No shards found under {SHARD_ROOT}. Did the array run?")

    print(f"\nMerging {len(states)} state(s)")

    # ---- PHASE 1: build and validate EVERY table, writing nothing ----
    #
    # Both helpers return (ok, dataframe) and deliberately do not write. A
    # run that wrote 10_parcel_features and then refused on
    # 14_stage1_training would leave the flat files describing two different
    # runs, and nothing downstream checks for that.
    #
    # NOTE (2026-09-23): the original test here was `if not merge_file(...)`,
    # which is ALWAYS False -- a 2-tuple is truthy however the merge went.
    # It also passed args.dry_run to helpers that take no such parameter, so
    # this raised TypeError before it could do damage. Had it not, the run
    # would have printed 'merge complete' having written nothing at all.
    # Unpack the result explicitly; never truth-test it.
    built: dict[str, pd.DataFrame] = {}
    ok = True
    for name, key in SHARD_FILES.items():
        good, df = merge_file(name, key, states)
        if good:
            built[name] = df
        else:
            ok = False

    good, parcels = merge_parcel_features(states)
    if good:
        built['10_parcel_features.parquet'] = parcels
    else:
        ok = False

    print()
    print('=' * 60)

    if not ok:
        print('=== merge FAILED -- NOTHING WAS WRITTEN ===')
        print('The existing flat files are untouched, so they are still')
        print('whatever the last good run left. Fix the shards and re-run.')
        print('Nothing downstream should run until this is clean: no')
        print('training script checks whether its inputs cover every state.')
        sys.exit(1)

    if args.dry_run:
        print('=== --dry-run: validated, wrote nothing ===')
        for name, df in built.items():
            print(f'  would write {name:<34} {len(df):>9,} rows, '
                  f'{len(df.columns)} cols')
        return

    # ---- PHASE 2: every table validated, so write them all ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in built.items():
        dest = OUT_DIR / name
        df.to_parquet(dest, index=False)
        print(f'  wrote {dest.name:<34} {len(df):>9,} rows, '
              f'{len(df.columns)} cols, {dest.stat().st_size / 1e6:.1f} MB')

    print()
    print('=== merge complete ===')
    print('NEXT: sbatch 03_train_stage1.slurm / 04_train_stage2.slurm')
    print('      sbatch 06_build_stage2b_training.slurm  -> 07_train_stage2b.slurm')
    print('      sbatch 06b_build_rerank_training.slurm -> 07b_train_rerank.slurm')


if __name__ == "__main__":
    main()
