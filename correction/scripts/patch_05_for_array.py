"""
patch_05_for_array.py
=====================
Applies three edits to 05_run_inference.py in place, so it can be run as a
per-state SLURM array instead of one national job that OOMs.

Run once, from $SCRIPTS:
    python patch_05_for_array.py            # patch
    python patch_05_for_array.py --check    # report only, change nothing

Writes 05_run_inference.py.bak-<timestamp> first. Idempotent: an edit that
is already applied is skipped, so re-running is harmless.

THE EDITS
  1. --out-dir. main() hardcodes DATA_DIR/inference for both outputs, so 48
     array tasks would race on the same two filenames and the last writer
     would win -- silently, since every task would report success.

  2. Filter parcel_features to the requested states. The national
     10_parcel_features.parquet (77.7M rows as of 2026-09-04) is loaded in
     full regardless of --states, so a CA-only run pays the whole national
     cost. It has a `state` column (already dropped at each merge site), so
     filtering is safe and cuts per-task memory by roughly the fraction of
     the country not being processed.

  3. Correct the "Plants with 0 candidates" message. It counts
     n_candidates == 0 over ALL scored plants, but unflagged plants get 0
     from the fillna, so the count includes plants that were never eligible
     for Stage 2. The CA run printed "353 flagged / 353 with >=1 candidate /
     141 with 0 candidates" where 353 + 141 = 494 = every plant. The
     parenthetical claims those 141 are "flagged but no parcel survived",
     which is wrong and is exactly the distinction 10_build_review_queue.py
     uses to split candidate_pick from confirm_reported.

WHY NOT JUST EDIT THE FILE
    Precise anchored replacements, verified before and after, beat
    hand-editing a 500-line file across a terminal that has been mangling
    multi-line pastes.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "05_run_inference.py"

EDITS = [
    dict(
        name="1. add --out-dir argument",
        marker='"--out-dir"',
        old='''    args = ap.parse_args()
    states = [s.strip() for s in args.states.split(",")]''',
        new='''    ap.add_argument("--out-dir", type=str, default=None,
                     help="write plant_summary.parquet/stage2_candidates.parquet "
                          "here instead of DATA_DIR/inference. Lets a per-state "
                          "SLURM array shard its output without 48 tasks racing "
                          "on the same two filenames.")
    args = ap.parse_args()
    states = [s.strip() for s in args.states.split(",")]''',
    ),
    dict(
        name="2. honour --out-dir when choosing the output directory",
        marker="args.out_dir else",
        old='''    inference_dir = C.DATA_DIR / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)''',
        new='''    inference_dir = Path(args.out_dir) if args.out_dir else C.DATA_DIR / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)''',
    ),
    dict(
        name="3. filter parcel_features to the requested states",
        marker="Parcel features narrowed",
        old='''    parcel_features = pd.read_parquet(C.FEATURES_OUTPUT_DIR / "10_parcel_features.parquet")''',
        new='''    parcel_features = pd.read_parquet(C.FEATURES_OUTPUT_DIR / "10_parcel_features.parquet")
    # Narrow to the requested states before anything else touches this frame.
    # It is the national table (77.7M rows on 2026-09-04) and was previously
    # loaded in full even for a single-state run, which is most of this
    # script's resident memory. The `state` column is already dropped at each
    # merge site downstream, so filtering on it here changes nothing else.
    if "state" in parcel_features.columns:
        n_before = len(parcel_features)
        parcel_features = parcel_features[parcel_features["state"].isin(states)]
        print(f"  Parcel features narrowed to {states}: "
              f"{n_before} -> {len(parcel_features)} rows")''',
    ),
    dict(
        name="4. fix the misleading zero-candidate count",
        marker="n_flagged_no_cands",
        old='''    print(f"  Plants with >=1 candidate: {int((stage1_out['n_candidates'] > 0).sum())}")
    print(f"  Plants with 0 candidates : {int((stage1_out['n_candidates'] == 0).sum())} "
          f"(flagged but no parcel survived K_RINGS + reported-parcel exclusion)")''',
        new='''    print(f"  Plants with >=1 candidate: {int((stage1_out['n_candidates'] > 0).sum())}")
    # Count zero-candidate plants among FLAGGED plants only. Unflagged plants
    # also carry n_candidates == 0 (from the fillna above) but were never
    # eligible for Stage 2, so counting them here overstated the figure and
    # mislabelled it -- the CA run reported 141, which was simply 494 total
    # minus 353 flagged. 10_build_review_queue.py splits candidate_pick from
    # confirm_reported on exactly this distinction.
    _flagged_mask = stage1_out["trigger_reason"] != "none"
    n_flagged_no_cands = int((_flagged_mask & (stage1_out["n_candidates"] == 0)).sum())
    n_unflagged = int((~_flagged_mask).sum())
    print(f"  Flagged w/ 0 candidates  : {n_flagged_no_cands} "
          f"(no parcel survived K_RINGS + reported-parcel exclusion)")
    print(f"  Not flagged (Stage 1 ok) : {n_unflagged} "
          f"(no candidates by design -- confirm_reported task in 10)")''',
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only")
    ap.add_argument("--file", default=str(TARGET))
    args = ap.parse_args()

    path = Path(args.file)
    print("=== patch_05_for_array.py ===")
    print(f"target: {path}\n")
    if not path.exists():
        print("ERROR: file not found.")
        sys.exit(2)

    src = original = path.read_text()
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

    for name in already:
        print(f"  SKIP    {name} (already applied)")
    for name in applied:
        print(f"  {'WOULD  ' if args.check else 'APPLIED'} {name}")
    for name, why in failed:
        print(f"  FAILED  {name} -- {why}")

    if failed:
        print("\nOne or more anchors did not match exactly once. The file has "
              "diverged from what this patcher expects; nothing was written. "
              "Apply those edits by hand rather than forcing it.")
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
    print(f"\nBackup : {backup.name}")
    print(f"Wrote  : {path.name}  ({len(original)} -> {len(src)} bytes)")

    import py_compile
    try:
        py_compile.compile(str(path), doraise=True)
        print("Syntax : OK")
    except py_compile.PyCompileError as err:
        print(f"Syntax : FAILED -- {err}")
        shutil.copy2(backup, path)
        print("Reverted from backup. Nothing changed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
