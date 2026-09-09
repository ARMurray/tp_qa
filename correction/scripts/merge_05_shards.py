"""
merge_05_shards.py
==================
Concatenates the per-state output of 05_run_inference_array.slurm into the
two flat files 10_build_review_queue.py reads:

    data/inference_shards/state=XX/plant_summary.parquet      ->
    data/inference_shards/state=XX/stage2_candidates.parquet  ->
        data/inference/plant_summary.parquet
        data/inference/stage2_candidates.parquet

WHY A MERGE STEP EXISTS AT ALL
    10_build_review_queue.py reads those two paths by name (its main() does
    pd.read_parquet(inference_dir / "plant_summary.parquet")). Rather than
    teach 10 about shards, the array writes shards and this recombines them
    into exactly the layout 10 already expects. One less thing to keep in
    sync.

WHAT IT CHECKS
    - every state asked for actually produced a shard; a silently missing
      state is the failure mode that matters, since the merged file would
      look perfectly well-formed without it
    - no duplicate CWNS_IDs across shards (states are disjoint, so any
      duplicate means a task ran with the wrong state or a stale shard
      survived from an earlier run)
    - stage2_candidates shards can legitimately be EMPTY for a state where
      no flagged plant kept a candidate; that is not an error

Usage:
    python merge_05_shards.py --states "AL AR AZ ... WY"
    python merge_05_shards.py --states "..." --dry-run
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

SHARD_ROOT = C.DATA_DIR / "inference_shards"
OUT_DIR = C.DATA_DIR / "inference"

FILES = ["plant_summary.parquet", "stage2_candidates.parquet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True,
                    help="space- or comma-separated list the array was run for")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-missing", action="store_true",
                    help="merge anyway when a state produced no shard")
    args = ap.parse_args()

    states = [s for s in args.states.replace(",", " ").split() if s]

    print("=== merge_05_shards.py ===")
    print(f"shards : {SHARD_ROOT}")
    print(f"output : {OUT_DIR}")
    print(f"states : {len(states)}\n")

    if not SHARD_ROOT.exists():
        print(f"ERROR: {SHARD_ROOT} does not exist. Run the array first.")
        sys.exit(2)

    missing = [s for s in states if not (SHARD_ROOT / f"state={s}").exists()]
    if missing:
        print(f"WARNING: {len(missing)} state(s) have no shard directory: "
              f"{' '.join(missing)}")
        if not args.allow_missing:
            print("\nRefusing to merge an incomplete set -- the merged file would "
                  "look well-formed while silently missing these states. Re-run "
                  "the failed array tasks, or pass --allow-missing if the gap is "
                  "intentional.")
            sys.exit(1)

    for fname in FILES:
        print(f"--- {fname} ---")
        frames, empty, absent = [], [], []
        for st in states:
            p = SHARD_ROOT / f"state={st}" / fname
            if not p.exists():
                absent.append(st)
                continue
            df = pd.read_parquet(p)
            if df.empty:
                empty.append(st)
                continue
            frames.append(df)

        if absent:
            print(f"  no file for {len(absent)} state(s): {' '.join(absent)}")
        if empty:
            print(f"  empty for {len(empty)} state(s): {' '.join(empty)}")

        if not frames:
            print("  nothing to merge -- skipping\n")
            continue

        merged = pd.concat(frames, ignore_index=True)
        print(f"  {len(frames)} shard(s) -> {len(merged)} rows")

        if fname == "plant_summary.parquet" and "CWNS_ID" in merged.columns:
            dup = merged["CWNS_ID"].duplicated().sum()
            if dup:
                print(f"  WARNING: {dup} duplicate CWNS_ID row(s). States are "
                      f"disjoint, so this means a task ran with the wrong state "
                      f"or a stale shard survived. Investigate before trusting "
                      f"the review queue built from this.")

        if args.dry_run:
            print("  --dry-run: not written\n")
            continue

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / fname
        merged.to_parquet(out, index=False)
        print(f"  wrote {out}\n")

    print("=== complete ===")
    if not args.dry_run:
        print("Next: 10_build_review_queue.py --states <...> --round <n>")


if __name__ == "__main__":
    main()
