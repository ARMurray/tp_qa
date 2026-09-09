"""
check_od_freshness.py
=====================
Preflight for 02_feature_engineering.py. Fails loudly if any object-
detection partition was written BEFORE the currently deployed best.pt --
i.e. by a different model.

THE PROBLEM THIS SOLVES
    01b resumes per plant and writes into data/od_features/{sub}/state=XX/.
    02 has no resume: it rebuilds the national feature table from whatever
    partitions it finds. Swap best.pt and re-run 01b for 48 states, and any
    state NOT in that submission keeps its old-model detections. 02 then
    silently blends two models into one table. Nothing errors, nothing in
    the logs mentions it, and the resulting Stage 2a rankings are quietly
    wrong for those states.

    That happened on 2026-09-03: AK, HI and PR were left over from the
    2026-08-21 run because NAIP is CONUS-only and they had been dropped
    from the state list.

THE RULE
    A partition whose newest parquet is older than best.pt's mtime cannot
    have been produced by the deployed model. No state list to maintain --
    the check stays correct across every future model swap.

    Caveat: this relies on best.pt's mtime being its UPLOAD time. scp -p
    and rsync -t PRESERVE the source mtime, which would make a freshly
    uploaded model look old and produce false "stale" reports. Use plain
    scp/rsync without -p/-t, or pass --model-time to override.

USAGE
    python check_od_freshness.py                 # report, exit 1 if stale
    python check_od_freshness.py --delete        # delete stale, exit 0
    python check_od_freshness.py --quarantine    # move aside, exit 0
    python check_od_freshness.py --allow AK HI PR  # known-uncoverable

    Exit codes:  0 = clean (or fixed)   1 = stale found   2 = setup error

    Chain it ahead of 02 in a slurm script so the pipeline stops rather
    than producing a mixed table:

        python $SCRIPTS/check_od_freshness.py || exit 1
        python $SCRIPTS/02_feature_engineering.py --states ...
"""
import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

SUBDIRS = ["plants", "tiles", "detections", "objects"]


def stamp(epoch: float) -> str:
    return dt.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M")


def newest_parquet_time(partition: Path) -> float | None:
    files = list(partition.glob("*.parquet"))
    if not files:
        return None
    return max(f.stat().st_mtime for f in files)


def resolve_model(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    cands = sorted(C.OD_MODEL_DIR.rglob("*.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="explicit best.pt (default: newest under OD_MODEL_DIR)")
    ap.add_argument("--model-time", help="override the model timestamp, YYYY-MM-DD "
                                         "(use when scp -p preserved the source mtime)")
    ap.add_argument("--allow", nargs="*", default=[],
                    help="states permitted to be stale (e.g. AK HI PR -- outside "
                         "NAIP/CONUS coverage and not reprocessable)")
    ap.add_argument("--delete", action="store_true", help="delete stale partitions")
    ap.add_argument("--quarantine", action="store_true",
                    help="move stale partitions to od_features_stale/ instead")
    args = ap.parse_args()

    if args.delete and args.quarantine:
        print("ERROR: --delete and --quarantine are mutually exclusive.")
        sys.exit(2)

    src_root = C.DATA_DIR / "od_features"
    if not src_root.exists():
        print(f"ERROR: {src_root} does not exist.")
        sys.exit(2)

    print("=== check_od_freshness.py ===")

    if args.model_time:
        try:
            model_mtime = dt.datetime.strptime(args.model_time, "%Y-%m-%d").timestamp()
        except ValueError:
            print(f"ERROR: --model-time must be YYYY-MM-DD, got {args.model_time!r}")
            sys.exit(2)
        print(f"Model time : {args.model_time} (from --model-time)")
    else:
        model = resolve_model(args.model)
        if model is None:
            print(f"ERROR: no .pt found under {C.OD_MODEL_DIR}.")
            sys.exit(2)
        model_mtime = model.stat().st_mtime
        print(f"Model      : {model}")
        print(f"Model time : {stamp(model_mtime)}")

    if args.allow:
        print(f"Allowed stale: {', '.join(args.allow)}")
    print()

    allow = set(args.allow)
    stale, empty, fresh = [], [], 0

    for sub in SUBDIRS:
        sub_dir = src_root / sub
        if not sub_dir.exists():
            continue
        for part in sorted(sub_dir.glob("state=*")):
            st = part.name.split("=", 1)[1]
            t = newest_parquet_time(part)
            if t is None:
                empty.append(f"{sub}/{part.name}")
                continue
            if t < model_mtime and st not in allow:
                stale.append((sub, part, st, t))
            else:
                fresh += 1

    print(f"{fresh} partition(s) newer than the model")
    if empty:
        print(f"{len(empty)} partition(s) hold no parquet files:")
        for e in empty[:10]:
            print(f"    {e}")
        if len(empty) > 10:
            print(f"    ... and {len(empty) - 10} more")

    if not stale:
        print("\nNo stale partitions. Safe to run 02.")
        sys.exit(0)

    by_state = {}
    for sub, part, st, t in stale:
        by_state.setdefault(st, []).append((sub, t))

    print(f"\n{len(stale)} STALE partition(s) across {len(by_state)} state(s) -- "
          f"written before the deployed model:")
    for st in sorted(by_state):
        times = [t for _, t in by_state[st]]
        subs = ", ".join(s for s, _ in by_state[st])
        print(f"    {st:4s} newest {stamp(max(times))}   [{subs}]")

    if not (args.delete or args.quarantine):
        print("\nThese hold detections from a DIFFERENT model. Running 02 now "
              "would blend two models into one national feature table.")
        print("Fix with one of:")
        print(f"    re-run 01b for: {' '.join(sorted(by_state))}")
        print("    python check_od_freshness.py --delete")
        print("    python check_od_freshness.py --quarantine")
        print("    python check_od_freshness.py --allow "
              f"{' '.join(sorted(by_state))}   (if uncoverable and intentional)")
        sys.exit(1)

    dst_root = C.DATA_DIR / "od_features_stale"
    print()
    for sub, part, st, _ in stale:
        if args.delete:
            shutil.rmtree(part)
            print(f"  deleted    {sub}/{part.name}")
        else:
            dst = dst_root / sub / part.name
            if dst.exists():
                shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(part), str(dst))
            print(f"  quarantined {sub}/{part.name}")

    print(f"\n{len(stale)} partition(s) handled. Safe to run 02.")
    sys.exit(0)


if __name__ == "__main__":
    main()
