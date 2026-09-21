"""
close_round.py
==============
Runs every local step that turns a finished review round into inputs the
next model training run can consume, then prints exactly what to upload.

    python -m sync.close_round --round 2 --dry-run
    python -m sync.close_round --round 2

    python -m sync.close_round              # catch-up: every reviewed round

Omitting --round closes every round with reviewed plants in app.db, folding
them into the master in a single dated layer. That is what you want the
first time this runs on a machine that has review history but has never
folded any of it in. Normal per-round use passes --round N.

FIRST RUN ON A MACHINE HOLDING THE MASTER: the master must already be a
.gpkg. If it is still Updates.gdb, run sync/migrate_master_to_gpkg.py once
first -- this script cannot create the master, only add layers to it.

WHY ONE COMMAND
---------------
A review round produces four different things with four different
destinations, and before this script they were four separate commands run
from memory in the right order. Forgetting one is silent: the round still
"looks" done, and the omission only surfaces as a training run that
mysteriously didn't improve.

    1. verified locations  -> the master Updates.gpkg  (training labels)
    2. NAIP tiles          -> detection/data/tiles/    (OD retraining)
    3. holdout truth       -> a separate parquet       (evaluation only)
    4. recall diagnostics  -> candidate_recall_failures.parquet (K_RINGS)

THE STEPS, IN ORDER
-------------------
  1. push_review_log.py
        Exports verdicts to parquet. This is the audit trail and the
        holdout router -- holdout verdicts go to their own file here and
        never enter anything downstream.

  2. update_master_locations.py
        Writes a new dated CWNS_Locations layer into the master gpkg, and
        appends this round's truth_outside_candidates rows to
        candidate_recall_failures.parquet. This is THE step that makes a
        review permanent.

  3. build_training_bins.py   (--skip-bins to omit)
        Rebuilds training_locations.gpkg from the master's newest layer.
        Run locally rather than on HPC so the narrow file the pipeline
        actually consumes is what gets uploaded, not the whole master.

  4. extract_review_tiles.py  (--skip-tiles to omit)
        Pulls NAIP tiles for every parcel the reviewer saw into the
        detection inventory. Slow -- it fetches imagery -- so it's the
        step most worth skipping on a re-run.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It does not upload anything or touch the HPC. It prints the upload list and
the sbatch sequence and stops. Transferring files and launching jobs on a
shared cluster stays a deliberate act.

HOW REVIEW REACHES EACH MODEL (worth understanding before changing this)
------------------------------------------------------------------------
Everything flows through the master -> training_locations.gpkg. There is
ONE derivation path; the old second one (11_ingest_review_log.py on HPC)
was deleted 2026-09-21.

  Stage 1 / Stage 2a   read the classes + corrections layers directly.
                        A reported_correct verdict adds a Correct row; a
                        candidate_correct or truth_outside_candidates
                        verdict adds an Incorrect row plus a correction.

  Stage 2b (06/07)     pairs each correction's reported parcel (label 0)
                        against its corrected parcel (label 1).

  Re-ranker (06b/07b)  is where review pays off most, and it needs no
                        extra plumbing. 06b resolves each correction's
                        Corrected_X/Y to its containing parcel, calls that
                        parcel label=1, and labels the other ~19 Stage 2a
                        candidates 0. So a single candidate_correct verdict
                        becomes one positive AND ~19 distribution-matched
                        hard negatives automatically.

                        This is why review_log_candidates_round{N}.parquet
                        is not fed anywhere: the app only ever showed the
                        reviewer TOP_K_SHOWN (5) candidates, while 06b
                        reconstructs the full top-20 pool from
                        stage2_candidates.parquet. The reconstruction is
                        strictly richer than the export. The export is kept
                        purely as a record of what was on screen.

                        06b only sees a plant's candidates if 01e has run
                        OD on them -- that is the expensive HPC step this
                        script's closing checklist reminds you about, and
                        the one most easily forgotten.
"""
import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parents[1]
BUILD_BINS = REPO_ROOT / "correction" / "scripts" / "build_training_bins.py"
TRAINING_GPKG_OUT = C.OUTGOING_DIR / "training_locations.gpkg"


def run(label: str, argv: list[str], cwd: Path, dry_run: bool) -> bool:
    """Run one step. Returns False on failure.

    Steps are NOT run with check=True: a failure in the tile fetch (step 4,
    which depends on a network service) should not make it look like the
    master update (step 2) failed too. Each step reports its own outcome and
    the summary at the end says plainly which ones ran.
    """
    print(f"\n{'=' * 72}\n>>> {label}\n{'=' * 72}")
    printable = " ".join(str(a) for a in argv)
    print(f"$ {printable}\n")
    if dry_run:
        print("  [--dry-run] not executed")
        return True
    result = subprocess.run([sys.executable, *argv], cwd=str(cwd))
    if result.returncode != 0:
        print(f"\n  *** {label} FAILED (exit {result.returncode}) ***")
        return False
    return True


def reviewed_rounds(db_path: Path) -> list[int]:
    """Every round with at least one reviewed plant, ascending."""
    if not db_path.exists():
        raise SystemExit(f"app.db not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT review_round FROM plants WHERE reviewed = 1 "
            "ORDER BY review_round").fetchall()
    return [r[0] for r in rows]


def main():
    ap = argparse.ArgumentParser(
        description="Close a review round: export, fold into the master, "
                    "rebuild training bins, extract tiles.")
    # Optional, not required (2026-09-21). Omitting it closes EVERY reviewed
    # round at once, which is what you want on a first run against a machine
    # that has review history but has never folded any of it into the master
    # -- the catch-up case. Passing --round N is the normal per-round use.
    ap.add_argument("--round", type=int, default=None,
                    help="round to close. Omit to close every reviewed round "
                         "found in app.db (catch-up).")
    ap.add_argument("--all", action="store_true",
                    help="pass --all to push_review_log (re-export every "
                         "reviewed plant for this round, not just new ones)")
    ap.add_argument("--include-holdout", action="store_true",
                    help="pass through to update_master_locations. Read that "
                         "script's docstring before using it.")
    ap.add_argument("--skip-bins", action="store_true",
                    help="don't rebuild training_locations.gpkg locally "
                         "(build it on HPC with 00_build_training_bins.slurm)")
    ap.add_argument("--skip-tiles", action="store_true",
                    help="don't fetch NAIP tiles. The slow step -- skip it "
                         "when re-running to fix something downstream.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every command without running any of them")
    args = ap.parse_args()

    rounds = [args.round] if args.round is not None else reviewed_rounds(C.APP_DB_PATH)
    if not rounds:
        raise SystemExit("No reviewed plants in app.db -- nothing to close.")

    label = f"round {args.round}" if args.round is not None \
        else f"ALL reviewed rounds {rounds}"
    print(f"=== close_round.py -- {label} ===")
    if args.round is None:
        print("  (no --round given: closing every reviewed round at once)")
    if args.dry_run:
        print("--dry-run: nothing will be written\n")

    results: list[tuple[str, bool, str]] = []

    # ---- 1. export verdicts --------------------------------------------
    # One export per round: push_review_log writes per-round parquet files
    # and tracks exports per round, so it has no "all rounds" mode.
    export_ok = True
    for rnd in rounds:
        argv = ["-m", "sync.push_review_log", "--round", str(rnd)]
        if args.all:
            argv.append("--all")
        if not run(f"1/4  push_review_log -- export round {rnd} + route holdout",
                   argv, APP_ROOT, args.dry_run):
            export_ok = False
    results.append(("push_review_log", export_ok,
                    f"audit trail + holdout routing ({len(rounds)} round(s))"))

    # ---- 2. fold into the master ---------------------------------------
    # ONE call covering every round: the master update reads app.db directly
    # and updates rows in place, so folding all rounds at once produces one
    # dated layer rather than a chain of same-day _vN layers.
    #
    # Not gated on step 1: it reads app.db, not the exported parquet, so an
    # export failure does not invalidate it.
    argv = ["-m", "sync.update_master_locations"]
    if args.round is not None:
        argv += ["--round", str(args.round)]
    if args.include_holdout:
        argv.append("--include-holdout")
    if args.dry_run:
        argv.append("--dry-run")
    ok_master = run("2/4  update_master_locations -- new dated layer in the master",
                    argv, APP_ROOT, args.dry_run)
    results.append(("update_master_locations", ok_master,
                    "THE step that makes a review permanent"))

    # ---- 3. rebuild training bins --------------------------------------
    # Gated on step 2: rebuilding bins from a master that failed to update
    # produces a stale training file that looks perfectly valid.
    if not args.skip_bins:
        if not ok_master:
            print("\n  SKIPPING build_training_bins -- the master update failed, "
                  "and bins built from an un-updated master would silently be "
                  "last round's labels.")
            results.append(("build_training_bins", False, "skipped: master update failed"))
        else:
            C.OUTGOING_DIR.mkdir(parents=True, exist_ok=True)
            argv = [str(BUILD_BINS), "--master", str(C.MASTER_GPKG),
                    "--out", str(TRAINING_GPKG_OUT)]
            ok = run("3/4  build_training_bins -- classes/corrections/unverified",
                     argv, REPO_ROOT, args.dry_run)
            results.append(("build_training_bins", ok, str(TRAINING_GPKG_OUT)))
    else:
        results.append(("build_training_bins", None, "skipped (--skip-bins)"))

    # ---- 4. NAIP tiles for reviewed parcels ----------------------------
    if not args.skip_tiles:
        argv = ["-m", "analysis.extract_review_tiles"]
        if args.dry_run:
            argv.append("--dry-run")
        ok = run("4/4  extract_review_tiles -- NAIP into the detection inventory",
                 argv, APP_ROOT, args.dry_run)
        results.append(("extract_review_tiles", ok, "detection/data/tiles/"))
    else:
        results.append(("extract_review_tiles", None, "skipped (--skip-tiles)"))

    # ---- summary + what to do next -------------------------------------
    print(f"\n{'=' * 72}\n=== round {args.round} summary ===\n{'=' * 72}")
    for name, ok, note in results:
        mark = "skip" if ok is None else ("ok  " if ok else "FAIL")
        print(f"  [{mark}] {name:<26} {note}")

    failed = [n for n, ok, _ in results if ok is False]
    if failed:
        print(f"\n  {len(failed)} step(s) failed: {', '.join(failed)}")
        print("  Fix and re-run -- every step is idempotent.")

    print(f"\nUPLOAD TO HPC ({C.OUTGOING_DIR}):")
    print(f"  training_locations.gpkg          -> correction/data/training/")
    print(f"  candidate_recall_failures.parquet -> correction/data/features/")
    for rnd in rounds:
        print(f"  holdout_truth_round{rnd}.parquet        -> merge into "
              f"correction/data/holdout/holdout_truth.parquet BY HAND")
    print(f"     (evaluation only -- it must never reach training)")

    print(f"\nTHEN, ON HPC -- in this order:")
    print(f"  1. sbatch 01a_extract_parcels.slurm      # only for plants new to the")
    print(f"     sbatch 01c_run_od_corrected_locations.slurm   #   corrections bin")
    print(f"  2. sbatch --export=SCOPE=\"train\" 01e_run_od_candidates.slurm")
    print(f"     # REQUIRED for the re-ranker. 06b silently drops any plant whose")
    print(f"     # candidates have no OD output, so skipping this is how a round's")
    print(f"     # hard negatives quietly fail to reach 07b.")
    print(f"  3. sbatch 02_feature_engineering.slurm")
    print(f"  4. sbatch 03_train_stage1.slurm / 04_train_stage2.slurm")
    print(f"     sbatch 06_build_stage2b_training.slurm -> 07_train_stage2b.slurm")
    print(f"     sbatch 06b_build_rerank_training.slurm -> 07b_train_rerank.slurm")
    print(f"  5. sbatch 12_score_holdout.slurm         # round-over-round comparison")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
