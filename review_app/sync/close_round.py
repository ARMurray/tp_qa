"""
close_round.py
==============
Runs every local step that turns a finished review round into inputs the
next model training run can consume, then prints exactly what to upload.

    python -m sync.close_round --round 2 --dry-run
    python -m sync.close_round --round 2

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


def main():
    ap = argparse.ArgumentParser(
        description="Close a review round: export, fold into the master, "
                    "rebuild training bins, extract tiles.")
    ap.add_argument("--round", type=int, required=True)
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

    print(f"=== close_round.py -- round {args.round} ===")
    if args.dry_run:
        print("--dry-run: nothing will be written\n")

    results: list[tuple[str, bool, str]] = []

    # ---- 1. export verdicts --------------------------------------------
    argv = ["-m", "sync.push_review_log", "--round", str(args.round)]
    if args.all:
        argv.append("--all")
    ok = run("1/4  push_review_log -- export verdicts + route holdout",
             argv, APP_ROOT, args.dry_run)
    results.append(("push_review_log", ok, "audit trail + holdout routing"))

    # ---- 2. fold into the master ---------------------------------------
    # Not gated on step 1: the master update reads app.db directly, not the
    # exported parquet, so an export failure does not invalidate it.
    argv = ["-m", "sync.update_master_locations", "--round", str(args.round)]
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
    print(f"  holdout_truth_round{args.round}.parquet       -> merge into "
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
