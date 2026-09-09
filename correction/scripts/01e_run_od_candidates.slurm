#!/bin/bash
#SBATCH --job-name=corr_01e_odcand
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/01e_odcand_%A.log

# Object detection on Stage 2a's top-K candidate parcels, so a re-ranker can
# use it. This is the step 05_run_inference.py's docstring defers as "the
# not-yet-written OD-on-arbitrary-top-K step".
#
#   sbatch --export=DRYRUN="1" 01e_run_od_candidates.slurm            # plan only
#   sbatch 01e_run_od_candidates.slurm                                # holdout corrections
#   sbatch --export=SCOPE="train" 01e_run_od_candidates.slurm         # re-rank TRAINING data
#   sbatch --export=SCOPE="all",STATES="OH,WI,KS" 01e_run_od_candidates.slurm
#
# SCOPE:
#   (unset)  holdout bins only -- default, 880 parcels, ~10 min. The
#            evaluation set. Writes to data/od_features_candidates/.
#   train    corrections layer MINUS holdout -- the plants a re-ranker may
#            train on. ~266 plants x 20, roughly an hour. Writes to a
#            SEPARATE root, data/od_features_candidates_train/, so a
#            --no-resume rebuild of one cannot discard the other.
#            Requires patch_01e_training_scope.py.
#   all      every flagged plant in stage2_candidates. Nationally that is
#            240k parcels at the observed ~85/min, i.e. roughly 48 HOURS.
#            Always scope it with STATES first.
#
# Other variables:
#   DRYRUN=1      report the work plan, fetch no imagery
#   BINS=...      holdout bins (default "corrections")
#   STATES=...    comma-separated state filter
#   TOPK=20       candidates per plant. Keep at 20: capping OD at the top 5
#                 means the re-ranker can only reorder the 5 you would have
#                 reviewed anyway, and cannot promote a true parcel sitting
#                 at rank 12 -- which is where 27 of the 36 missing points
#                 of candidate recall actually live (08, 2026-09-08).
#   WORKERS=32    NAIP fetch threads
#   NORESUME=1    reprocess candidates already written. Needed after an OD
#                 model swap: resume is keyed on (CWNS_ID, ll_uuid) and
#                 cannot tell which model wrote a row.
#
# RATE LIMITING: 32 workers in ONE job is fine -- 01b's header notes trouble
# only at 52 concurrent ARRAY TASKS x 32 workers. One transient 429 appeared
# in the 880-parcel holdout run and the retry logic absorbed it. Do not fan
# this into an array without throttling the way 01b does.
#
# WALLTIME is 8h, raised from 4h now that SCOPE=all on a few states is a
# realistic invocation. A national --all run needs far more and should be
# split by state instead.

source /work/GRDVULN/correction/scripts/_common.sh

CMD="python $SCRIPTS/01e_run_od_candidates.py --workers ${WORKERS:-32}"

case "${SCOPE:-holdout}" in
    all)
        CMD="$CMD --all"
        ;;
    train)
        if ! grep -q -- "--training-corrections" "$SCRIPTS/01e_run_od_candidates.py"; then
            echo "ERROR: 01e_run_od_candidates.py has no --training-corrections option."
            echo "  Run 'sbatch patch_01e_training_scope.slurm' first."
            exit 1
        fi
        CMD="$CMD --training-corrections"
        ;;
    holdout)
        [ -n "${BINS:-}" ] && CMD="$CMD --holdout-bins $BINS"
        ;;
    *)
        echo "ERROR: SCOPE must be all, train, or unset (holdout)."
        exit 2
        ;;
esac

[ -n "${DRYRUN:-}" ]   && CMD="$CMD --dry-run"
[ -n "${STATES:-}" ]   && CMD="$CMD --states $STATES"
[ -n "${TOPK:-}" ]     && CMD="$CMD --top-k $TOPK"
[ -n "${NORESUME:-}" ] && CMD="$CMD --no-resume"

echo "Running: $CMD"
eval "$CMD"
RC=$?

echo "Finished: $(date)  (exit $RC)"
exit $RC
