#!/bin/bash
#SBATCH --job-name=corr_09_holdout
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/09_holdout_%A.log

# Samples the frozen evaluation holdout. RUN THIS ONCE.
#
#   sbatch --export=DRYRUN="1",NCORRECT="300",NUNLABELED="75" 09_build_holdout.slurm
#   sbatch --export=NCORRECT="300",NUNLABELED="75" 09_build_holdout.slurm
#
# Variables (all optional):
#   DRYRUN=1        inspect the draw, write nothing
#   NCORRECTIONS    known-misplaced plants   (default 50)
#   NCORRECT        known-correct plants     (default 150)
#   NINCORRECT      wrong-but-unresolved     (default 0 -- bin is empty)
#   NUNLABELED      random unlabeled plants  (default 75)
#   SEED            sampling seed            (default 20260824)
#   STATES          restrict the pool, e.g. "OH,PA"
#   COHORT          cohort label             (default A)
#   APPEND          append a NEW cohort, e.g. APPEND="B"
#   FORCE=1         overwrite the manifest -- destroys comparability
#
# Runtime is dominated by the parcel lookup, which scans each state's Regrid
# parquet once. Budget 5-15 minutes; more with a large NUNLABELED, since those
# plants scatter across states the labeled draw would not otherwise touch.
#
# The script refuses to overwrite an existing manifest without FORCE=1. That is
# deliberate: the holdout is sampled once and frozen, because a re-draw rotates
# plants between train and eval and silently invalidates every score recorded
# so far. To GROW the holdout, use APPEND instead.
#
# After this succeeds, 03/04/06/07 will FAIL until they import holdout and call
# exclude_holdout(). That is intended -- fail-closed beats silently training on
# the evaluation set.

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

CMD="python $SCRIPTS/09_build_holdout.py"
[ -n "${NCORRECTIONS:-}" ] && CMD="$CMD --n-corrections $NCORRECTIONS"
[ -n "${NCORRECT:-}" ]     && CMD="$CMD --n-correct $NCORRECT"
[ -n "${NINCORRECT:-}" ]   && CMD="$CMD --n-incorrect $NINCORRECT"
[ -n "${NUNLABELED:-}" ]   && CMD="$CMD --n-unlabeled $NUNLABELED"
[ -n "${SEED:-}" ]         && CMD="$CMD --seed $SEED"
[ -n "${STATES:-}" ]       && CMD="$CMD --states $STATES"
[ -n "${COHORT:-}" ]       && CMD="$CMD --cohort $COHORT"
[ -n "${APPEND:-}" ]       && CMD="$CMD --append-cohort $APPEND"
[ "${DRYRUN:-0}" = "1" ]   && CMD="$CMD --dry-run"
[ "${FORCE:-0}" = "1" ]    && CMD="$CMD --force-resample"

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
