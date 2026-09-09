#!/bin/bash
#SBATCH --job-name=corr_patch_01e
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/patch_01e_%A.log

# Adds --training-corrections to 01e_run_od_candidates.py.
#
#   sbatch --export=CHECK="1" patch_01e_training_scope.slurm   # report only
#   sbatch patch_01e_training_scope.slurm                      # apply
#
# Run CHECK first and read the log before applying. The patcher backs the
# file up, verifies the result compiles, and reverts itself if it does not --
# but a --check run costs nothing and shows exactly which anchors matched.
#
# Idempotent: applying twice is a no-op.

source /work/GRDVULN/correction/scripts/_common.sh

CMD="python $SCRIPTS/patch_01e_training_scope.py"
[ -n "${CHECK:-}" ] && CMD="$CMD --check"

echo "Running: $CMD"
eval "$CMD"
RC=$?

echo "Finished: $(date)  (exit $RC)"
exit $RC
