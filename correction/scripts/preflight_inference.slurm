#!/bin/bash
#SBATCH --job-name=corr_preflight05
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/preflight05_%A.log

# Read-only check before 05_run_inference_array: are the feature tables,
# 01a candidate parcels and 01b detections built for the FULL plant universe,
# is detection output fresh against best.pt, and is the re-ranker older than
# the current Stage 2a model. See preflight_inference.py's docstring.
#
#   sbatch preflight_inference.slurm
#   sbatch preflight_inference.slurm "OH WI KS"     # a subset
#
# Writes correction/diagnostics/preflight_inference_<host>_<stamp>.txt.
# Then, from a login node:
#   cd /work/GRDVULN/tp_qa
#   git add correction/diagnostics/
#   git commit -m "preflight"
#   git push

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

STATES_ARG="${1:-$DEFAULT_STATES}"
CMD="python $SCRIPTS/preflight_inference.py --states \"$STATES_ARG\""

echo "Running: $CMD"
eval "$CMD"
RC=$?

echo "Finished: $(date)  (exit $RC)"
exit $RC
