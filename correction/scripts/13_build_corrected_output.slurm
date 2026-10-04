#!/bin/bash
#SBATCH --job-name=corr_13_output
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/13_output_%A.log

# The corrected-locations product. CUTOFF is required -- it is the move rule,
# set from calibrate_move_rule.py and chosen by the project owner.
#
#   sbatch --export=CUTOFF=0.95 13_build_corrected_output.slurm   # decided 2026-10-04
#   sbatch --export=CUTOFF=0.95,REQUIRE_DETECTION=1 13_build_corrected_output.slurm
#
# Needs the national 01e array + 05b first; refuses otherwise.

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

if [ -z "${CUTOFF:-}" ]; then
    echo "ERROR: set CUTOFF, e.g. --export=CUTOFF=0.95"
    exit 1
fi

CMD="python $SCRIPTS/13_build_corrected_output.py --cutoff $CUTOFF"
[ -n "${REQUIRE_DETECTION:-}" ] && CMD="$CMD --require-detection"

echo "Running: $CMD"
eval "$CMD"
RC=$?
echo "Finished: $(date)  (exit $RC)"
exit $RC
