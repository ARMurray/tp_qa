#!/bin/bash
#SBATCH --job-name=corr_diag_odcls
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/diag_odcls_%A.log

#   sbatch diagnose_od_class_shift.slurm
#
# Read-only. Loads the newest best.pt from OD_MODEL_DIR, reads its class
# names, and measures how much Stage 1/2/2b importance sits on OD classes
# the new weights can no longer emit. Writes nothing; safe to run while
# other jobs are queued.
#
# Upload the new best.pt to $ROOT/models/object_detection/ BEFORE running,
# or this will report on whatever weights are already there.

source /work/GRDVULN/correction/scripts/_common.sh

python "$SCRIPTS/diagnose_od_class_shift.py"

echo "Finished: $(date)"
