#!/bin/bash
#SBATCH --job-name=corr_diag_odcand
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/diag_odcand_%A.log

#   sbatch diagnose_candidate_od.slurm
#
# Read-only. Run AFTER 01e_run_od_candidates.slurm. Measures whether object
# detection discriminates among a plant's top-K candidates, and whether the
# TRUE parcel fires more than its competitors -- directly against the
# holdout truth, with no model and therefore no distribution-shift caveat.
#
# Worth running before building any re-rank step: if true parcels fire no
# more often than competitors, a re-rank cannot help no matter how the
# model is trained, and the answer costs 15 minutes instead of a day.

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

python "$SCRIPTS/diagnose_candidate_od.py"

echo "Finished: $(date)"
