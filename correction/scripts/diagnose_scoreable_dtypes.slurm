#!/bin/bash
#SBATCH --job-name=corr_diag_scoreable
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/diag_scoreable_%A.log

#   sbatch --export=STATES="OH" diagnose_scoreable_dtypes.slurm
#   sbatch --export=STATES="OH",LIMIT="50" diagnose_scoreable_dtypes.slurm

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

STATES=${STATES:?ERROR: set STATES, e.g. --export=STATES="OH"}
LIMIT=${LIMIT:-20}

python "$SCRIPTS/diagnose_scoreable_dtypes.py" --states "$STATES" --limit "$LIMIT"

echo "Finished: $(date)"
