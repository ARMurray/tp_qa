#!/bin/bash
#SBATCH --job-name=corr_diag_bytearray
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/diag_bytearray_%A.log

#   sbatch --export=STATES="OH,MS,DE" diagnose_bytearray_columns.slurm

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

STATES=${STATES:?ERROR: set STATES, e.g. --export=STATES="OH,MS,DE"}

python "$SCRIPTS/diagnose_bytearray_columns.py" --states "$STATES"

echo "Finished: $(date)"
