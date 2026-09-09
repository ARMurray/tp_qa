#!/bin/bash
#SBATCH --job-name=corr_diag_s1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/diag_s1_%A.log

#   sbatch diagnose_stage1_attrition.slurm

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

python "$SCRIPTS/diagnose_stage1_attrition.py"

echo "Finished: $(date)"
