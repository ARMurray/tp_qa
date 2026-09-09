#!/bin/bash
#SBATCH --job-name=corr_inspect_s2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/inspect_s2_%A.log

#   sbatch inspect_stage2_columns.slurm

source /work/GRDVULN/correction/scripts/_common.sh

python "$SCRIPTS/inspect_stage2_columns.py"

echo "Finished: $(date)"
