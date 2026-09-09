#!/bin/bash
#SBATCH --job-name=corr_check_competition
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/check_competition_%A.log

#   sbatch check_competition_losses.slurm

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

python "$SCRIPTS/check_competition_losses.py"

echo "Finished: $(date)"
