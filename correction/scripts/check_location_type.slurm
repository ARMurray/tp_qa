#!/bin/bash
#SBATCH --job-name=corr_check_loctype
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/check_loctype_%A.log

#   sbatch check_location_type.slurm

source /work/GRDVULN/correction/scripts/_common.sh

python "$SCRIPTS/check_location_type.py"

echo "Finished: $(date)"
