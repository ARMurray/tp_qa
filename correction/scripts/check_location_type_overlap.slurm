#!/bin/bash
#SBATCH --job-name=corr_check_loctype_overlap
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/check_loctype_overlap_%A.log

#   sbatch check_location_type_overlap.slurm

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

python "$SCRIPTS/check_location_type_overlap.py"

echo "Finished: $(date)"
