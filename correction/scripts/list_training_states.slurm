#!/bin/bash
#SBATCH --job-name=corr_list_states
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/list_states_%A.log

#   sbatch list_training_states.slurm

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

python "$SCRIPTS/list_training_states.py"

echo "Finished: $(date)"
