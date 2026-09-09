#!/bin/bash
#SBATCH --job-name=corr_diag_fetch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/diag_fetch_%A.log

#   sbatch --export=STATE=KY diagnose_fetch_failures.slurm

source /work/GRDVULN/correction/scripts/_common.sh

STATE=${STATE:?ERROR: set STATE, e.g. --export=STATE=KY}

python "$SCRIPTS/diagnose_fetch_failures.py" --state "$STATE"

echo "Finished: $(date)"
