#!/bin/bash
#SBATCH --job-name=corr_diag_01b
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/diag_01b_%A.log

# Summarizes 01b's plants/ output for one state and identifies which
# training-universe plants are MISSING from it entirely (i.e. failed the
# "no containing parcel found" check, which 01b only counts, not lists).
#
#   sbatch --export=STATE=OH diagnose_01b_output.slurm

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

STATE=${STATE:?ERROR: set STATE, e.g. --export=STATE=OH}

python "$SCRIPTS/diagnose_01b_output.py" --state "$STATE"

echo "Finished: $(date)"
