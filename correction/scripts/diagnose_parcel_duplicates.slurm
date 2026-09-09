#!/bin/bash
#SBATCH --job-name=corr_diag_parcel_dupes
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/diag_parcel_dupes_%A.log

# Checks whether PARCEL_BASE has duplicate rows per ll_uuid -- the suspected
# root cause of the 5.47x row inflation in 02_feature_engineering.py's parcel
# features step on the 2026-08-25 OH/MS/DE full-universe pilot.
#
#   sbatch --export=STATES="OH,MS,DE" diagnose_parcel_duplicates.slurm
#
# Memory set to 32G, well above what a plain GROUP BY over one state's parcel
# table should need -- OH alone is the largest state in this pipeline so far
# (4.4M candidate parcels from 01a; PARCEL_BASE itself is presumably larger
# since it's the full store, not just k=18-ring candidates).

source /work/GRDVULN/correction/scripts/_common.sh

STATES=${STATES:?ERROR: set STATES, e.g. --export=STATES="OH,MS,DE"}

CMD="python $SCRIPTS/diagnose_parcel_duplicates.py --states $STATES"
echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
