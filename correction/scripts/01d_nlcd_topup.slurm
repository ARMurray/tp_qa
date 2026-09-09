#!/bin/bash
#SBATCH --job-name=corr_01d_topup
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/01d_topup_%A.log

# Targeted NLCD zonal stats for parcels 01a's k-ring window missed, appended
# into the existing per-state nlcd_{STATE}_k{K}.parquet files.
#
# Run 08 first -- this reads data/diagnostics/missing_parcels_topup.csv.
#
#   sbatch --export=DRYRUN="1" 01d_nlcd_topup.slurm   # report only, no writes
#   sbatch 01d_nlcd_topup.slurm                        # apply
#
# WARNING: this MUTATES 01a outputs that later stages read. The script backs
# each file up to *.pre-topup-{timestamp}.parquet first. Do the dry run before
# the real one.
#
# 01d alone changes nothing downstream. After it succeeds:
#     02_feature_engineering.slurm  ->  06_build_stage2b_training.slurm
#     ->  07_train_stage2b.slurm  ->  re-run 08 to confirm the gain.

source /work/GRDVULN/correction/scripts/_common.sh

CMD="python $SCRIPTS/01d_nlcd_topup.py"
[ "${DRYRUN:-0}" = "1" ] && CMD="$CMD --dry-run"
[ -n "${UUIDS_FILE:-}" ] && CMD="$CMD --uuids-file $UUIDS_FILE"

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
