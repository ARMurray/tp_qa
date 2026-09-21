#!/bin/bash
#SBATCH --job-name=corr_00_bins
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/00_bins_%A.log

# Builds data/training/training_locations.gpkg from the uploaded Updates.gpkg.
#
# Review verdicts reach this file through the MASTER, not through a separate
# review feed: review_app/sync/close_round.py writes them into Updates.gpkg
# as a new dated layer before it is uploaded. The old second path
# (11_ingest_review_log.py + --review-gpkg) was deleted 2026-09-21 -- one
# source of truth, one derivation.
#
#   sbatch 00_build_training_bins.slurm
#   sbatch --export=LAYER="CWNS_Locations_20260901" 00_build_training_bins.slurm   # pin an older layer
#
# Small/fast job -- debug partition is fine.

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

CMD="python \"$SCRIPTS/build_training_bins.py\" \
    --master \"$ROOT/data/training/Updates.gpkg\" \
    --out    \"$ROOT/data/training/training_locations.gpkg\""
# No --layer by default: build_training_bins.py resolves the newest dated
# CWNS_Locations layer in the gpkg, which is the one carrying the most
# recent round of review. Set LAYER to pin an older one deliberately.
[ -n "${LAYER:-}" ] && CMD="$CMD --layer \"$LAYER\""

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
