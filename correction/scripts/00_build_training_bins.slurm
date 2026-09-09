#!/bin/bash
#SBATCH --job-name=corr_00_bins
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/00_bins_%A.log

# Builds data/training/training_locations.gpkg from the uploaded Updates.gdb,
# optionally unioning in review-derived rows from 11_ingest_review_log.py
# (Updates.gdb wins on any CWNS_ID conflict -- see build_training_bins.py's
# own printed diagnostics for exactly which rows that affects, if any).
#
#   sbatch 00_build_training_bins.slurm
#   sbatch --export=LAYER="CWNS_Locations_20260901" 00_build_training_bins.slurm
#   sbatch --export=REVIEWGPKG=1 00_build_training_bins.slurm
#
# REVIEWGPKG=1 points --review-gpkg at the standard path
# 11_ingest_review_log.py always writes to
# (data/training/review/review_derived_locations.gpkg) -- no need to spell
# the path out yourself unless it's been moved.
#
# Small/fast job -- debug partition is fine.

source /work/GRDVULN/correction/scripts/_common.sh

CMD="python \"$SCRIPTS/build_training_bins.py\" \
    --gdb   \"$ROOT/data/training/Updates.gdb\" \
    --layer \"${LAYER:-CWNS_Locations_20260820}\" \
    --out   \"$ROOT/data/training/training_locations.gpkg\""
[ "${REVIEWGPKG:-0}" = "1" ] && \
    CMD="$CMD --review-gpkg \"$ROOT/data/training/review/review_derived_locations.gpkg\""

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
