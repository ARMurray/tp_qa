#!/bin/bash
#SBATCH --job-name=corr_06_s2b
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/06_s2b_%A.log

# Builds data/features/16_stage2b_training.parquet -- the contrastive
# reported-vs-corrected training table for Stage 2b.
#
#   sbatch 06_build_stage2b_training.slurm
#
# PREREQ: 01c_run_od_corrected_locations.py must have completed, and
# 10_parcel_features.parquet must exist (from the nationwide 02 run).

source /work/GRDVULN/correction/scripts/_common.sh

python "$SCRIPTS/06_build_stage2b_training.py"

echo "Finished: $(date)"
