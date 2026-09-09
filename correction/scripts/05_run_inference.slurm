#!/bin/bash
#SBATCH --job-name=corr_05_inference
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/05_inference_%A.log

# Stage 1 -> Stage 2a inference (Stage 2b excluded this round -- see script
# docstring / TPQA_MASTER_REFERENCE.md S3,S10).
#
#   sbatch 05_run_inference.slurm "OH,MS,DE"
#   sbatch --export=TOPK="20" 05_run_inference.slurm "OH,MS,DE"
#
# STATES is a POSITIONAL argument (after the script name), NOT --export=STATES=...
# -- same reasoning as 02_feature_engineering.slurm: sbatch's --export uses
# COMMAS to separate multiple name=value pairs within the --export argument
# itself, so a comma-separated state list passed that way gets silently
# truncated at the first comma by SLURM's own parser. Confirmed 2026-08-26:
# `--export=STATES="OH,MS,DE"` actually ran with just "OH" -- no error, no
# warning, just silently wrong scope. Positional args aren't subject to that
# delimiter at all.
#
# Requires, for every state listed:
#   - 05_plant_features.parquet / 10_parcel_features.parquet built with
#     02_feature_engineering.py --full-universe --states <these states>
#   - 01a NLCD output (nlcd_{STATE}_k18.parquet)
#   - 01b OD output (data/od_features/plants/state=XX/)
#   - trained stage1_rf_model.joblib / stage2_rf_model.joblib in models/
#
# Memory: candidate tables can run into the low millions of rows for a large
# state's worth of flagged plants (K_RINGS=18 candidates per plant) -- OH
# alone produced 5.6M raw candidates. 64G is a starting point based on that,
# not a hard measurement for larger multi-state runs.

source /work/GRDVULN/correction/scripts/_common.sh

STATES_ARG="${1:?ERROR: pass states as a positional arg, e.g. sbatch 05_run_inference.slurm \"OH,MS,DE\"}"

CMD="python $SCRIPTS/05_run_inference.py --states $STATES_ARG"
[ -n "${TOPK:-}" ] && CMD="$CMD --top-k $TOPK"

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
