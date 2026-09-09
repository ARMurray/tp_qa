#!/bin/bash
#SBATCH --job-name=corr_02
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/02_%A.log

# Feature engineering: assembles plant features, parcel features, and the
# Stage 1 / Stage 2 training tables.
# Output: data/features/{05_plant,10_parcel,14_stage1_training,15_stage2_training}.parquet
#
#   sbatch 02_feature_engineering.slurm "OH,PA"
#   sbatch 02_feature_engineering.slurm            # all states with 01a output
#
# STATES is passed as a POSITIONAL argument (after the script name), NOT via
# --export=STATES=... . This is deliberate: sbatch's --export uses COMMAS to
# separate multiple name=value pairs within the --export argument itself, so
# a comma-separated state list passed that way gets silently truncated at
# the first comma by SLURM's own parser -- confirmed 2026-08-21, a 52-state
# --export=STATES="AK,AL,..." submission actually ran with just "AK". Shell
# quoting protects against the SHELL's word-splitting but does nothing
# against SLURM's own --export delimiter. Positional args aren't subject to
# that at all, so this sidesteps the problem entirely rather than requiring
# a different internal delimiter + conversion step.
#
# NOTE: unlike 01a/01b this is NOT an array job -- it builds ONE combined
# feature table across all requested states, so it must see them together.
#
# MEMORY (2026-08-21): Part 2's parcel-feature build writes each state's
# result to its own file and releases it before moving to the next state,
# rather than holding every state's full parcel-attribute table in memory
# simultaneously -- for a nationwide run (52 states, ~2382 training-universe
# plants) that pattern would have meant an estimated 20-25M+ rows held at
# once, extrapolating from Ohio's 935k rows for 84 plants alone. This
# substantially reduces peak memory during accumulation, but the FINAL
# combined parcel_features table (needed whole for the Stage 1/2 joins) is
# still nationwide-sized, hence the generous 200G request rather than OH's
# 128G. If this still OOMs, the fallback is to split STATES across 2-3
# separate 02 runs with distinct output paths and combine the resulting
# parquet files by hand -- ask before doing that, it needs a small script.

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

STATES_ARG="${1:-}"

CMD="python $SCRIPTS/02_feature_engineering.py"
[ -n "$STATES_ARG" ] && CMD="$CMD --states $STATES_ARG"
[ "${FULL_UNIVERSE:-0}" = "1" ] && CMD="$CMD --full-universe"

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
