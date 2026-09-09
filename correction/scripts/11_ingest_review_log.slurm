#!/bin/bash
#SBATCH --job-name=corr_11_ingest_review
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/11_ingest_review_%A.log

# Converts reviewed, non-holdout verdicts into Updates.gdb-compatible
# classes/corrections rows (review_derived_locations.gpkg) plus
# candidate_recall_failures.parquet. See script docstring for the full
# verdict -> bin mapping and why this supersedes the earlier Stage-2b-only
# approach.
#
#   sbatch 11_ingest_review_log.slurm 1
#   sbatch 11_ingest_review_log.slurm 1 --dry-run
#
# ROUND is a positional argument (not --export=ROUND=...) for the same
# comma-truncation reasoning as 02/05/10's wrappers -- doesn't matter for a
# bare integer with no commas, but keeping the convention consistent across
# every wrapper in this pipeline is worth more than the one saved keystroke.
#
# Prerequisite: review_log_round{N}.parquet must already be sitting in
# data/review_log_incoming/ -- this script does not sync it down itself
# (that's a manual MobaXterm upload from the local review app's
# data/outgoing/, per sync/push_review_log.py).
#
# Memory sized modestly -- this is a handful of parquet/gpkg rows and a few
# small DuckDB lookups against PARCEL_BASE for candidate_correct centroids,
# nowhere near the scale of 01a/02's nationwide parcel work.

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

ROUND_ARG="${1:?ERROR: pass round number as a positional arg, e.g. sbatch 11_ingest_review_log.slurm 1}"
shift || true   # allow a trailing --dry-run after the round number

CMD="python $SCRIPTS/11_ingest_review_log.py --round $ROUND_ARG $*"

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
