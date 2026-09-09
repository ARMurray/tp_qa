#!/bin/bash
#SBATCH --job-name=corr_05_merge
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/05_merge_%A.log

# Concatenates data/inference_shards/state=XX/ into the two flat files that
# 10_build_review_queue.py reads by name:
#     data/inference/plant_summary.parquet
#     data/inference/stage2_candidates.parquet
#
#   sbatch merge_05_shards.slurm "AL AR AZ ... WY"
#   sbatch --export=DRYRUN="1" merge_05_shards.slurm "AL AR AZ ... WY"
#   sbatch --export=ALLOWMISSING="1" merge_05_shards.slurm "AL AR ... WY"
#
# STATES is a POSITIONAL argument, space-separated -- matching
# 05_run_inference_array.slurm. Do NOT pass it via --export: sbatch's own
# --export parser splits on commas, which silently truncated a state list to
# its first element on 2026-08-26 (see 02_feature_engineering.slurm's header).
# Space-separated positional args aren't subject to that at all.
#
# Variables:
#   DRYRUN=1        report shard counts and row totals, write nothing
#   ALLOWMISSING=1  merge even when a state produced no shard. OFF by
#                   default on purpose: a missing state yields a perfectly
#                   well-formed merged file that is silently short, and
#                   nothing downstream would notice.
#
# Memory: the merge concatenates every state's candidate table in one frame.
# stage2_candidates is capped at --top-k rows per flagged plant (default 20),
# so this is far smaller than the raw candidate pools 05 works with -- on the
# order of 10^5 rows nationally, not 10^8. 128G is generous; lower it if the
# partition is busy.

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

STATES_ARG="${1:?ERROR: pass states as a positional arg, e.g. sbatch merge_05_shards.slurm \"AL AR AZ\"}"

CMD="python $SCRIPTS/merge_05_shards.py --states \"$STATES_ARG\""
[ -n "${DRYRUN:-}" ]       && CMD="$CMD --dry-run"
[ -n "${ALLOWMISSING:-}" ] && CMD="$CMD --allow-missing"

echo "Running: $CMD"
eval "$CMD"
RC=$?

echo "Finished: $(date)  (exit $RC)"
exit $RC
