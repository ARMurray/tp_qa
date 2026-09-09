#!/bin/bash
#SBATCH --job-name=corr_10_review_queue
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/10_review_queue_%A.log

# Builds data/review_queue/review_queue_round{N}.parquet from 05's inference
# output. See script docstring for the Stage2b->Stage2a substitution and the
# two review-task-type design (candidate_pick / confirm_reported).
#
#   sbatch 10_build_review_queue.slurm "OH,MS,DE" 1
#   sbatch --export=BATCHSIZE="150" 10_build_review_queue.slurm "OH,MS,DE" 1
#
# Positional args (STATES then ROUND) -- same comma-truncation reasoning as
# 02/05's wrappers: --export=STATES="OH,MS,DE" gets silently truncated at the
# first comma by sbatch's own parser. Round is positional too rather than
# --export for consistency, even though it has no commas to truncate.
#
# Requires 05_run_inference.py to have already run for these exact states --
# reads data/inference/plant_summary.parquet and stage2_candidates.parquet
# directly, does not regenerate them.

source /work/GRDVULN/correction/scripts/_common.sh

STATES_ARG="${1:?ERROR: pass states as a positional arg, e.g. sbatch 10_build_review_queue.slurm \"OH,MS,DE\" 1}"
ROUND_ARG="${2:?ERROR: pass round number as the second positional arg, e.g. sbatch 10_build_review_queue.slurm \"OH,MS,DE\" 1}"

CMD="python $SCRIPTS/10_build_review_queue.py --states $STATES_ARG --round $ROUND_ARG"
[ -n "${BATCHSIZE:-}" ]     && CMD="$CMD --batch-size $BATCHSIZE"
[ -n "${MODELVERSION:-}" ]  && CMD="$CMD --model-version $MODELVERSION"
[ -n "${SEED:-}" ]          && CMD="$CMD --seed $SEED"

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
