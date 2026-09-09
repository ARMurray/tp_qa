#!/bin/bash
#SBATCH --job-name=corr_12_holdout
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/12_holdout_%A.log

# Scores the frozen holdout. Read-only.
#
#   sbatch 12_score_holdout.slurm                      # the OD re-rank
#   sbatch --export=NOOD="1" 12_score_holdout.slurm    # the ablation
#
# RUN BOTH and compare. This is the clean read: every holdout plant was
# excluded from Stage 1, 2a, 2b and the re-ranker by the anti-joins, so
# unlike 07b's internal test split these numbers involve no data the models
# selected hyperparameters against.
#
# Expect LOWER figures than 07b reported. Two reasons, both legitimate:
#   - 07b's split, while grouped by plant, still came from the same pool the
#     hyperparameter search ran over
#   - the holdout measured 64% candidate recall@20 against the training
#     pool's 81%. If the holdout is genuinely the harder sample, everything
#     downstream reads lower.
# A gap is not evidence of a bug. A LARGE gap is worth investigating.
#
# Variables:
#   NOOD=1        score stage2_candidates_reranked_noOD.parquet
#   BINS=...      holdout bins (default "corrections,correct")
#   COHORT=A      restrict to one cohort. 09 keeps cohort A comparable
#                 across every round; the all-cohort number gets tighter
#                 over time but is not comparable to earlier runs.
#
# REQUIRES 05b to have been run for the matching arm.

source /work/GRDVULN/correction/scripts/_common.sh

CMD="python $SCRIPTS/12_score_holdout.py"
[ -n "${NOOD:-}" ]   && CMD="$CMD --no-od"
[ -n "${BINS:-}" ]   && CMD="$CMD --bins $BINS"
[ -n "${COHORT:-}" ] && CMD="$CMD --cohort $COHORT"

echo "Running: $CMD"
eval "$CMD"
RC=$?

echo "Finished: $(date)  (exit $RC)"
exit $RC
