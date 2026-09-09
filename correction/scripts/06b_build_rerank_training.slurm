#!/bin/bash
#SBATCH --job-name=corr_06b_rerank
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/06b_rerank_%A.log

# Builds data/features/17_rerank_training.parquet: one row per Stage 2a
# candidate parcel, labelled 1 for the plant's true parcel and 0 for the
# competitors Stage 2a also surfaced.
#
#   sbatch --export=DRYRUN="1" 06b_build_rerank_training.slurm
#   sbatch 06b_build_rerank_training.slurm
#   sbatch --export=DROPNOPOS="1" 06b_build_rerank_training.slurm
#
# REQUIRES, in this order:
#   1. 05_run_inference (+ merge_05_shards) -> stage2_candidates.parquet
#   2. sbatch --export=SCOPE="train" 01e_run_od_candidates.slurm
#        -> data/od_features_candidates_train/
#      This reads ONLY the training root. The holdout's OD lives in
#      data/od_features_candidates/ and must never be joined in here.
#
# Variables:
#   DRYRUN=1     report the table's shape and the true-vs-competitor OD
#                rates, write nothing. Worth doing first: if those two rates
#                are close, no re-ranker will help and the rest of this line
#                of work is not worth building.
#   DROPNOPOS=1  exclude plants whose true parcel never made the top 20.
#                Off by default -- those pools are real and their negatives
#                teach what a near-miss looks like. 07b can weight them via
#                the pool_has_positive column instead.
#   TOPK=20      must match what 01e fetched, or rows will have no OD.
#
# Memory: joins the national stage2_candidates table (240k rows) against
# per-state parcel parquet for the true-parcel lookup. 64G is ample.

source /work/GRDVULN/correction/scripts/_common.sh

CMD="python $SCRIPTS/06b_build_rerank_training.py"
[ -n "${DRYRUN:-}" ]    && CMD="$CMD --dry-run"
[ -n "${DROPNOPOS:-}" ] && CMD="$CMD --drop-no-positive"
[ -n "${TOPK:-}" ]      && CMD="$CMD --top-k $TOPK"

echo "Running: $CMD"
eval "$CMD"
RC=$?

echo "Finished: $(date)  (exit $RC)"
exit $RC
