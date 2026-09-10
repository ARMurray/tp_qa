#!/bin/bash
#SBATCH --job-name=check_review_queue
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/check_review_queue_%A.log

# Verifies review_queue_round{N}.parquet carries stage2a_score, stage2b_score,
# and rerank_fallback for candidate_pick rows.
#
#   sbatch check_review_queue_scores.slurm 2

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

ROUND_ARG="${1:?ERROR: pass round number as a positional arg, e.g. sbatch check_review_queue_scores.slurm 2}"

echo "Running: python $SCRIPTS/check_review_queue_scores.py --round $ROUND_ARG"
python "$SCRIPTS/check_review_queue_scores.py" --round "$ROUND_ARG"

echo "Finished: $(date)"
