#!/bin/bash
#SBATCH --job-name=corr_05b_rerank
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/05b_rerank_%A.log

# Scores stage2_candidates.parquet with the re-ranker, writing rerank_score
# alongside the existing stage2_prob_correct.
#
#   sbatch 05b_rerank_candidates.slurm "AZ,AR,CA"
#   sbatch --export=NOOD="1" 05b_rerank_candidates.slurm "AZ,AR,CA"
#
# STATES is now POSITIONAL, not --export -- sbatch's --export parser
# truncates at the first comma in a value, which silently dropped every
# state after the first when STATES was passed as --export=STATES="...".
# Same fix as 10_build_review_queue.slurm already uses.

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

STATES_ARG="${1:?ERROR: pass states as a positional arg, e.g. sbatch 05b_rerank_candidates.slurm \"AZ,AR,CA\"}"

CMD="python $SCRIPTS/05b_rerank_candidates.py --states $STATES_ARG"
[ -n "${NOOD:-}" ]       && CMD="$CMD --no-od"
[ -n "${TOPK:-}" ]       && CMD="$CMD --top-k $TOPK"
[ -n "${NOFALLBACK:-}" ] && CMD="$CMD --no-fallback"

echo "Running: $CMD"
eval "$CMD"
RC=$?

echo "Finished: $(date)  (exit $RC)"
exit $RC