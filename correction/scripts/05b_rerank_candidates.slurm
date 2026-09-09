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
#   sbatch 05b_rerank_candidates.slurm                       # with OD
#   sbatch --export=NOOD="1" 05b_rerank_candidates.slurm     # ablation model
#   sbatch --export=STATES="OH,WI,KS" 05b_rerank_candidates.slurm
#
# This is CHEAP -- it loads a model and scores rows. Minutes, not hours. The
# expensive prerequisite is the imagery: 01e --all over 240k candidates, run
# via 01e_run_od_candidates_array.slurm.
#
# Variables:
#   NOOD=1        use rerank_rf_model_noOD.joblib and skip the OD join.
#                 Needs no imagery at all and is worth roughly 60% of the
#                 full gain (+8.2pp of the +14.8pp recall@1 over Stage 2a,
#                 means across seeds 42/7/13/99). The deployable-today path
#                 while the 01e array is still running.
#   STATES=...    comma-separated. Use this to score states the 01e array
#                 has already finished, rather than waiting for all 48.
#   TOPK=20       must match what 06b/07b trained on
#   NOFALLBACK=1  do not fall back to Stage 2a ordering for pools where no
#                 candidate fired. Off by default: with nothing detected,
#                 every OD column is constant within that competition and
#                 carries no ranking information.
#
# OUTPUT keeps BOTH scores per candidate, so a reviewer can see what Stage
# 2a thought and what the re-ranker thought, and plants resolved by fallback
# are flagged with rerank_fallback.

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

CMD="python $SCRIPTS/05b_rerank_candidates.py"
[ -n "${NOOD:-}" ]       && CMD="$CMD --no-od"
[ -n "${STATES:-}" ]     && CMD="$CMD --states $STATES"
[ -n "${TOPK:-}" ]       && CMD="$CMD --top-k $TOPK"
[ -n "${NOFALLBACK:-}" ] && CMD="$CMD --no-fallback"

echo "Running: $CMD"
eval "$CMD"
RC=$?

echo "Finished: $(date)  (exit $RC)"
exit $RC
