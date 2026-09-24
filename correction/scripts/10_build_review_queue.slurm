#!/bin/bash
#SBATCH --job-name=corr_10_review_queue
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/10_review_queue_%A.log

# Builds data/review_queue/review_queue_round{N}.parquet from 05's inference
# output. See script docstring for the Stage2b->Stage2a substitution and the
# two review-task-type design (candidate_pick / confirm_reported).
#
#   sbatch --export=ROUND=3 10_build_review_queue.slurm          # all 48 states
#   sbatch --export=ROUND=3,BATCHSIZE=150 10_build_review_queue.slurm
#   sbatch --export=ROUND=3 10_build_review_queue.slurm "OH,MS,DE"   # subset
#
# ROUND is required. STATES defaults to _common.sh's INFERENCE_STATES -- the
# 48 states 05_run_inference_array covers, so it matches what was inferred.
# A subset is a comma list, POSITIONAL, because sbatch's --export splits on
# commas and would truncate it. The old form, positional states then round
# (sbatch 10_build_review_queue.slurm "OH,MS,DE" 1), still works.
#
# PRIOR-ROUND EXCLUSION: 10 skips every plant in
# data/review_log_incoming/review_log_round*.parquet and
# holdout_truth_round*.parquet. Those files come from the review machine via
# git (review_app/data/outgoing/), so this wrapper copies them from the repo
# before running -- nothing to do by hand. Pull first so the copies are
# current.
#
# Requires 05_run_inference.py to have already run for these exact states --
# reads data/inference/plant_summary.parquet and stage2_candidates.parquet
# directly, does not regenerate them.

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

ROUND_ARG="${ROUND:-${2:-}}"
if [ -z "$ROUND_ARG" ]; then
    echo "ERROR: set the round, e.g. sbatch --export=ROUND=3 10_build_review_queue.slurm"
    exit 1
fi
STATES_ARG="${1:-${INFERENCE_STATES// /,}}"

# Prior-round verdicts, from the repo copy of the review machine's outgoing/.
REPO_OUTGOING="$ROOT/../review_app/data/outgoing"
INCOMING="$ROOT/data/review_log_incoming"
mkdir -p "$INCOMING"
n_copied=0
for f in "$REPO_OUTGOING"/review_log_round*.parquet "$REPO_OUTGOING"/holdout_truth_round*.parquet; do
    [ -e "$f" ] || continue
    case "$(basename "$f")" in review_log_candidates_*) continue ;; esac
    cp -f "$f" "$INCOMING"/ && n_copied=$((n_copied + 1))
done
echo "Prior-round review files copied into $INCOMING: $n_copied"
[ "$n_copied" -eq 0 ] && echo "WARNING: none found in $REPO_OUTGOING -- earlier rounds' plants can be re-queued."

CMD="python $SCRIPTS/10_build_review_queue.py --states $STATES_ARG --round $ROUND_ARG"
[ -n "${BATCHSIZE:-}" ]     && CMD="$CMD --batch-size $BATCHSIZE"
[ -n "${MODELVERSION:-}" ]  && CMD="$CMD --model-version $MODELVERSION"
[ -n "${SEED:-}" ]          && CMD="$CMD --seed $SEED"

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
