#!/bin/bash
#SBATCH --job-name=corr_05_arr
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/05_arr_%A_%a.log

# Per-state Stage 1 -> Stage 2a inference, one array task per state.
#
#   sbatch --array=0-47%8 05_run_inference_array.slurm "AL AR AZ ... WY"
#
# STATES here is SPACE-separated and POSITIONAL, matching 05_run_inference.slurm's
# positional convention. (05_run_inference.py itself still wants a comma-separated
# --states; this script hands it exactly one state, so the distinction is moot.)
#
# WHY AN ARRAY
# A single national run OOM-killed at 64G on 2026-09-07: Stage 2a accumulates
# every state's candidates in one frame before scoring, and CA alone produced
# 5.6M. Extrapolating 16,198 plants gives roughly 180M rows -- not an allocation
# problem, a design one. Per-state tasks cap the candidate frame at one state's
# worth, which CA cleared in 7 minutes at 64G.
#
# REQUIRES the --out-dir patch (patch_05_for_array.py). Without it every task
# writes DATA_DIR/inference/{plant_summary,stage2_candidates}.parquet and the
# last finisher silently wins. The guard below refuses to run if it is missing.
#
# THROTTLE: %8 is a starting point. Each task loads the state-filtered parcel
# table plus its own candidates; 8 concurrent at 64G is 512G of cluster memory.
# Lower it if the partition is busy.
#
# AFTER the array finishes, merge the shards into the two flat files
# 10_build_review_queue.py expects:
#     sbatch merge_05_shards.slurm

source /work/GRDVULN/correction/scripts/_common.sh

STATES_ARG="${1:?ERROR: pass states as a positional arg, e.g. sbatch --array=0-47%8 05_run_inference_array.slurm \"AL AR AZ\"}"
read -ra STATE_LIST <<< "$STATES_ARG"
STATE="${STATE_LIST[$SLURM_ARRAY_TASK_ID]}"

if [ -z "$STATE" ]; then
    echo "ERROR: array index $SLURM_ARRAY_TASK_ID has no state."
    echo "  ${#STATE_LIST[@]} state(s) given -- use --array=0-$(( ${#STATE_LIST[@]} - 1 ))"
    exit 1
fi

if ! grep -q -- "--out-dir" "$SCRIPTS/05_run_inference.py"; then
    echo "ERROR: 05_run_inference.py has no --out-dir option."
    echo "  Run 'python $SCRIPTS/patch_05_for_array.py' first, or every array"
    echo "  task will overwrite the same two output files."
    exit 1
fi

OUT_DIR="$ROOT/data/inference_shards/state=$STATE"
echo "Processing state: $STATE  (array index $SLURM_ARRAY_TASK_ID of ${#STATE_LIST[@]})"
echo "Output: $OUT_DIR"

CMD="python $SCRIPTS/05_run_inference.py --states $STATE --out-dir $OUT_DIR"
[ -n "${TOPK:-}" ] && CMD="$CMD --top-k $TOPK"

echo "Running: $CMD"
eval "$CMD"
RC=$?

echo "Finished: $(date)  (exit $RC)"
exit $RC
