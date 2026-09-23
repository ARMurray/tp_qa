#!/bin/bash
#SBATCH --job-name=corr_02
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-51
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/02_%A_%a.log

# Feature engineering, ONE STATE PER ARRAY TASK.
#
# Each task writes its own shard under data/feature_shards/state=XX/ rather
# than the flat filenames. That is what --shard does, and it is not optional
# here: without it all 52 tasks write the same four files and the last one to
# finish wins, leaving feature tables that cover a single state. Nothing
# errors; 03 and 04 just train on a fiftieth of the data.
#
# THE MERGE IS PART OF THE JOB, NOT AN AFTERTHOUGHT. Submit both together so
# it cannot be forgotten:
#
#   sbatch 02_feature_engineering.slurm
#     -> prints "Submitted batch job 1234567"
#   sbatch --dependency=afterok:1234567 02b_merge_feature_shards.slurm
#
#   Written as two literal steps rather than JID=$(sbatch --parsable ...)
#   because $(...) is bash syntax and this cluster's login shell may be tcsh,
#   where it silently does the wrong thing -- and the cost of finding out is a
#   52-task array that finishes with nothing chained to it. In bash the
#   one-liner is fine:
#       JID=$(sbatch --parsable 02_feature_engineering.slurm)
#       sbatch --dependency=afterok:$JID 02b_merge_feature_shards.slurm
#
# afterok means the merge only runs if EVERY array task succeeded, which is
# exactly the condition under which merging is safe -- a merge over a partial
# array would produce a well-formed file that silently omits states.
#
# Re-running one failed state:
#   sbatch --array=35 02_feature_engineering.slurm       # just OH
#   sbatch 02b_merge_feature_shards.slurm                 # then re-merge
#
# MEMORY: 64G, down from the 200G a single national job needed. One state is
# a fraction of the national parcel set, and the whole reason for the array is
# that the peak was per-state anyway. Raise it if a large state (CA, TX) gets
# OOM-killed -- check with `seff <jobid>` after a run rather than guessing.

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

STATES=(AK AL AR AZ CA CO CT DC DE FL GA HI IA ID IL IN KS KY LA MA MD ME MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA PR RI SC SD TN TX UT VA VT WA WI WV WY)

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#STATES[@]}" ]; then
    echo "ERROR: array index ${SLURM_ARRAY_TASK_ID} is past the end of the"
    echo "       ${#STATES[@]}-element state list. Fix --array in this file."
    exit 1
fi

STATE="${STATES[$SLURM_ARRAY_TASK_ID]}"
echo "State: $STATE  (array index ${SLURM_ARRAY_TASK_ID} of ${#STATES[@]})"

CMD="python \"$SCRIPTS/02_feature_engineering.py\" --states $STATE --shard"

[ "${FULL_UNIVERSE:-0}" = "1" ] && CMD="$CMD --full-universe"

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
