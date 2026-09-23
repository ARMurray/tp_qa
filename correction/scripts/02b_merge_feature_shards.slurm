#!/bin/bash
#SBATCH --job-name=corr_02b
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/02b_merge_%A.log

# Unions 02's per-state shards into the flat feature tables that 03, 04, 05,
# 06, 06b and 10 all read by name.
#
# Submit it chained to the array so it cannot be forgotten:
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
# afterok is the right dependency: merging a partial array produces a file
# that looks fine and silently omits whole states.
#
# STATES defaults to _common.sh's DEFAULT_STATES -- the same 51-element list
# 02's array uses. This used to keep its own copy, which is how a list gets
# to disagree with itself. Override when
# the array covered a subset:
#
#   sbatch --export=STATES="OH PA WI" 02b_merge_feature_shards.slurm
#   sbatch --export=ALLPRESENT=1 02b_merge_feature_shards.slurm   # whatever is there
#
# Naming the states is better than ALLPRESENT: --all-present cannot tell a
# state whose task failed from one you never asked for.
#
# MEMORY: 128G. This holds the concatenated 10_parcel_features in memory,
# which was 77.7M rows nationally as of 2026-09-04 and is the largest single
# object anywhere in the pipeline.

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh


CMD="python \"$SCRIPTS/02b_merge_feature_shards.py\""

if [ "${ALLPRESENT:-0}" = "1" ]; then
    CMD="$CMD --all-present"
else
    CMD="$CMD --states \"${STATES:-$DEFAULT_STATES}\""
fi

[ -n "${DRYRUN:-}" ] && CMD="$CMD --dry-run"

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
