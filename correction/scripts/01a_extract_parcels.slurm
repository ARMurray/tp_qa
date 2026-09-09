#!/bin/bash
#SBATCH --job-name=corr_01a
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/01a_%A_%a.log

# Candidate parcel extraction + NLCD zonal stats, one state per array task.
# Output: data/nlcd_features/nlcd_{STATE}_k{K}.parquet
#
# Submit as an array, one task per state:
#   sbatch --array=0-1 --export=STATES="OH PA" 01a_extract_parcels.slurm
#
# Or a single state:
#   sbatch --export=STATES="OH" 01a_extract_parcels.slurm
#
# Force a full reprocess even if the output parquet already exists:
#   sbatch --export=STATES="OH",NORESUME="1" 01a_extract_parcels.slurm
#
# INFERENCE (full universe -- every treatment plant, not just labeled ones):
#   sbatch --array=0-2 --export=STATES="OH MS DE",FULLUNIVERSE="1",NORESUME="1" \
#          01a_extract_parcels.slurm
#
# Variables:
#   STATES            REQUIRED. Space-separated. Spaces, not commas: sbatch's
#                     --export uses commas to separate its own name=value
#                     pairs and would truncate a comma list at the first one.
#   FULLUNIVERSE=1    process EVERY treatment plant in the state, not just the
#                     labeled training universe. Needed for inference.
#   NORESUME=1        reprocess even if the state's output parquet exists.
#
# WHY FULLUNIVERSE ALSO NEEDS NORESUME
# -------------------------------------
# 01a resumes at WHOLE-STATE granularity -- it skips a state entirely if
# nlcd_{STATE}_k{K}.parquet already exists (01a_extract_parcels.py:248). A
# state processed in training-universe mode therefore has an output file that
# looks complete but covers only labeled plants, and a later FULLUNIVERSE
# submission would skip it and silently produce nothing new. Pair the two
# flags. (01b is different -- it resumes PER PLANT, so leave resume on there
# and it will process only the plants full-universe mode adds.)
#
# ORDERING TRAP: 01d appends top-up rows into these same per-state files.
# NORESUME=1 overwrites the file wholesale, discarding them. Run 01d AFTER any
# full-universe rebuild of a state, never before.
#
# Memory is set high because NLCD zonal extraction over a state's candidate
# parcels is the memory-heaviest step in the pipeline. Drop it if a state
# runs comfortably.

source /work/GRDVULN/correction/scripts/_common.sh

STATES=${STATES:?ERROR: set STATES, e.g. --export=STATES="OH PA"}
read -ra STATE_ARR <<< "$STATES"
IDX=${SLURM_ARRAY_TASK_ID:-0}
STATE=${STATE_ARR[$IDX]}

if [ -z "$STATE" ]; then
    echo "ERROR: array index $IDX has no matching state in STATES=\"$STATES\"."
    echo "       ${#STATE_ARR[@]} state(s) given -- use --array=0-$((${#STATE_ARR[@]} - 1))"
    exit 1
fi

echo "Processing state: $STATE  (array index $IDX of ${#STATE_ARR[@]})"

CMD="python $SCRIPTS/01a_extract_parcels.py --states $STATE"
[ "${FULLUNIVERSE:-0}" = "1" ] && CMD="$CMD --full-universe"
[ "${NORESUME:-0}" = "1" ]     && CMD="$CMD --no-resume"

if [ "${FULLUNIVERSE:-0}" = "1" ] && [ "${NORESUME:-0}" != "1" ]; then
    echo "WARNING: FULLUNIVERSE=1 without NORESUME=1. If this state already has"
    echo "         a training-universe output file, it will be SKIPPED and no"
    echo "         full-universe candidates will be produced. See header."
fi

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
