#!/bin/bash
#SBATCH --job-name=corr_01b
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/01b_%A_%a.log

# Object detection over candidate parcels, streaming NAIP from Planetary
# Computer. Output: data/od_features/{tiles,detections,objects,plants}/state=XX/
#
#   sbatch --array=0-1 --export=STATES="OH PA" 01b_run_object_detection.slurm
#   sbatch --export=STATES="OH",LIMIT="50" 01b_run_object_detection.slurm
#
# INFERENCE (full universe -- every treatment plant, not just labeled ones):
#   sbatch --array=0-2%2 --export=STATES="OH MS DE",FULLUNIVERSE="1" \
#          01b_run_object_detection.slurm
#
# Variables:
#   STATES            REQUIRED. Space-separated (see 01a's header for why).
#   FULLUNIVERSE=1    run OD on EVERY treatment plant in the state, not just
#                     the labeled training universe. Needed for inference.
#   LIMIT=N           stop after N plants. Use for a smoke test first.
#   NORESUME=1        force full reprocess (see below -- rarely what you want).
#
# RESUME IS ON BY DEFAULT: if this job hits the time limit or gets preempted,
# resubmit the identical command -- plants already written to plants/state=XX/
# are skipped automatically. Set NORESUME=1 to force a full reprocess.
#
# LEAVE RESUME ON FOR FULLUNIVERSE. Unlike 01a (which resumes at whole-state
# granularity and therefore needs NORESUME to widen a state), 01b resumes PER
# PLANT. A full-universe run over a state that already has training-universe
# output will skip the plants it already has and process only the ones full
# universe adds -- which is exactly right, and on a large state it is most of
# the saving. NORESUME=1 would re-fetch and re-detect every already-done plant
# for nothing.
#
# THROTTLING: Planetary Computer rate-limits. 52 concurrent array tasks at 32
# workers each was enough to trip it even with retry logic. Three states is
# well under that, but %2 in the --array spec is cheap insurance if a big
# state's full universe turns out to be much larger than expected.
#
# --workers is bound to the CPU allocation, but this is I/O-bound work, so the
# two aren't really coupled; see config.py NAIP_WORKERS note.

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

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

echo "Checking outbound HTTPS to Planetary Computer..."
curl -s -o /dev/null -w "  STAC API HTTP status: %{http_code}\n" \
    https://planetarycomputer.microsoft.com/api/stac/v1 \
    || echo "  curl failed -- outbound HTTPS may be blocked on this node"

CMD="python $SCRIPTS/01b_run_object_detection.py --states $STATE --workers ${SLURM_CPUS_PER_TASK:-32}"
[ "${FULLUNIVERSE:-0}" = "1" ] && CMD="$CMD --full-universe"
[ -n "${LIMIT:-}" ]            && CMD="$CMD --limit $LIMIT"
[ "${NORESUME:-0}" = "1" ]     && CMD="$CMD --no-resume"

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
