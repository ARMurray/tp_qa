#!/bin/bash
#SBATCH --job-name=corr_01e_arr
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/01e_arr_%A_%a.log

# Per-state candidate object detection, one array task per state. This is the
# national SCOPE=all run, split so it can actually finish.
#
#   sbatch --array=0-47%6 01e_run_od_candidates_array.slurm "AL AR AZ ... WY"
#   sbatch --array=0-2%3 --export=WORKERS="16" 01e_run_od_candidates_array.slurm "OH WI KS"
#
# WHY AN ARRAY RATHER THAN ONE JOB
#   240,546 candidates at the measured 39/min is ~100 hours -- past any
#   reasonable walltime, and a single job that dies at hour 90 loses
#   everything. Per-state tasks give failure isolation, per-state resume,
#   and some real parallelism.
#
#   No merge step is needed, unlike 05: 01e writes to
#   od_features_candidates/candidates/state=XX/ already, so tasks never
#   touch the same files.
#
# THROUGHPUT IS RATE-LIMITED, NOT CPU-LIMITED
#   Two measured points, both at 32 workers in ONE job:
#     880 candidates  (holdout) -> 85/min, one 429
#     6,555 candidates (train)  -> 39/min, 429s throughout
#   Same worker count, half the throughput. That is an account-level
#   sustained rate limit, not a concurrency limit, which means N concurrent
#   tasks does NOT give N times the throughput. Past some point more
#   concurrency just buys exponential backoff and runs SLOWER.
#
#   So: do not max the throttle. Start at %4 or %6 and MEASURE. Aggregate
#   throughput is what matters -- sum the per-task rates from the logs, not
#   any single task's.
#
#   FIND THE CEILING FIRST. Run three mid-size states at one throttle, then
#   again at another, and compare aggregate candidates/min:
#     sbatch --array=0-2%3 01e_run_od_candidates_array.slurm "IA MN MO"
#   Three hours of testing can save forty hours of backoff.
#
# THROTTLE AND WORKERS TRADE OFF
#   Total concurrent requests is roughly (throttle x WORKERS). The single-job
#   runs used 32 and hit limits, so aim near or below that in aggregate:
#     %4  x WORKERS=8   = 32 concurrent   conservative, closest to measured
#     %6  x WORKERS=16  = 96 concurrent   likely over the limit -- measure
#   Lower WORKERS when raising the throttle.
#
# Variables:
#   WORKERS=8     NAIP fetch threads PER TASK (default 8, not 01e's 32 --
#                 there are now several tasks running at once)
#   TOPK=20       candidates per plant. Must match what 06b/07b trained on.
#   NORESUME=1    reprocess candidates already written. Needed after an OD
#                 model swap -- resume is keyed on (CWNS_ID, ll_uuid) and
#                 cannot tell which model wrote a row. NOT needed to restart
#                 a timed-out task: leave it off and it picks up where it
#                 stopped.
#   DRYRUN=1      report the per-state work plan, fetch nothing
#
# RESTARTING FAILED TASKS
#   sacct -j <arrayjobid> --format=JobID,State,Elapsed -X | grep -v COMPLETED
#   then resubmit just those indices, WITHOUT NORESUME, e.g.
#   sbatch --array=12,31,40 01e_run_od_candidates_array.slurm "AL AR ... WY"
#
# WALLTIME: 12h per task. TX and CA are the risk -- at 39/min a state with
# 20,000 candidates needs ~8.5 hours. If one times out, resubmit that index
# and resume continues from the last flush.

source /work/GRDVULN/correction/scripts/_common.sh

STATES_ARG="${1:?ERROR: pass states as a positional arg, e.g. sbatch --array=0-47%6 01e_run_od_candidates_array.slurm \"AL AR AZ\"}"
read -ra STATE_LIST <<< "$STATES_ARG"
STATE="${STATE_LIST[$SLURM_ARRAY_TASK_ID]}"

if [ -z "$STATE" ]; then
    echo "ERROR: array index $SLURM_ARRAY_TASK_ID has no state."
    echo "  ${#STATE_LIST[@]} state(s) given -- use --array=0-$(( ${#STATE_LIST[@]} - 1 ))"
    exit 1
fi

echo "Processing state: $STATE  (array index $SLURM_ARRAY_TASK_ID of ${#STATE_LIST[@]})"

CMD="python $SCRIPTS/01e_run_od_candidates.py --all --states $STATE"
CMD="$CMD --workers ${WORKERS:-8}"
[ -n "${TOPK:-}" ]     && CMD="$CMD --top-k $TOPK"
[ -n "${NORESUME:-}" ] && CMD="$CMD --no-resume"
[ -n "${DRYRUN:-}" ]   && CMD="$CMD --dry-run"

echo "Running: $CMD"
START=$(date +%s)
eval "$CMD"
RC=$?
ELAPSED=$(( ($(date +%s) - START) / 60 ))

echo "Finished: $(date)  (exit $RC, ${ELAPSED} min)"
echo "THROUGHPUT NOTE: divide this task's 'Candidates processed' by ${ELAPSED}"
echo "  min, then SUM across concurrent tasks to get aggregate rate. Compare"
echo "  that against the 39/min single-job baseline before raising the throttle."
exit $RC
