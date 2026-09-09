#!/bin/bash
# ==============================================================================
# check_01a_01b_complete.sh
# ==============================================================================
# Verifies a nationwide 01a/01b array run actually completed cleanly across
# all 52 states before trusting it enough to run 02 against it. Checks:
#   1. sacct state for every array task (catches FAILED/TIMEOUT/OOM/CANCELLED)
#   2. output actually exists for all 52 states on both sides
#   3. names the specific missing states (not just a count) for easy rerun
#   4. per-state fetch-failure counts from 01b logs, to spot rough states
#
# Usage:
#   sbatch --export=JOBID_01A=<jobid>,JOBID_01B=<jobid> check_01a_01b_complete.slurm
# ==============================================================================
set -uo pipefail

ROOT=/work/GRDVULN/correction

# The 52 states/territories in the training universe, per
# list_training_states.py's 2026-08-21 output. If the training universe
# changes (rebuilt training_locations.gpkg with a different plant set),
# re-run that script and update this list.
EXPECTED_STATES=(AK AL AR AZ CA CO CT DC DE FL GA HI IA ID IL IN KS KY LA MA \
    MD ME MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA PR RI SC SD TN \
    TX UT VA VT WA WI WV WY)

echo "=========================================="
echo "check_01a_01b_complete.sh"
echo "Expected states: ${#EXPECTED_STATES[@]}"
echo "=========================================="

overall_ok=1

# --- 1. sacct: any array task that didn't COMPLETE ---
check_sacct() {
    local label=$1 jobid=$2
    echo ""
    echo "--- $label (job $jobid): sacct status ---"
    if [ -z "$jobid" ]; then
        echo "  SKIPPED (no job id provided)"
        return
    fi
    # -P makes output pipe-delimited; filter on FIELD 1 (JobID) not containing
    # a "." -- .batch/.extern sub-step lines have a dot in the JobID field
    # (e.g. "7400037_3.batch"), the real per-array-task line doesn't
    # (e.g. "7400037_3"). An earlier version of this filter was backwards
    # (kept the dotted sub-step lines instead of excluding them).
    local bad
    bad=$(sacct -j "$jobid" --format=JobID,State,ExitCode -n -P 2>/dev/null \
          | awk -F'|' '$1 !~ /\./' | grep -v "COMPLETED")
    if [ -z "$bad" ]; then
        echo "  All array tasks COMPLETED."
    else
        echo "  NON-COMPLETED TASKS FOUND:"
        echo "$bad" | sed 's/^/    /'
        overall_ok=0
    fi
}
check_sacct "01a" "${JOBID_01A:-}"
check_sacct "01b" "${JOBID_01B:-}"

# --- 2/3. output existence per state, naming what's missing ---
check_output() {
    local label=$1 pattern=$2
    echo ""
    echo "--- $label: output per state ---"
    local missing=()
    for st in "${EXPECTED_STATES[@]}"; do
        local path
        path=$(eval echo "$pattern")
        if ! compgen -G "$path" > /dev/null 2>&1 && [ ! -e "$path" ]; then
            missing+=("$st")
        fi
    done
    local n_present=$((${#EXPECTED_STATES[@]} - ${#missing[@]}))
    echo "  Present: $n_present / ${#EXPECTED_STATES[@]}"
    if [ ${#missing[@]} -gt 0 ]; then
        echo "  MISSING states: ${missing[*]}"
        overall_ok=0
    fi
}
check_output "01a (nlcd_features)" "$ROOT/data/nlcd_features/nlcd_\${st}_k18.parquet"
check_output "01b (od_features/plants)" "$ROOT/data/od_features/plants/state=\${st}"

# --- 4. per-state fetch-failure summary from 01b logs ---
echo ""
echo "--- 01b: fetch failures per state (from logs) ---"
if [ -n "${JOBID_01B:-}" ]; then
    logs=("$ROOT"/logs/01b_"${JOBID_01B}"_*.log)
    if [ -e "${logs[0]}" ]; then
        for f in "${logs[@]}"; do
            state_line=$(grep "^Processing state:" "$f" | head -1)
            fail_line=$(grep "Fetch failures" "$f" | head -1)
            nothing_line=$(grep "Nothing to do" "$f" | head -1)
            if [ -n "$fail_line" ]; then
                n_fail=$(echo "$fail_line" | grep -oP '\d+' | head -1)
                if [ "${n_fail:-0}" -gt 3 ]; then
                    echo "  $state_line -- $fail_line  <-- worth a closer look"
                fi
            elif [ -n "$nothing_line" ]; then
                : # resume found everything already done in an earlier run --
                  # a valid, successful early exit, not a crash. main() returns
                  # here before ever reaching the "=== Summary ===" print, so
                  # this must be checked for explicitly rather than treating a
                  # missing summary as evidence of failure.
            elif [ -n "$state_line" ]; then
                echo "  $state_line -- NO SUMMARY FOUND IN LOG (job likely crashed before finishing)"
                overall_ok=0
            fi
        done
        echo "  (states with <=3 fetch failures not listed individually -- normal/expected)"
    else
        echo "  No log files found matching 01b_${JOBID_01B}_*.log"
    fi
else
    echo "  SKIPPED (no job id provided)"
fi

echo ""
echo "=========================================="
if [ "$overall_ok" -eq 1 ]; then
    echo "RESULT: Looks clean -- OK to proceed to 02."
else
    echo "RESULT: Issues found above -- resolve before running 02."
fi
echo "=========================================="