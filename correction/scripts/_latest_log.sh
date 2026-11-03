#!/bin/bash
# ==============================================================================
# _latest_log.sh -- keep a current copy of every job's log in the repo
# ==============================================================================
# Sourced at the top of _common.sh (and by the few .slurm files that do not
# use _common.sh). When the job ends -- success, failure, or SLURM's SIGTERM at
# the time limit -- the job's own log (#SBATCH --output, untouched in
# correction/logs/) is copied to
#
#     correction/diagnostics/latest_logs/<log name without the job id>.log
#
# overwriting the previous run's copy. So 03_7544427.log lands as 03.log, an
# array task 02_7544427_34.log as 02_34.log. The folder always holds the most
# recent log of every step and never grows, so it can be committed and pushed
# after each step instead of running collect_logs.slurm (2026-10-08).
#
# A SIGKILL (scancel -s KILL, node failure, or the end of SLURM's KillWait
# after the time limit) cannot be trapped; that run leaves no copy and the
# previous run's copy stays. The header line of each copy says which job it is.
# ==============================================================================

TPQA_LATEST_LOG_DIR=/work/GRDVULN/tp_qa/correction/diagnostics/latest_logs

_tpqa_copy_latest_log() {
    local status=$1
    [ -n "${SLURM_JOB_ID:-}" ] || return 0
    local src
    src=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | sed -n 's/^StdOut=//p' | head -n 1)
    [ -n "$src" ] || return 0
    # Older SLURM reports the --output pattern unresolved.
    src=${src//%A/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}
    src=${src//%a/${SLURM_ARRAY_TASK_ID:-}}
    src=${src//%j/$SLURM_JOB_ID}
    src=${src//%x/${SLURM_JOB_NAME:-}}
    [ -f "$src" ] || return 0

    local name
    name=$(basename "$src")
    [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && name=${name//_${SLURM_ARRAY_JOB_ID}/}
    name=${name//_${SLURM_JOB_ID}/}

    mkdir -p "$TPQA_LATEST_LOG_DIR" 2>/dev/null || return 0
    {
        echo "# latest copy of $src"
        echo "# job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+ task $SLURM_ARRAY_TASK_ID} on ${SLURMD_NODENAME:-?}, exit status $status, copied $(date '+%Y-%m-%d %H:%M:%S')"
        echo "#"
        cat "$src"
    } > "$TPQA_LATEST_LOG_DIR/$name.tmp" 2>/dev/null \
        && mv -f "$TPQA_LATEST_LOG_DIR/$name.tmp" "$TPQA_LATEST_LOG_DIR/$name"
    return 0
}

# Multi-source runs: refresh correction/diagnostics/multi_status.txt (one file saying where every step stands)
# after the log copy, so it is current whenever a job ends. flock keeps array tasks from running it at once; a
# task that finds it busy skips -- the next one to end refreshes it. Does nothing in cwns_only mode, and never
# affects the job's exit status. Pushing it is manual: bash push_diagnostics.sh
_tpqa_refresh_status() {
    [ "${TPQA_SEED_MODE:-multi}" = "multi" ] || return 0
    command -v python >/dev/null 2>&1 || return 0
    command -v flock >/dev/null 2>&1 || return 0
    ( flock -n 9 || exit 0
      timeout 240 python /work/GRDVULN/tp_qa/correction/scripts/multi_status.py --quiet
    ) 9>/work/GRDVULN/tp_qa/correction/logs/.multi_status.lock >/dev/null 2>&1
    return 0
}

# ...and push it, with the logs, so the results are on GitHub when the job is done (multi mode, best effort).
# Commits ONLY correction/diagnostics/ (a path-limited commit: anything else staged or modified is left alone) and
# pushes; if the push is rejected because the remote moved, it pulls --rebase once and retries. An array commits
# once, from the last task still in the queue (a race between two tasks ending together can skip it -- then run
# `bash push_diagnostics.sh`). Needs the compute node to reach GitHub with the same credentials as the login node;
# when it cannot, nothing breaks and the log and status file are simply not pushed until push_diagnostics.sh runs.
# TPQA_NO_AUTOPUSH=1 turns it off.
_tpqa_push_diagnostics() {
    [ "${TPQA_SEED_MODE:-multi}" = "multi" ] || return 0
    [ "${TPQA_NO_AUTOPUSH:-0}" = "1" ] && return 0
    command -v git >/dev/null 2>&1 || return 0
    local repo="${TPQA_REPO:-/work/GRDVULN/tp_qa}"
    if [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && command -v squeue >/dev/null 2>&1; then
        local left
        left=$(squeue -h -j "$SLURM_ARRAY_JOB_ID" -t PENDING,RUNNING 2>/dev/null | wc -l)
        [ "${left:-1}" -le 1 ] || return 0
    fi
    (
        if command -v flock >/dev/null 2>&1; then flock -n 8 || exit 0; fi
        cd "$repo" 2>/dev/null || exit 0
        git add correction/diagnostics >/dev/null 2>&1
        git diff --cached --quiet -- correction/diagnostics && exit 0
        timeout 60 git commit -q -m "diagnostics: ${SLURM_JOB_NAME:-job} ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}} ($(date '+%Y-%m-%d %H:%M'))" -- correction/diagnostics || exit 0
        timeout 120 git push -q origin HEAD || {
            timeout 120 git pull --rebase --autostash -q && timeout 120 git push -q origin HEAD
        }
    ) 8>"${TPQA_PUSH_LOCK:-/work/GRDVULN/tp_qa/correction/logs/.push_diagnostics.lock}" >/dev/null 2>&1
    return 0
}

# Bash does not run the EXIT trap when killed by an untrapped signal, and
# SLURM's time limit arrives as SIGTERM -- turn it into an ordinary exit.
trap 'exit 143' TERM
trap '_tpqa_copy_latest_log $?; _tpqa_refresh_status; _tpqa_push_diagnostics' EXIT
