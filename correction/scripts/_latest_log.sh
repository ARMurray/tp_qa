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

# Bash does not run the EXIT trap when killed by an untrapped signal, and
# SLURM's time limit arrives as SIGTERM -- turn it into an ordinary exit.
trap 'exit 143' TERM
trap '_tpqa_copy_latest_log $?' EXIT
