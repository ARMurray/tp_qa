#!/bin/bash
# ==============================================================================
# _common.sh -- shared preamble sourced by every correction pipeline job
# ==============================================================================
# Not submitted directly. Each 0Nx_*.slurm does:
#     source /work/GRDVULN/tp_qa/correction/scripts/_common.sh
#
# Keeps the module-load + venv-activate + sanity-check logic in ONE place so
# the five job scripts can't drift out of sync with each other.
# ==============================================================================

ROOT=/work/GRDVULN/correction
SCRIPTS="$ROOT/scripts"
VENV_DIR="$ROOT/.venv"

echo "=========================================="
echo "Node:    ${SLURMD_NODENAME:-unknown}"
echo "Job ID:  ${SLURM_JOB_ID:-none}"
echo "CPUs:    ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Started: $(date)"
echo "=========================================="

for modname in python/3.11 python/3.10 python/3.9 python3.11 python3.10 python3.9; do
    if module load "$modname" 2>/dev/null; then
        echo "Loaded module: $modname"
        break
    fi
done

if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: venv not found at $VENV_DIR"
    echo "Run 'bash $SCRIPTS/setup_env.sh' on a login node first."
    exit 1
fi

source "$VENV_DIR/bin/activate"

python -c "import sys; assert sys.version_info >= (3,9), f'Python too old: {sys.version}'" || {
    echo "ERROR: venv python is too old -- rebuild it with setup_env.sh"
    exit 1
}

echo "Python:  $(python --version 2>&1)  ($(which python))"
echo "------------------------------------------"

# Fail the job if any command in the pipeline fails, rather than marching on
# and writing partial output that looks like a successful run.
set -o pipefail
