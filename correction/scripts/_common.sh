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

ROOT=/work/GRDVULN/tp_qa/correction
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

# ==============================================================================
# DEFAULT_STATES -- the training universe, so the array wrappers do not need a
# 250-character --export on every submission.
# ==============================================================================
# Every per-state array job wants the same list, and passing it by hand meant
# either a shell variable or a long inline paste. Both fail on this cluster:
# the login shell is tcsh, where `export VAR=...` is a syntax error and
# `VAR=$(...)` does not mean what it looks like. Keeping the list here, in the
# file every wrapper already sources, removes the question entirely --
# `sbatch --array=0-50%6 01b_run_object_detection.slurm` with no --export at
# all.
#
# 51 entries: 49 states + PR. `python list_training_states.py` prints the live
# per-state counts; if it ever reports a state not listed here, add it AND bump
# the --array upper bound, because the two have to agree.
#
# DC IS DELIBERATELY EXCLUDED (2026-09-23). The District has exactly one
# treatment plant, Blue Plains, and its location is already known -- so it
# contributes nothing to training while costing a full array task. It was also
# the state that exposed the empty-shard crash in 02: no labelled plant fell
# inside a parcel, build_stage1_training built a column-less DataFrame, and the
# task died with KeyError: 'CWNS_ID'. That bug is fixed independently (a state
# with zero matches now yields an empty shard), so this exclusion is a
# modelling decision, not a workaround -- do not re-add DC expecting it to
# help.
#
# THIS IS THE LABEL UNIVERSE, NOT THE PARCEL STORE. A state is in this list
# because it has labelled plants -- which says nothing about whether Regrid
# parcels for it were ever downloaded to PARCEL_BASE. If they were not, 01b
# and 01c resolve no parcel for any plant there and quietly count every one
# into "No parcel found" while exiting 0.
#
#   python check_parcel_coverage.py
#
# reports exactly which training states have no parcel data, how many plants
# that silently costs, and prints a usable state list to override with. Run it
# before widening the state list, not after.
#
# Override for a subset the usual way, which still works:
#   sbatch --array=0-2 --export=STATES="OH MS DE" 01b_run_object_detection.slurm
DEFAULT_STATES="AK AL AR AZ CA CO CT DE FL GA HI IA ID IL IN KS KY LA MA MD ME MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA PR RI SC SD TN TX UT VA VT WA WI WV WY"
DEFAULT_STATES_N=51

# Comma-separated form. The array jobs take spaces (one state per task); the
# single jobs that build one combined table take commas. That split is real --
# see 01a's header -- so both forms live here rather than each wrapper
# reformatting the list itself.
DEFAULT_STATES_CSV="${DEFAULT_STATES// /,}"

# ==============================================================================
# INFERENCE_STATES -- DEFAULT_STATES minus AK, HI and PR (48 entries)
# ==============================================================================
# Used by 05_run_inference_array.slurm and merge_05_shards.slurm only.
#
# NAIP is CONUS-only, so AK, HI and PR have no imagery and no detection output
# (see check_od_freshness.py's 2026-09-03 note). They stay in DEFAULT_STATES
# because their labelled plants still train Stage 1/2, but they are not
# inferred on. Keeping them in the inference list made merge_05_shards refuse
# the whole national merge for three states that were never meant to run.
#
# The two lists have to be separate rather than one derived at each use site:
# the array index -> state mapping is positional, so --array=0-47 must match
# THIS list's length.
INFERENCE_STATES="AL AR AZ CA CO CT DE FL GA IA ID IL IN KS KY LA MA MD ME MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT WA WI WV WY"
INFERENCE_STATES_N=48
