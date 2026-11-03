#!/bin/bash
#SBATCH --job-name=corr_01f
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/01f_%A_%a.log

# Multi-source seeds -> seed sites, one state per array task (docs/PLAN_20261103_multi_source_seeds.md).
# Output: data/seed_sites/state=XX/{seed_parcels,seed_sites}.parquet
#
#   sbatch --array=0-50%6 01f_build_seed_sites.slurm                       # training universe, all states
#   sbatch --array=0-1 --export=STATES="OH PA" 01f_build_seed_sites.slurm
#
# Inference (every treatment plant). NORESUME is needed for a state already built in training mode, for the
# same reason as 01a: resume is per state, so a training-mode file would be mistaken for a complete one.
#   sbatch --array=0-50%6 --export=FULLUNIVERSE="1",NORESUME="1" 01f_build_seed_sites.slurm
#
# Run after build_plant_seeds.py's plant_seeds.csv is in the repo, before 01e --seed-sites and 02.

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

STATES="${STATES:-$DEFAULT_STATES}"
read -ra STATE_ARR <<< "$STATES"
IDX=${SLURM_ARRAY_TASK_ID:-0}
STATE=${STATE_ARR[$IDX]}

if [ -z "$STATE" ]; then
    echo "ERROR: array index $IDX has no matching state in STATES=\"$STATES\"."
    echo "       ${#STATE_ARR[@]} state(s) given -- use --array=0-$((${#STATE_ARR[@]} - 1))"
    exit 1
fi

echo "Processing state: $STATE  (array index $IDX of ${#STATE_ARR[@]})"
echo "Seed mode: ${TPQA_SEED_MODE:-multi}"

CMD="python $SCRIPTS/01f_build_seed_sites.py --states $STATE"
[ "${FULLUNIVERSE:-0}" = "1" ] && CMD="$CMD --full-universe"
[ "${NORESUME:-0}" = "1" ]     && CMD="$CMD --no-resume"

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
