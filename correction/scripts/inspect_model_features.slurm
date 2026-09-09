#!/bin/bash
#SBATCH --job-name=corr_inspect_features
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/inspect_features_%A.log

# Audits each deployed model's ACTUAL fitted feature set against the table
# that produced it -- cardinality/row-count ratio test for scaffolding
# columns (IDs, geometry blobs, merge-suffix artifacts), not a name blocklist.
#
#   sbatch inspect_model_features.slurm
#   sbatch --export=STAGES="stage1,stage2" inspect_model_features.slurm

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

CMD="python $SCRIPTS/inspect_model_features.py"
[ -n "${STAGES:-}" ] && CMD="$CMD --stages $STAGES"

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
