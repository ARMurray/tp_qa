#!/bin/bash
#SBATCH --job-name=corr_03
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/03_%A.log

# Stage 1 model: is the reported location correct?
# Reads  data/features/14_stage1_training.parquet
# Writes models/stage1_*.{joblib,parquet}
#
#   sbatch 03_train_stage1.slurm
#
# PREREQ: 02 must have produced a stage1 table with enough of BOTH classes.
# The local OH-only run had 47 Correct / 1 Incorrect, which is not trainable --
# stratified CV will fail or produce a degenerate model. Check the class counts
# printed at the end of 02's log before submitting this.

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

python "$SCRIPTS/03_train_stage1.py"

echo "Finished: $(date)"
