#!/bin/bash
#SBATCH --job-name=corr_04
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/04_%A.log

# Stage 2 model: which nearby parcel is the true location?
# Reads  data/features/15_stage2_training.parquet
# Writes models/stage2_*.{joblib,parquet}
#
#   sbatch 04_train_stage2.slurm
#
# PREREQ: same caution as 03. The local OH-only run produced 4 positive labels
# against 73,142 negatives -- spatial CV (KMeans leave-one-cluster-out) can't
# form meaningful folds from 4 positives. Check 02's log first.

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

python "$SCRIPTS/04_train_stage2.py"

echo "Finished: $(date)"
