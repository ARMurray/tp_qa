#!/bin/bash
#SBATCH --job-name=corr_07_stage2b
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/07_stage2b_%A.log

# Stage 2b: OD-aware final-candidate-selection model.
# Reads  data/features/16_stage2b_training.parquet
# Writes models/stage2b_*.{joblib,parquet}
#
#   sbatch 07_train_stage2b.slurm
#   sbatch --export=SEED="42" 07_train_stage2b.slurm
#   sbatch --export=NOOD="1" 07_train_stage2b.slurm      # non-OD baseline
#
# Variables (all optional):
#   SEED              random_state for RF + train/test split (default 42)
#   NITER             random search iterations               (default 20)
#   NJOBS             parallelism                            (default -1)
#   NOOD=1            OD ablation. Writes _noOD artifacts, never overwrites
#                     the real model.
#   ALLOWNOHOLDOUT=1  proceed without a holdout manifest. Only for deliberate
#                     pre-holdout runs -- a missing manifest should normally
#                     stop the job rather than silently train on the
#                     evaluation set.
#
# Small training set (hundreds of rows, not thousands) -- fast to train, but
# expect noisier metrics than Stage 1/2a. A single seed's test metrics are not
# a result. Run several and report the spread:
#
#   for s in 42 43 44 45 46; do
#     sbatch --export=SEED="$s" 07_train_stage2b.slurm
#   done
#
# NOTE: every seed writes to the SAME model/artifact paths, so a multi-seed
# sweep leaves only the last-finishing run's files behind. That is fine when
# you are reading the spread out of the logs, which is the intended use --
# but do not treat the surviving stage2b_rf_model.joblib as "the" model from
# a sweep. Pick a seed and run it alone to produce the artifact you keep.

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

CMD="python $SCRIPTS/07_train_stage2b.py"
[ -n "${SEED:-}" ]              && CMD="$CMD --seed $SEED"
[ -n "${NITER:-}" ]             && CMD="$CMD --n-iter $NITER"
[ -n "${NJOBS:-}" ]             && CMD="$CMD --n-jobs $NJOBS"
[ "${NOOD:-0}" = "1" ]          && CMD="$CMD --no-od"
[ "${ALLOWNOHOLDOUT:-0}" = "1" ] && CMD="$CMD --allow-no-holdout"

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
