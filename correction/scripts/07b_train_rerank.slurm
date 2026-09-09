#!/bin/bash
#SBATCH --job-name=corr_07b_rerank
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/07b_rerank_%A.log

# Trains the Stage 2a re-ranker on 17_rerank_training.parquet.
#
#   sbatch 07b_train_rerank.slurm                        # the model
#   sbatch --export=NOOD="1" 07b_train_rerank.slurm      # ablation baseline
#   sbatch --export=SEED="7" 07b_train_rerank.slurm      # a different seed
#
# RUN BOTH. The --no-od run is the point of the exercise: if recall@1 is the
# same with and without OD features, then object detection is not steering
# the ranking and the 48-hour national 01e fetch is not justified. The
# holdout measured true parcels firing at 46.4% vs competitors at 9.0%
# (diagnose_candidate_od.py, 2026-09-08), so a gap is expected -- but
# expected is not measured.
#
# NOOD artifacts carry a _noOD suffix and never overwrite the real model.
#
# THEN RUN SEVERAL SEEDS. The test split is a few dozen plants, so one
# seed's recall@1 carries a standard error around +/-8pp. A single run
# showing "+5pp from OD" is not evidence. Three to five seeds and the spread
# is.
#
# Variables:
#   NOOD=1        drop every od_* feature
#   SEED=n        random_state for the split and the forest (default 42)
#   NITER=n       RandomizedSearchCV iterations (default 20)
#   DROPNOPOS=1   train only on pools containing the true parcel. Off by
#                 default -- near-miss pools are real at inference.
#   TESTFRAC=f    plant-level test fraction (default 0.25)

source /work/GRDVULN/correction/scripts/_common.sh

CMD="python $SCRIPTS/07b_train_rerank.py"
[ -n "${NOOD:-}" ]      && CMD="$CMD --no-od"
[ -n "${SEED:-}" ]      && CMD="$CMD --seed $SEED"
[ -n "${NITER:-}" ]     && CMD="$CMD --n-iter $NITER"
[ -n "${DROPNOPOS:-}" ] && CMD="$CMD --drop-no-positive-pools"
[ -n "${TESTFRAC:-}" ]  && CMD="$CMD --test-frac $TESTFRAC"

echo "Running: $CMD"
eval "$CMD"
RC=$?

echo "Finished: $(date)  (exit $RC)"
exit $RC
