#!/bin/bash -l
#SBATCH --mem=24G
#SBATCH --output=/work/GRDVULN/infrastructure/logs/train_%x_%A.out
#SBATCH --error=/work/GRDVULN/infrastructure/logs/train_%x_%A.err
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
# Usage: sbatch --job-name=train_39 submit_manifest_pipeline.sh 39
# (--job-name makes the log filenames identify the state, same convention as
#  submit_state_pipeline.sh)
#
# Optional second arg pins a non-default manifest, e.g. a Stage 3 shortlist:
#   sbatch --job-name=s3_39 submit_manifest_pipeline.sh 39 /path/to/shortlist.parquet
#
# Much smaller than the production state pipeline -- the training set is
# ~2,000-2,400 parcels NATIONALLY, so most states are 10-60 samples and
# finish in minutes. --mem/--time are deliberately loose for the first runs;
# tighten after `seff <jobid>` on a couple of real states.

STATE_FIPS=$1
MANIFEST=${2:-/work/GRDVULN/infrastructure/data/samples/training_manifest.parquet}

if [ -z "$STATE_FIPS" ]; then
    echo "Usage: sbatch --job-name=train_<FIPS> submit_manifest_pipeline.sh <2-digit FIPS> [manifest.parquet]"
    exit 1
fi

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found: $MANIFEST"
    echo "Run submit_build_manifest.sh first."
    exit 1
fi

module load python/3.13
source /work/GRDVULN/infrastructure/envs/naip_pipeline_env/bin/activate

python /work/GRDVULN/infrastructure/pipeline/09_run_manifest_pipeline_hpc.py \
    --state "$STATE_FIPS" \
    --manifest "$MANIFEST" \
    --workers 10 \
    --device cpu

# No --weights: 09 auto-selects the newest .pt in models/, same as 07. To pin
# a specific model for comparison across rounds:
#   --weights /work/GRDVULN/infrastructure/models/<specific_file>.pt
#
# --class-list defaults to all six config.CLASSES so the samples table schema
# stays fixed as the deployed model gains classes. Don't narrow it to the
# 3 currently-trained classes -- that's exactly the schema drift the fixed
# list exists to prevent.
