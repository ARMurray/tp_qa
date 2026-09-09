#!/bin/bash -l
#SBATCH --mem=32G
#SBATCH --output=/work/GRDVULN/infrastructure/logs/manifest_build_%A.out
#SBATCH --error=/work/GRDVULN/infrastructure/logs/manifest_build_%A.err
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
# Usage: sbatch submit_build_manifest.sh [/path/to/training_locations.gpkg]
#
# Runs ONCE, nationally. Resolves every labeled training point to its county
# and its parcel, writing data/samples/training_manifest.parquet.
#
# Reads one parcel parquet per county across ~1,200 counties, so this is
# I/O-bound and slow relative to how little it computes -- 30-60 min is
# normal. --time is generous on purpose; check `seff <jobid>` afterwards.
#
# NOTE the input: training_locations.gpkg layer 'classes', written by
# Update_Training_Plants.R. NOT plants.gpkg -- plants.gpkg holds corrected
# geometry, which would invert the label for every Incorrect sample.

CLASSES_GPKG=${1:-/work/GRDVULN/Location_Repair/02_feature_engineering/training_locations.gpkg}

if [ ! -f "$CLASSES_GPKG" ]; then
    echo "ERROR: classes gpkg not found: $CLASSES_GPKG"
    echo "Pass the correct path as the first argument."
    exit 1
fi

module load python/3.13
source /work/GRDVULN/infrastructure/envs/naip_pipeline_env/bin/activate

python /work/GRDVULN/infrastructure/pipeline/08_build_training_manifest_hpc.py \
    --classes-gpkg "$CLASSES_GPKG" \
    --out /work/GRDVULN/infrastructure/data/samples/training_manifest.parquet
