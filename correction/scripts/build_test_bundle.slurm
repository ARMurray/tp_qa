#!/bin/bash
#SBATCH --job-name=corr_build_bundle
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/build_bundle_%A.log

# Assembles /work/GRDVULN/tp_qa/correction/testing/tpqa_test/ -- a self-contained,
# downloadable copy of the pipeline plus a subset of every input it reads.
#
#   sbatch build_test_bundle.slurm
#   sbatch --export=STATE="OH",NPLANTS="30",NCOUNTIES="2" build_test_bundle.slurm
#   sbatch --export=CLEAN="1" build_test_bundle.slurm
#   sbatch --export=NORASTER="1" build_test_bundle.slurm
#
# Variables (all optional):
#   STATE       default OH
#   NPLANTS     target plant count (default 30)
#   NCOUNTIES   counties to draw plants from (default 2). More counties means
#               more labelled plants but a bigger parcel payload -- the Regrid
#               county files are the main size driver.
#   NORASTER=1  skip the NLCD raster clip. Saves size, but 01a and 01d then
#               can't be run locally.
#   CLEAN=1     delete any existing bundle first. Use this on a re-run --
#               otherwise stale files from a previous bundle survive alongside
#               the new ones, which is exactly the class of confusion this
#               whole exercise is meant to eliminate.
#
# Memory: 64G because 10_parcel_features.parquet is read whole before
# filtering (4.9M rows for the OH/MS/DE pilot). If the nationwide version is
# ever in place, this may need raising.

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

CMD="python $SCRIPTS/build_test_bundle.py"
[ -n "${STATE:-}" ]      && CMD="$CMD --state $STATE"
[ -n "${NPLANTS:-}" ]    && CMD="$CMD --n-plants $NPLANTS"
[ -n "${NCOUNTIES:-}" ]  && CMD="$CMD --n-counties $NCOUNTIES"
[ "${NORASTER:-0}" = "1" ] && CMD="$CMD --no-raster"
[ "${CLEAN:-0}" = "1" ]    && CMD="$CMD --clean"

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
