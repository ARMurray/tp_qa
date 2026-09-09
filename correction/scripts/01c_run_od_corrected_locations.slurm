#!/bin/bash
#SBATCH --job-name=corr_01c
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/01c_%A.log

# Runs object detection on the CORRECTED (true) location for every Stage 2
# training plant -- the companion half of the contrastive pair 01b already
# built for the REPORTED (wrong) location. Output goes to a separate root
# (data/od_features_corrected/) so it never mixes with 01b's output.
#
#   sbatch 01c_run_od_corrected_locations.slurm
#   sbatch --export=LIMIT=10,NORESUME=1 01c_run_od_corrected_locations.slurm
#
# Small job by design -- ~259-316 corrections nationwide, not a full state
# sweep, so this is a single job, not an array, and should finish quickly
# given the throughput already proven in 01b's runs today.

source /work/GRDVULN/correction/scripts/_common.sh

echo "Checking outbound HTTPS to Planetary Computer..."
curl -s -o /dev/null -w "  STAC API HTTP status: %{http_code}\n" \
    https://planetarycomputer.microsoft.com/api/stac/v1 \
    || echo "  curl failed -- outbound HTTPS may be blocked on this node"

CMD="python $SCRIPTS/01c_run_od_corrected_locations.py --workers ${SLURM_CPUS_PER_TASK:-32}"
[ -n "${LIMIT:-}" ]    && CMD="$CMD --limit $LIMIT"
[ "${NORESUME:-0}" = "1" ] && CMD="$CMD --no-resume"

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
