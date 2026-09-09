#!/bin/bash
#SBATCH --job-name=corr_08_cover
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/08_cover_%A.log

# Candidate-coverage diagnostic. Answers: the true k-ring radius in km, what
# fraction of known corrections the candidate pool can reach (the pipeline
# ceiling), why each missing one is missing, and whether the ~130 missing
# negative rows are a parcel gap or a join bug.
#
# Reads  training_locations.gpkg, Regrid parquet, nlcd_{STATE}_k{K}.parquet,
#        data/features/10_parcel_features.parquet
# Writes data/diagnostics/correction_coverage.parquet
#        data/diagnostics/missing_parcels_topup.csv   (feeds 01d)
#
#   sbatch 08_diagnose_candidate_coverage.slurm
#   sbatch --export=STATES="OH,PA" 08_diagnose_candidate_coverage.slurm
#
# Read-only apart from the two diagnostic outputs -- safe to run any time.
# Memory is modest: the point-in-parcel lookups are per-state and the NLCD
# membership checks read only the ll_uuid column.

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

CMD="python $SCRIPTS/08_diagnose_candidate_coverage.py"
[ -n "${STATES:-}" ] && CMD="$CMD --states $STATES"

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
