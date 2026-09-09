#!/bin/bash
#SBATCH --job-name=corr_08b_rings
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/08b_rings_%A.log

# Prices the K_RINGS decision from 08's output. Reads one parquet, writes two
# CSVs -- no compute, so it runs fine on a login node too:
#
#   python 08b_analyze_ring_misses.py
#
# Or via sbatch:
#   sbatch 08b_analyze_ring_misses.slurm
#   sbatch --export=KS="20,25,30,40",RECKM="75" 08b_analyze_ring_misses.slurm
#
# Run 08_diagnose_candidate_coverage.slurm first -- this reads the
# correction_coverage.parquet it writes.

source /work/GRDVULN/correction/scripts/_common.sh

CMD="python $SCRIPTS/08b_analyze_ring_misses.py"
[ -n "${KS:-}" ]     && CMD="$CMD --candidate-ks $KS"
[ -n "${RECKM:-}" ]  && CMD="$CMD --records-error-km $RECKM"
[ -n "${CURK:-}" ]   && CMD="$CMD --current-k $CURK"

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
