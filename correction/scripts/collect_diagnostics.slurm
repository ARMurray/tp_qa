#!/bin/bash
#SBATCH --job-name=corr_diag
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/diag_%A.log

# Writes one text file describing the state of the pipeline on this machine,
# for committing and reading somewhere else.
#
#   sbatch collect_diagnostics.slurm
#
# Then, once it finishes:
#
#   cd /work/GRDVULN/tp_qa
#   git add correction/diagnostics/
#   git commit -m "diagnostics"
#   git push
#
# The output lands in correction/diagnostics/, which is deliberately NOT
# gitignored -- unlike logs/ and data/ -- because the whole point is that it
# travels. Everything in it is small and textual.
#
# Read-only: it opens config, counts files, reads the deployed model's class
# names and tails recent logs. It writes nothing but its own report, so it is
# safe to run at any point, including while other jobs are running.
#
# Small and fast, so the debug partition is right. 8G is generous -- the only
# real memory cost is reading one .pt to check its classes.

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

CMD="python \"$SCRIPTS/collect_diagnostics.py\""

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"
