#!/bin/bash
#SBATCH --job-name=corr_s1kw
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/s1kw_%A.log

# Read-only: does Stage 1 undervalue a utility-sounding owner on the reported
# parcel? See stage1_keyword_diagnostic.py.
#
#   sbatch stage1_keyword_diagnostic.slurm
#
# Writes correction/diagnostics/stage1_keyword_<host>_<stamp>.txt. Then, from
# a login node:
#   cd /work/GRDVULN/tp_qa
#   git add correction/diagnostics/
#   git commit -m "stage1 keyword diagnostic"
#   git push

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

CMD="python $SCRIPTS/stage1_keyword_diagnostic.py"
echo "Running: $CMD"
eval "$CMD"
RC=$?
echo "Finished: $(date)  (exit $RC)"
exit $RC
