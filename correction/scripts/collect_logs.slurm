#!/bin/bash
#SBATCH --job-name=corr_logs
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/collectlogs_%A.log

# Copies job logs from correction/logs/ (gitignored) into
# correction/diagnostics/logs/ (not gitignored), so a whole log can be
# committed and read on a machine that cannot see the cluster.
#
#   sbatch collect_logs.slurm                                   # newest 3 logs
#   sbatch --export=PATTERN="04_*" collect_logs.slurm           # Stage 2 logs
#   sbatch --export=PATTERN="02_*",LATEST=6 collect_logs.slurm
#   sbatch --export=JOB="48213" collect_logs.slurm              # one job's logs
#   sbatch --export=CLEAR=1,PATTERN="04_*" collect_logs.slurm   # only this batch
#
# This is small enough to run on a login node if that is easier --
# `python collect_logs.py --pattern "04_*"` does the same thing. The wrapper
# exists so it works the same way as every other step here.
#
# Variables:
#   PATTERN="04_*"   glob against correction/logs/ (default *.log)
#   LATEST=3         how many of the newest matches to copy
#   JOB=48213        every log whose name contains this job id; ignores
#                    PATTERN and LATEST
#   MAXKB=512        per-file cap. Over it, the MIDDLE is dropped and the head
#                    and tail are kept, because the inputs are at the top and
#                    the results are at the bottom. Truncation is announced
#                    inside the copy.
#   CLEAR=1          delete earlier copies first, so the directory holds only
#                    this batch. Use it when the old ones are answered and
#                    would just be confusing.
#   KEEPPROGRESS=1   do not collapse \r progress-bar output
#   DRYRUN=1         list what would be copied, copy nothing
#
# AFTERWARDS, from a login node -- the job cannot push for you:
#   cd /work/GRDVULN/tp_qa
#   git add correction/diagnostics/logs/
#   git commit -m "logs for review"
#   git push

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

CMD="python $SCRIPTS/collect_logs.py"
[ -n "${PATTERN:-}" ]      && CMD="$CMD --pattern \"$PATTERN\""
[ -n "${LATEST:-}" ]       && CMD="$CMD --latest $LATEST"
[ -n "${JOB:-}" ]          && CMD="$CMD --job $JOB"
[ -n "${MAXKB:-}" ]        && CMD="$CMD --max-kb $MAXKB"
[ -n "${CLEAR:-}" ]        && CMD="$CMD --clear"
[ -n "${KEEPPROGRESS:-}" ] && CMD="$CMD --keep-progress"
[ -n "${DRYRUN:-}" ]       && CMD="$CMD --dry-run"

echo "Running: $CMD"
eval "$CMD"
RC=$?

echo "Finished: $(date)  (exit $RC)"
exit $RC
