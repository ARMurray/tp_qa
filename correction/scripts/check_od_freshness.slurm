#!/bin/bash
#SBATCH --job-name=corr_odfresh
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/odfresh_%A.log

# Checks every data/od_features/{plants,tiles,detections,objects}/state=XX/
# partition against the mtime of the deployed best.pt. Anything older was
# written by a DIFFERENT model, and 02 (which has no resume) would blend it
# into the national feature table with nothing in the logs saying so.
#
#   sbatch check_od_freshness.slurm                      # report only
#   sbatch --export=ACTION="delete" check_od_freshness.slurm
#   sbatch --export=ACTION="quarantine" check_od_freshness.slurm
#   sbatch --export=ALLOW="AK HI PR" check_od_freshness.slurm
#   sbatch --export=ACTION="delete",ALLOW="AK HI PR" check_od_freshness.slurm
#
# Variables:
#   ACTION=delete|quarantine  what to do with stale partitions. Omit to
#                             report only; the job then FAILS (exit 1) when
#                             anything is stale, which is what makes it
#                             usable as a --dependency=afterok gate.
#   ALLOW="AK HI PR"          states permitted to be stale. AK/HI/PR sit
#                             outside NAIP and CONUS Albers (EPSG:5070) and
#                             cannot be reprocessed by this pipeline, so
#                             listing them records a deliberate coverage
#                             gap rather than an oversight.
#   MODELTIME=YYYY-MM-DD      override the model timestamp. Needed only if
#                             best.pt was copied with scp -p or rsync -t,
#                             which PRESERVE the source mtime and make a
#                             freshly uploaded model look old.
#
# Chain it ahead of 02 so a stale partition stops the pipeline:
#   sbatch --dependency=afterok:<thisjobid> 02_feature_engineering.slurm

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

CMD="python $SCRIPTS/check_od_freshness.py"
[ -n "${ALLOW:-}" ]     && CMD="$CMD --allow $ALLOW"
[ -n "${MODELTIME:-}" ] && CMD="$CMD --model-time $MODELTIME"

case "${ACTION:-report}" in
    delete)     CMD="$CMD --delete" ;;
    quarantine) CMD="$CMD --quarantine" ;;
    report)     ;;
    *) echo "ERROR: ACTION must be delete, quarantine, or unset (report)."
       exit 2 ;;
esac

echo "Running: $CMD"
eval "$CMD"
RC=$?

echo "Finished: $(date)  (exit $RC)"
exit $RC
