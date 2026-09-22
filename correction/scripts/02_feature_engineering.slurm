#!/bin/bash
#SBATCH --job-name=corr_02
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=05:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-51
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/02_%A_%a.log

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

STATES=(AK AL AR AZ CA CO CT DC DE FL GA HI IA ID IL IN KS KY LA MA MD ME MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA PR RI SC SD TN TX UT VA VT WA WI WV WY)

STATE="${STATES[$SLURM_ARRAY_TASK_ID]}"

CMD="python $SCRIPTS/02_feature_engineering.py --states $STATE"

[ "${FULL_UNIVERSE:-0}" = "1" ] && CMD="$CMD --full-universe"

echo "Running: $CMD"
eval "$CMD"

echo "Finished: $(date)"