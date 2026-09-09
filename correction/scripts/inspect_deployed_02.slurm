#!/bin/bash
#SBATCH --job-name=corr_inspect_deployed
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/inspect_deployed_%A.log

#   sbatch inspect_deployed_02.slurm

python /work/GRDVULN/correction/scripts/inspect_deployed_02.py

echo "Finished: $(date)"
