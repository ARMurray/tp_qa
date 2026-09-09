#!/bin/bash
#SBATCH --job-name=corr_check
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/correction/logs/check_%A.log

# Verifies the nationwide 01a/01b array run completed cleanly before
# trusting it enough to run 02.
#
#   sbatch --export=JOBID_01A=7399988,JOBID_01B=7400040 check_01a_01b_complete.slurm
#
# Get the job IDs from your earlier sbatch output, or:  sacct -u $USER --format=JobID,JobName%20 | grep corr_01

bash /work/GRDVULN/correction/scripts/check_01a_01b_complete.sh

echo "Finished: $(date)"
