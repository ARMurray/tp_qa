#!/bin/bash
#SBATCH --job-name=corr_namematch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=compute
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/namematch_%A.log

# Read-only benchmark: owner name vs facility name, method by method, on the
# training corrections' top-20 pools. See name_match_benchmark.py.
#
#   sbatch --export=INSTALL=1 name_match_benchmark.slurm   # first run
#   sbatch name_match_benchmark.slurm                      # afterwards
#   sbatch --export=METHODS="fuzzy_norm,distinct_idf" name_match_benchmark.slurm
#
# INSTALL=1 pip-installs rapidfuzz and sentence-transformers into the shared
# venv from inside the job, so nothing runs on a login node. torch is already
# there (ultralytics brought it), so this adds little. The embedding models
# (~100 MB each) download once into $ROOT/.hf_cache on first use.
#
# Needs correction/data/cwns/FACILITIES.txt -- tracked in git, so a pull
# brings it.
#
# Writes correction/diagnostics/name_match_benchmark_<host>_<stamp>.txt and
# name_match_pairs_<stamp>.csv. Then, from a login node:
#   cd /work/GRDVULN/tp_qa
#   git add correction/diagnostics/
#   git commit -m "name match benchmark"
#   git push

source /work/GRDVULN/tp_qa/correction/scripts/_common.sh

export HF_HOME="$ROOT/.hf_cache"
mkdir -p "$HF_HOME"

if [ "${INSTALL:-0}" = "1" ]; then
    echo "Installing rapidfuzz + sentence-transformers into $VENV_DIR ..."
    pip install --quiet rapidfuzz sentence-transformers || {
        echo "ERROR: pip install failed"; exit 1; }
fi

CMD="python $SCRIPTS/name_match_benchmark.py"
[ -n "${METHODS:-}" ] && CMD="$CMD --methods $METHODS"

echo "Running: $CMD"
eval "$CMD"
RC=$?

echo "Finished: $(date)  (exit $RC)"
exit $RC
