#!/bin/bash
# ==============================================================================
# setup_env.sh -- one-time environment build for /work/GRDVULN/correction
# ==============================================================================
# Run this ONCE on a login node before submitting any pipeline jobs:
#     bash /work/GRDVULN/tp_qa/correction/scripts/setup_env.sh
#
# Every SLURM wrapper sources the venv this creates rather than building its
# own -- building a venv inside each job wastes minutes of allocation and (as
# hit on 2026-08-19) can silently resolve against a stale system Python.
#
# The default system python3 on this cluster's compute nodes was found to be
# 3.6, which is far too old: it pulls 2020-era package wheels and chokes on
# modern syntax. The module-load loop below finds a real Python first.
# ==============================================================================
set -euo pipefail

ROOT=/work/GRDVULN/tp_qa/correction
VENV_DIR="$ROOT/.venv"

echo "=== correction pipeline environment setup ==="
echo "Root:  $ROOT"
echo "Venv:  $VENV_DIR"

echo "--- available python modules ---"
module avail python 2>&1 || true

LOADED=""
for modname in python/3.11 python/3.10 python/3.9 python3.11 python3.10 python3.9; do
    if module load "$modname" 2>/dev/null; then
        echo "Loaded module: $modname"
        LOADED="$modname"
        break
    fi
done
if [ -z "$LOADED" ]; then
    echo "WARNING: no python module loaded -- falling back to system python3."
    echo "         If the version check below fails, look at the module list above"
    echo "         and add the right name to the loop in this script."
fi

python3 --version
python3 -c "import sys; assert sys.version_info >= (3,9), f'Python too old: {sys.version}'"

mkdir -p "$ROOT"/{scripts,logs}
mkdir -p "$ROOT"/models/object_detection
mkdir -p "$ROOT"/data/{cwns,training,nlcd_features,od_features,features,reference}

if [ -d "$VENV_DIR" ]; then
    echo "Venv already exists. Delete it and re-run to rebuild:  rm -rf $VENV_DIR"
else
    echo "Creating venv..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# Grouped by which script needs them, so it's obvious what to trim if a
# dependency ever becomes a problem:
#   all         : pandas pyarrow numpy
#   01a         : rasterio, exactextract, h3, duckdb
#   01b         : ultralytics torch pillow pystac-client planetary-computer pyproj
#   02          : geopandas shapely duckdb h3 pyogrio
#   03/04       : scikit-learn, imbalanced-learn (BalancedRandomForestClassifier -- Stage 2
#                 needs the balanced-bootstrap mechanism specifically, per model_utils.py/
#                 04_train_stage2.py docstrings), joblib
#   bins builder: geopandas pyogrio (OpenFileGDB read)
pip install \
    numpy pandas pyarrow \
    duckdb h3 exactextract \
    geopandas shapely pyogrio pyproj rasterio \
    scikit-learn imbalanced-learn joblib \
    pystac-client planetary-computer \
    ultralytics torch pillow

echo "--- verifying imports ---"
python -c "
import sys
import numpy, pandas, pyarrow, duckdb, h3, exactextract, geopandas, shapely, pyproj, rasterio
import sklearn, imblearn, joblib, pystac_client, planetary_computer
print('core deps OK, python', sys.version.split()[0])
"
python -c "import ultralytics, torch; print('torch/ultralytics OK, cuda:', torch.cuda.is_available())"

echo "--- duckdb spatial extension ---"
python -c "
import duckdb
con = duckdb.connect()
con.execute('INSTALL spatial; LOAD spatial;')
print('duckdb spatial OK')
con.close()
"

echo
echo "=== Setup complete ==="
echo "Venv: $VENV_DIR"
echo
echo "Remaining manual steps before running the pipeline:"
echo "  1. (keyword lists already populated in config.py)"
echo "  2. Upload CWNS text exports        -> $ROOT/data/cwns/"
echo "  3. Upload Updates.gdb              -> $ROOT/data/training/"
echo "  4. Upload census gdb + OSM gpkg    -> $ROOT/data/reference/"
echo "  5. Upload trained best.pt          -> $ROOT/models/object_detection/"
echo "  6. Run build_training_bins.py to produce data/training/training_locations.gpkg"
