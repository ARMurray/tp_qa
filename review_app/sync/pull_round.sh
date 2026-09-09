#!/bin/bash
# pull_round.sh
# ==============
# Pulls a review round's queue + holdout manifest down from HPC into
# data/incoming/, ready for `python -m backend.queue_loader --round N`.
#
# ADJUST THE HOST/PATH BELOW -- I don't know your actual HPC access method
# (SSH alias, VPN-mapped path, etc.), so this is a template, not a working
# script as-is.
#
# Usage:
#   ./pull_round.sh 1

set -euo pipefail

ROUND="${1:?Usage: ./pull_round.sh <round_number>}"

HPC_HOST="atmos3"                                        # <-- adjust
HPC_BASE="/work/GRDVULN/correction/data"                 # <-- adjust if paths ever change
LOCAL_INCOMING="$(dirname "$0")/../data/incoming"

echo "Pulling round $ROUND queue + holdout manifest from $HPC_HOST..."

scp "${HPC_HOST}:${HPC_BASE}/review_queue/review_queue_round${ROUND}.parquet" \
    "$LOCAL_INCOMING/"

scp "${HPC_HOST}:${HPC_BASE}/holdout/holdout_manifest.parquet" \
    "$LOCAL_INCOMING/"
scp "${HPC_HOST}:${HPC_BASE}/holdout/holdout_truth.parquet" \
    "$LOCAL_INCOMING/" || echo "  (holdout_truth.parquet not found -- fine if round 1's unlabeled bin hasn't been reviewed anywhere yet)"

echo "Done. Now run: python -m backend.queue_loader --round $ROUND"
