#!/bin/bash
#SBATCH --job-name=corr_odfresh_rpt
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=debug
#SBATCH --account=grdvuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/work/GRDVULN/tp_qa/correction/logs/odfresh_rpt_%A.log

# Shell only (no Python), so it can also run on a login node:
#   bash od_freshness_report.slurm
#
# Why: on 2026-10-03, 05b skipped 137 candidate-OD files as "written before
# the deployed best.pt" (coverage 53.7% -> 0.5%). The freshness check takes the
# NEWEST *.pt anywhere under models/object_detection/, so this lists every .pt
# with its mtime, whether best.pt still matches the committed blob, and the
# candidate-OD files' dates per root.
#
# Writes correction/diagnostics/od_freshness_<timestamp>.txt; commit and push it.

source /work/GRDVULN/tp_qa/correction/scripts/_latest_log.sh   # log copy -> diagnostics/latest_logs/

ROOT=/work/GRDVULN/tp_qa/correction
OUT="$ROOT/diagnostics/od_freshness_$(date +%Y%m%d-%H%M%S).txt"
cd "$ROOT/.." || exit 1

{
echo "od_freshness_report  host=$(hostname)  when=$(date -Iseconds)"
echo "git HEAD: $(git rev-parse --short HEAD)"
echo
echo "=== every .pt under models/object_detection (newest first) ==="
find "$ROOT/models/object_detection" -name '*.pt' \
    -printf '%TY-%Tm-%Td %TH:%TM  %10s  %p\n' | sort -r
echo
echo "=== best.pt vs the committed blob ==="
echo "working file : $(git hash-object correction/models/object_detection/best.pt)"
echo "HEAD blob    : $(git rev-parse HEAD:correction/models/object_detection/best.pt)"
echo "(equal = same weights as git; a differing mtime is then only a touch/copy)"
echo
for d in od_features_candidates od_features_candidates_queue od_features_candidates_train; do
    p="$ROOT/data/$d"
    echo "=== $d: parquet files per modification date ==="
    if [ -d "$p" ]; then
        find "$p" -name '*.parquet' -printf '%TY-%Tm-%Td\n' | sort | uniq -c
        echo "  newest:"
        find "$p" -name '*.parquet' -printf '%TY-%Tm-%Td %TH:%TM  %p\n' | sort | tail -3
    else
        echo "  (missing)"
    fi
    echo
done
} > "$OUT" 2>&1

echo "wrote $OUT"
