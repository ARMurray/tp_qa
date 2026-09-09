#!/bin/bash -l
# submit_all_manifest.sh
# ----------------------
# Submits one job per state present in the training manifest.
#
# Usage:
#   ./submit_all_manifest.sh              # all states, skips any already done
#   ./submit_all_manifest.sh --dry-run    # print what would be submitted
#   ./submit_all_manifest.sh --force      # resubmit everything, wipes partitions
#
# Runs under bash even from a tcsh login shell (the #! line handles it), so
# invoke it as ./submit_all_manifest.sh rather than `source`-ing it.
#
# Sizing: ~2,591 samples nationally at ~13.6 tiles each is ~35k tiles. Most
# states are 10-60 samples and finish in a few minutes; the big ones (FL 186,
# TX 150, KS 153) are still well under an hour.

REPO=/work/GRDVULN/infrastructure
MANIFEST=$REPO/data/samples/training_manifest.parquet
OUT=$REPO/data/inference_train
PY=$REPO/envs/naip_pipeline_env/bin/python

DRY_RUN=0
FORCE=0
for a in "$@"; do
    case "$a" in
        --dry-run) DRY_RUN=1 ;;
        --force)   FORCE=1 ;;
        *) echo "Unknown option: $a"; exit 1 ;;
    esac
done

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found: $MANIFEST"
    echo "Run: sbatch submit_build_manifest.sh"
    exit 1
fi

cd "$REPO" || exit 1
mkdir -p logs

# Every state in the manifest -- including states whose samples all lack a
# parcel, since 09 now runs those window-only. Filtering on parcel_found here
# would silently drop exactly the Tier2_no_parcel population we care about.
STATES=$("$PY" -c "
import pandas as pd
m = pd.read_parquet('$MANIFEST')
print(' '.join(sorted(m['state_fips'].astype(str).str.zfill(2).unique())))
")

if [ -z "$STATES" ]; then
    echo "ERROR: no states read from manifest"
    exit 1
fi

echo "States in manifest: $(echo $STATES | wc -w)"
echo

N_SUB=0
N_SKIP=0

for ST in $STATES; do
    SAMPLE_DIR="$OUT/samples/state=$ST"

    # 09's writer is append-only: a re-run stacks new part files under old
    # ones rather than replacing them, and a schema change then yields a
    # partition that can't be concatenated. So a redo means DELETING all four
    # tables for that state, not just passing --no-resume.
    if [ -d "$SAMPLE_DIR" ] && [ "$(ls -A "$SAMPLE_DIR" 2>/dev/null)" ]; then
        if [ "$FORCE" -eq 1 ]; then
            echo "  [$ST] wiping existing partitions (--force)"
            if [ "$DRY_RUN" -eq 0 ]; then
                for T in tiles detections objects samples; do
                    rm -rf "$OUT/$T/state=$ST"
                done
            fi
        else
            echo "  [$ST] already has output -- skipping (use --force to redo)"
            N_SKIP=$((N_SKIP + 1))
            continue
        fi
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [$ST] would submit"
    else
        sbatch --job-name="train_$ST" "$REPO/submit_manifest_pipeline.sh" "$ST" >/dev/null \
            && echo "  [$ST] submitted" \
            || echo "  [$ST] SUBMIT FAILED"
    fi
    N_SUB=$((N_SUB + 1))
done

echo
echo "Submitted: $N_SUB   Skipped: $N_SKIP"
[ "$DRY_RUN" -eq 1 ] && { echo "(dry run -- nothing actually submitted)"; exit 0; }

cat <<'EOF'

Monitor:
  squeue -u $USER

When the queue drains, CHECK COVERAGE BEFORE READING ANY RESULTS:
  python inspect_samples.py

That verifies every manifest row produced a sample and hard-fails on mixed
schemas. Small states are the risk -- CT has 1 tileable sample, NV 2, DE and
PR 4 -- and a silent failure there costs samples from a class with only 254
members nationally.

Find failures:
  grep -il "error\|traceback" logs/train_*_*.err

Resubmit just the states that came back short (inspect_samples.py names them):
  sbatch --job-name=train_XX submit_manifest_pipeline.sh XX
EOF
