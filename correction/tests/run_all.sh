#!/usr/bin/env bash
# Run every test in this folder (synthetic data; needs duckdb+spatial, geopandas, h3, scikit-learn). From the repo root:
#   bash correction/tests/run_all.sh
set -u
fail=0
for t in test_seed_sites test_seed_features test_stage1_sites test_stage1_train test_inference_multi \
         test_output_provenance test_holdout_multi test_window_seeds test_01e_seed_sites test_multi_status test_naip_mosaic; do
    if python "correction/tests/$t.py" > /tmp/$t.log 2>&1; then echo "ok    $t"; else echo "FAIL  $t  (see /tmp/$t.log)"; fail=1; fi
done
exit $fail
