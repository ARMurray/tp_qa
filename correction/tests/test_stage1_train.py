"""
Smoke test for 03_train_stage1.py on a synthetic seed-site table (multi mode): grouped splits, the
plant-level accept threshold and the saved artifacts. Also unit-tests model_utils.plant_accept_table.

    python correction/tests/test_stage1_train.py
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import config as C
import model_utils as MU


def test_plant_accept_table():
    # plant p1: best site 0.9 right; p2: 0.8 wrong; p3: 0.7 right; p4: 0.2 right (a site, not the best, is wrong)
    g = ["p1", "p1", "p2", "p2", "p3", "p4", "p4"]
    y = [1, 0, 0, 1, 1, 1, 0]
    p = [0.9, 0.1, 0.8, 0.3, 0.7, 0.2, 0.1]
    thr, tab = MU.plant_accept_table(g, y, p, target_precision=0.6, min_accepted=1)
    assert list(tab["n_accepted"]) == [1, 2, 3, 4] and list(tab["n_right"]) == [1, 1, 2, 3]
    assert np.allclose(tab["precision"], [1, 0.5, 2 / 3, 0.75])
    assert thr == 0.2 and tab["capture"].iloc[-1] == 3 / 4      # 3 of the 4 plants have... p2 also has a correct site
    # nothing qualifies
    thr2, _ = MU.plant_accept_table(g, y, p, target_precision=0.99, min_accepted=2)
    assert thr2 is None


def make_table(n_plants=600, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_plants):
        cid = f"{i:08d}"
        cwns_right = rng.random() < 0.7
        for k in range(rng.integers(1, 4)):
            is_cwns = k == 0
            right = cwns_right if is_cwns else (not cwns_right and k == 1 and rng.random() < 0.8)
            rows.append(dict(
                CWNS_ID=cid, site_id=f"{cid}_s{k + 1}", ll_uuid=f"u{i}_{k}", **{"class": "Correct" if right else "Incorrect"},
                n_origins=int(1 + right + (rng.random() < 0.2)), n_seeds=int(1 + rng.integers(0, 3)),
                has_cwns_point=is_cwns, has_icis=bool(rng.random() < 0.6), site_rank=k + 1,
                name_match_score=float(np.clip(rng.normal(0.7 if right else 0.2, 0.15), 0, 1)),
                lbcs_function=rng.choice(["Utility", "Other"], p=[0.8, 0.2] if right else [0.2, 0.8]),
                osm_ww=False, log_gisacre=float(rng.normal(2 if right else 1, 1))))
    return pd.DataFrame(rows)


def test_train():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        C.MODELS_DIR = tmp / "models"
        C.FEATURES_OUTPUT_DIR = tmp / "features"
        C.CWNS_DIR = tmp / "cwns"
        for d in (C.MODELS_DIR, C.FEATURES_OUTPUT_DIR, C.CWNS_DIR):
            d.mkdir()
        t = make_table()
        t.to_parquet(C.FEATURES_OUTPUT_DIR / "14_stage1_training.parquet", index=False)
        rng = np.random.default_rng(1)
        ids = t.CWNS_ID.unique()
        pd.DataFrame({"CWNS_ID": ids, "LATITUDE": rng.uniform(30, 47, len(ids)),
                      "LONGITUDE": rng.uniform(-120, -75, len(ids))}).to_csv(
            C.CWNS_DIR / "PHYSICAL_LOCATION.txt", index=False, encoding="latin1")

        spec = importlib.util.spec_from_file_location("s1", HERE.parent / "scripts" / "03_train_stage1.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        sys.argv = ["03", "--n-iter", "2", "--n-jobs", "2", "--allow-no-holdout"]
        assert C.MULTI
        m.main()
        for f in ("stage1_rf_model.joblib", "stage1_optimal_threshold.joblib", "stage1_accept_threshold.joblib",
                  "stage1_plant_threshold_analysis.parquet"):
            assert (C.MODELS_DIR / f).exists(), f
        thr = joblib.load(C.MODELS_DIR / "stage1_accept_threshold.joblib")
        youden = joblib.load(C.MODELS_DIR / "stage1_optimal_threshold.joblib")
        assert thr >= youden
        model = joblib.load(C.MODELS_DIR / "stage1_rf_model.joblib")
        assert "site_id" not in model["feature_columns"] and "CWNS_ID" not in model["feature_columns"]


if __name__ == "__main__":
    test_plant_accept_table()
    test_train()
    print("stage 1 training tests passed")
