"""multi_status.py: every section runs on a small synthetic run, and on an empty one, without crashing.
    python correction/tests/test_multi_status.py"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import config as C

spec = importlib.util.spec_from_file_location("ms", HERE.parent / "scripts" / "multi_status.py")
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)


def point(tmp):
    C.ROOT = tmp
    C.DATA_DIR = tmp / "data"
    for name, rel in [("NLCD_OUTPUT_DIR", "data/nlcd_features_multi"), ("SEED_SITES_DIR", "data/seed_sites"),
                      ("OD_SEED_SITES_DIR", "data/od_seed_sites"), ("FEATURES_OUTPUT_DIR", "data/features_multi"),
                      ("MODELS_DIR", "models_multi"), ("INFERENCE_DIR", "data/inference_multi"),
                      ("OUTPUT_DIR", "data/output_multi"), ("OD_MODEL_DIR", "models/object_detection")]:
        setattr(C, name, tmp / rel)
    C.SEEDS_CSV = tmp / "plant_seeds.csv"
    C.TRAINING_GPKG = tmp / "t.gpkg"
    ms.OUT = tmp / "diagnostics" / "multi_status.txt"
    ms.LOG_DIR = tmp / "diagnostics" / "latest_logs"


def main():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        point(tmp)
        # 1. nothing exists yet: every section reports it and the file is still written
        ms.main()
        txt = ms.OUT.read_text()
        assert "section failed" not in txt, txt
        assert "ENVIRONMENT" in txt and "JOB LOGS" in txt

        # 2. a small run
        (tmp / "data/nlcd_features_multi").mkdir(parents=True)
        (tmp / "data/nlcd_features").mkdir(parents=True)
        pd.DataFrame({"ll_uuid": list("abcd")}).to_parquet(tmp / "data/nlcd_features_multi/nlcd_OH_k18.parquet")
        pd.DataFrame({"ll_uuid": list("ab")}).to_parquet(tmp / "data/nlcd_features/nlcd_OH_k18.parquet")
        d = tmp / "data/seed_sites/state=OH"
        d.mkdir(parents=True)
        pd.DataFrame({"CWNS_ID": ["A", "A", "B"], "match_kind": ["inside", "none", "snapped"], "snap_m": [0, None, 40.0],
                      "site_id": ["A_s1", None, "B_s1"], "source": ["cwns_point", "icis_fac_frs", "cwns_point"]}) \
            .to_parquet(d / "seed_parcels.parquet")
        pd.DataFrame({"CWNS_ID": ["A", "B"], "site_id": ["A_s1", "B_s1"], "rep_ll_uuid": ["u1", "u2"], "n_origins": [1, 1],
                      "n_origins_all": [2, 1], "n_seeds": [2, 1], "site_n_parcels": [1, 2], "oversized": [False, False],
                      "snapped": [False, True], "has_cwns_point": [True, False], "n_sites": [2, 2],
                      "d_nearest_other_site_m": [80.0, 80.0], "d_to_cwns_point_m": [0.0, 80.0]}).to_parquet(d / "seed_sites.parquet")
        od = tmp / "data/od_seed_sites/candidates/state=OH"
        od.mkdir(parents=True)
        pd.DataFrame({"CWNS_ID": ["A"], "ll_uuid": ["u1"], "od_ran": [True], "od_has_detection": [False]}).to_parquet(od / "part-1.parquet")
        f = tmp / "data/features_multi"
        f.mkdir(parents=True)
        pd.DataFrame({"CWNS_ID": ["A", "A", "B"], "site_id": ["A_s1", "A_s2", "B_s1"], "class": ["Correct", "Incorrect", "Incorrect"],
                      "has_cwns_point": [True, False, True], "n_origins": [2, 1, 1], "od_ran": [True, False, True]}) \
            .to_parquet(f / "14_stage1_training.parquet")
        pd.DataFrame({"CWNS_ID": ["A", "A"], "label": [1, 0], "on_seed_parcel": [True, False], "on_parcel_cwns_point": [False, False],
                      "on_parcel_icis": [True, False]}).to_parquet(f / "15_stage2_training.parquet")
        m = tmp / "models_multi"
        m.mkdir()
        joblib.dump(0.5, m / "stage1_optimal_threshold.joblib")
        joblib.dump(0.7, m / "stage1_accept_threshold.joblib")
        pd.DataFrame({"threshold": [0.9, 0.6], "n_accepted": [1, 2], "n_right": [1, 2], "precision": [1.0, 1.0], "capture": [.5, 1.0]}) \
            .to_parquet(m / "stage1_plant_threshold_analysis.parquet")
        i = tmp / "data/inference_multi"
        i.mkdir(parents=True)
        pd.DataFrame({"CWNS_ID": ["A", "B"], "STATE_CODE": "OH", "trigger_reason": ["none", "low_confidence"],
                      "stage1_route": "model", "stage1_accepted": [True, False], "multi_pass": [False, False], "n_sites": [1, 2],
                      "accepted_sources": ["icis_fac_frs", None], "n_candidates": [0, 12], "stage1_prob_correct": [.9, .2]}) \
            .to_parquet(i / "plant_summary.parquet")
        o = tmp / "data/output_multi"
        o.mkdir(parents=True)
        pd.DataFrame({"CWNS_ID": ["A", "B"], "status": ["moved", "flagged_not_moved"], "loc_basis": ["stage1_accept_seed", "flagged_not_moved"],
                      "led_by": ["icis_fac_frs", None], "loc_sources": ["icis_fac_frs", None], "loc_n_origins": [1, 0],
                      "moved_m": [900.0, 0.0], "multi_pass": [False, False], "built": "2026-11-03"}).to_parquet(o / "cwns_corrected_locations.parquet")
        (tmp / "diagnostics/latest_logs").mkdir(parents=True)
        (tmp / "diagnostics/latest_logs/01f_3.log").write_text("# latest copy\n# job 1 on n1, exit status 1, copied now\n#\nTraceback boom\n")
        ms.main()
        txt = ms.OUT.read_text()
        assert "section failed" not in txt, txt
        for needle in ("ratio median 2.00", "seed -> parcel match", "coverage of seed sites: 1 / 2", "Correct sites per plant",
                       "stage1_accept_threshold.joblib", "= 0.700", "accepted by sources: icis_fac_frs", "moved plants led by: icis_fac_frs",
                       "last 60 lines of 01f_3.log", "Traceback boom", "precision 1.000", "50-100 m: 100%", "WITHOUT the CWNS point (1)"):
            assert needle in txt, needle
        print("multi status tests passed")


if __name__ == "__main__":
    main()
