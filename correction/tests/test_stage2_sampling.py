"""04's negative sampling keeps every negative on a seed parcel / seed site (they are what Stage 2 must beat at
inference) and fills the rest of the 1:neg_ratio budget from hard and easy negatives as before.
    python correction/tests/test_stage2_sampling.py"""
import importlib.util
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
for name in ("imblearn", "imblearn.ensemble"):      # not installed here; sampling does not need it
    sys.modules.setdefault(name, mock.MagicMock())
spec = importlib.util.spec_from_file_location("s04", HERE.parent / "scripts" / "04_train_stage2.py")
s04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s04)

n = 5000
rng = np.random.RandomState(0)
s2 = pd.DataFrame({
    "ll_uuid": [f"p{i}" for i in range(n)], "label": 0,
    "lbcs_activity": "Residential", "lbcs_ownership": "Private",
    "osm_ww": False, "has_water": rng.rand(n) < 0.3, "owner_is_utility": False, "owner_is_govt": False,
    "has_ww_keyword": False, "on_seed_parcel": False, "in_seed_site": False})
s2.loc[:9, "label"] = 1                                   # 10 plants' truths, 7 of them on a seed parcel
s2.loc[:6, "on_seed_parcel"] = True
s2.loc[10:24, "on_seed_parcel"] = True                    # 15 seed-parcel negatives (e.g. declined CWNS-point parcels)
s2.loc[25:29, "in_seed_site"] = True                      # 5 more: merged parcels of a seed site
out = s04.sample_negatives(s2, neg_ratio=50)
neg = out[out["label"] == 0]
assert (out["label"] == 1).sum() == 10
assert set(s2.loc[10:29, "ll_uuid"]) <= set(neg["ll_uuid"]), "a seed-parcel negative was dropped"
assert len(neg) == 500, len(neg)                           # the 1:50 budget is unchanged, seed negatives included
assert out["ll_uuid"].is_unique

# single-seed tables (no seed columns) sample exactly as before
old = s04.sample_negatives(s2.drop(columns=["on_seed_parcel", "in_seed_site"]), neg_ratio=50)
assert (old["label"] == 0).sum() == 500
print("stage 2 sampling tests passed")
