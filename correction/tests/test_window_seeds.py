"""01a adds every plant seed as a window center in multi mode.   python correction/tests/test_window_seeds.py"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import config as C

C.SEEDS_CSV = HERE.parents[1] / "correction" / "data" / "candidates" / "plant_seeds.csv"   # the repo copy
spec = importlib.util.spec_from_file_location("a01", HERE.parent / "scripts" / "01a_extract_parcels.py")
a01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a01)

seeds = pd.read_csv(C.SEEDS_CSV, dtype={"cwns_id": str}, nrows=5000)
ids = seeds.cwns_id.drop_duplicates().head(20)
plants = pd.DataFrame({"CWNS_ID": ids, "STATE_CODE": "XX", "LATITUDE": 0.0, "LONGITUDE": 0.0})
pts = a01.load_plant_seed_points(plants)
want = seeds[seeds.cwns_id.isin(ids)]
assert len(pts) == len(want) and set(pts.columns) == {"CWNS_ID", "STATE_CODE", "LATITUDE", "LONGITUDE"}
assert (pts.STATE_CODE == "XX").all() and pts.LATITUDE.between(-20, 75).all()
# a plant's every seed becomes a center: the CWNS point plus the others
assert pts.groupby("CWNS_ID").size().max() > 1
print("window seed tests passed")
