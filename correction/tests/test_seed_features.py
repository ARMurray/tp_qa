"""window_cells and add_seed_features.   python correction/tests/test_seed_features.py"""
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import config as C
import seed_sites as SS

LAT, LON = 40.0, -100.0


def off(dlat_m=0.0, dlon_m=0.0):
    return LAT + dlat_m / 111_000, LON + dlon_m / (111_000 * 0.766)


seeds = pd.DataFrame([
    dict(seed_id="A1", source="cwns_point", origin="cwns", independent=True, lat=LAT, lon=LON),
    dict(seed_id="A2", source="icis_fac_frs", origin="icis_facility", independent=False, lat=LAT, lon=LON),       # copy
    dict(seed_id="A3", source="icis_feat_INF", origin="icis_feature", independent=True, lat=off(300)[0], lon=LON),  # 300 m N
    dict(seed_id="A4", source="icis_fac_addr", origin="icis_facility", independent=True, lat=off(5000)[0], lon=LON),  # 5 km N
])
sp = pd.DataFrame({"seed_id": ["A1", "A2", "A3", "A4"], "ll_uuid": ["P1", "P1", "P2", None]})
sites = pd.DataFrame({"ll_uuids": [["P1"], ["P2", "P2b"]], "n_origins": [1, 1], "is_best_site": [True, False]})

# candidates: at the CWNS point, 300 m north, 1.5 km east, far away
rows = [("P1", *off(0, 0)), ("P2", *off(300, 0)), ("P5", *off(0, 1500)), ("P9", *off(0, 30000)), ("P2b", *off(400, 0))]
cand = pd.DataFrame(rows, columns=["ll_uuid", "centroid_lat", "centroid_lng"])
out = SS.add_seed_features(cand, seeds, sp, sites).set_index("ll_uuid")

assert set(SS.SEED_FEATURE_COLS) <= set(out.columns)
assert out.at["P1", "d_seed_cwns_point"] < 1 and out.at["P1", "d_seed_icis_fac_frs"] < 1       # copy still measured by distance
assert abs(out.at["P2", "d_seed_icis_feat_INF"]) < 1 and 290 < out.at["P2", "d_seed_cwns_point"] < 310
assert np.isnan(out.at["P1", "d_seed_cwns_addr"])                                                # no such source
assert out.at["P1", "n_origins_500m"] == 2 and out.at["P5", "n_origins_2km"] == 2             # copy adds no origin: cwns + icis_feature
assert out.at["P1", "n_origins_1km"] == 2 and out.at["P5", "n_origins_1km"] == 0 and out.at["P9", "n_origins_2km"] == 0
assert out.at["P1", "n_seeds_1km"] == 2                                                         # independent seeds only (A1, A3)
assert out.at["P1", "on_seed_parcel"] and out.at["P1", "on_parcel_cwns_point"] and out.at["P1", "on_parcel_icis"]
assert out.at["P1", "seed_parcel_n_origins"] == 1                                               # A2 is a copy
assert out.at["P2", "on_parcel_icis"] and not out.at["P2", "on_parcel_cwns_point"]
assert not out.at["P5", "on_seed_parcel"] and out.at["P5", "seed_parcel_n_origins"] == 0
assert out.at["P2b", "in_seed_site"] and not out.at["P2b", "seed_site_is_best"] and out.at["P1", "seed_site_is_best"]
assert not out.at["P9", "in_seed_site"]
assert out.dtypes["on_seed_parcel"] == bool and out.dtypes["n_origins_1km"].kind == "i"

# no seeds at all still returns the full schema
empty = SS.add_seed_features(cand, seeds.iloc[0:0], sp.iloc[0:0], sites.iloc[0:0])
assert set(SS.SEED_FEATURE_COLS) <= set(empty.columns) and (empty["n_origins_1km"] == 0).all()
assert empty["d_seed_nearest"].isna().all()

# window: union of disks around the plant cell and each seed
pc = h3.latlng_to_cell(LAT, LON, 9)
w = SS.window_cells(seeds, pc, 3)
assert pc in w and len(w) > len(h3.grid_disk(pc, 3))                                            # the 5 km seed widens it
assert h3.latlng_to_cell(*off(5000), 9) in w
print("seed feature tests passed")
