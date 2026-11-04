"""01e --seed-sites --dry-run reads 01f's output and plans one parcel per site (heavy imports stubbed: no torch here).
    python correction/tests/test_01e_seed_sites.py"""
import contextlib
import importlib.util
import io
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
for name in ("torch", "ultralytics", "planetary_computer", "pystac_client", "rasterio", "rasterio.enums",
             "rasterio.windows"):
    sys.modules.setdefault(name, mock.MagicMock())
import config as C

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    C.SEED_SITES_DIR = tmp / "seed_sites"
    C.OD_SEED_SITES_DIR = tmp / "od_seed_sites"
    d = C.SEED_SITES_DIR / "state=ZZ"
    d.mkdir(parents=True)
    pd.DataFrame({"CWNS_ID": ["A", "A", "B"], "STATE_CODE": "ZZ", "rep_ll_uuid": ["P1", "P3", "P1"],
                  "site_id": ["A_s1", "A_s2", "B_s1"]}).to_parquet(d / "seed_sites.parquet", index=False)
    pd.DataFrame({"CWNS_ID": ["A"], "seed_id": ["x"]}).to_parquet(d / "seed_parcels.parquet", index=False)

    spec = importlib.util.spec_from_file_location("e01", HERE.parent / "scripts" / "01e_run_od_candidates.py")
    e01 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e01)
    c = e01.load_seed_site_parcels(None)
    assert list(c.columns) == ["CWNS_ID", "ll_uuid", "STATE_CODE", "stage2_prob_correct",
                               "fallback_lat", "fallback_lon"] and len(c) == 3
    assert c["fallback_lat"].isna().all()  # no seed coordinates in this fixture: the parcel's own point is used

    # fallback seed: the CWNS point when it is on the rep parcel, else an independent seed, never the mean
    sites = pd.DataFrame({"site_id": ["A_s1", "B_s1", "C_s1"], "rep_ll_uuid": ["P1", "P2", "P9"],
                          "site_lat": [9.0, 9.0, 7.5], "site_lon": [9.0, 9.0, 7.5]})
    sp = pd.DataFrame({"site_id": ["A_s1", "A_s1", "B_s1", "B_s1", "B_s1"],
                       "ll_uuid": ["P1", "P1", "P2", "P2", "P7"],
                       "source": ["icis_fac_frs", "cwns_point", "icis_fac_addr", "icis_fac_frs", "cwns_point"],
                       "independent": [True, True, False, True, True],
                       "lat": [1.0, 2.0, 3.0, 4.0, 5.0], "lon": [1.0, 2.0, 3.0, 4.0, 5.0]})
    lat, lon = e01.seed_fallback_points(sites, sp)
    assert list(lat) == [2.0, 4.0, 7.5], lat  # A: CWNS point; B: independent seed on P2; C: no seed -> site mean

    # cap: a small parcel is tiled as is; an oversized one becomes one tile around the seed
    from shapely.geometry import box
    small = box(0, 0, 300, 300)
    g, info = e01.capped_footprint(small, float("nan"), float("nan"))
    assert g is small and not info["parcel_capped"]
    huge = box(-1_000_000, 1_000_000, -950_000, 1_050_000)  # 2,500 km2 in EPSG:5070
    g, info = e01.capped_footprint(huge, float("nan"), float("nan"))
    assert info["parcel_capped"] and info["parcel_cap_reason"] == "area"
    assert abs(g.area - C.TILE_SIZE_M ** 2) < 1 and huge.contains(g)
    g, info = e01.capped_footprint(huge, -105.0, 40.0)
    assert info["parcel_capped"] and abs(g.area - C.TILE_SIZE_M ** 2) < 1 and not huge.contains(g)
    # thin diagonal parcel under the area cap but over the tile cap
    from shapely.geometry import LineString
    thin = LineString([(0, 0), (60_000, 60_000)]).buffer(5)
    assert thin.area < C.MAX_PARCEL_AREA_M2
    g, info = e01.capped_footprint(thin, float("nan"), float("nan"))
    assert info["parcel_cap_reason"] == "tile_count", info

    # resume: od_ran False rows are retried, od_ran True rows and files without od_ran count as done
    part = C.OD_SEED_SITES_DIR / "candidates" / "state=ZZ"
    part.mkdir(parents=True)
    pd.DataFrame({"CWNS_ID": ["A", "A"], "ll_uuid": ["P1", "P3"], "od_ran": [True, False]}) \
        .to_parquet(part / "part-1.parquet", index=False)
    pd.DataFrame({"CWNS_ID": ["B"], "ll_uuid": ["P1"]}).to_parquet(part / "part-2.parquet", index=False)
    C.OD_MODEL_DIR = tmp / "no_models"
    assert e01.processed_keys(C.OD_SEED_SITES_DIR, "ZZ") == {("A", "P1"), ("B", "P1")}

    sys.argv = ["01e", "--seed-sites", "--dry-run"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        e01.main()
    out = buf.getvalue()
    assert "od_features_seed_sites" in out.replace("\\", "/") or str(C.OD_SEED_SITES_DIR) in out
    assert "ZZ: 3 candidate parcel(s), 2 plant(s)" in out and "--dry-run" in out, out
    sys.argv = ["01e", "--seed-sites", "--all"]
    try:
        e01.main()
        raise AssertionError("expected exclusive-scope exit")
    except SystemExit as e:
        assert e.code == 2
print("01e seed-site scope tests passed")
