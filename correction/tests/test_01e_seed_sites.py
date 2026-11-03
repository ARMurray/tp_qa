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
    assert list(c.columns) == ["CWNS_ID", "ll_uuid", "STATE_CODE", "stage2_prob_correct"] and len(c) == 3

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
