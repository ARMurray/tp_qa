"""
Tests for the multi-source Stage 1 table (02_feature_engineering.build_stage1_site_training) on the synthetic
parcel store from test_seed_sites.py.

    python correction/tests/test_stage1_sites.py
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import Point

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import config as C
import seed_sites as SS
import test_seed_sites as T


def load_02():
    spec = importlib.util.spec_from_file_location("feat_eng", HERE.parent / "scripts" / "02_feature_engineering.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def seed_rows(cid, specs):
    return [T.seed(f"{cid}_{src}_1", src, origin, *latlon, cid=cid) for src, origin, latlon in specs]


def main():
    fe = load_02()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        T.write_store(tmp / "parcels")
        C.PARCEL_BASE = tmp / "parcels"
        C.SEED_SITES_DIR = tmp / "seed_sites"
        C.OD_SEED_SITES_DIR = tmp / "od_seed_sites"
        C.apply_population_filter = lambda df, label, min_pop=None: df
        fe.NM.load_facility_names = lambda *a, **k: {"A": "Xville WWTP", "B": "Xville WWTP", "C": "Acme Plant"}
        fe.POP_FLOOR = 0
        con = C.duckdb_connect()

        P1, P2, P3 = T.centre(0, 0), T.centre(1, 0), T.centre(0, 20)
        seeds = pd.DataFrame(
            seed_rows("A", [("cwns_point", "cwns", P1), ("icis_fac_frs", "icis_facility", P2),
                            ("icis_feat_INF", "icis_feature", P3)]) +
            seed_rows("B", [("cwns_point", "cwns", P3), ("icis_fac_frs", "icis_facility", P1)]) +
            seed_rows("C", [("cwns_point", "cwns", P3), ("icis_fac_frs", "icis_facility", P1)]))
        m = SS.match_seeds_to_parcels(con, "ZZ", seeds)
        sp = seeds.merge(m, on="seed_id")
        parcels = SS.fetch_parcels(con, "ZZ", sp["ll_uuid"].dropna().unique())
        sp, sites = SS.build_sites(sp, parcels)
        d = C.SEED_SITES_DIR / "state=ZZ"
        d.mkdir(parents=True)
        sp.to_parquet(d / "seed_parcels.parquet", index=False)
        sites.to_parquet(d / "seed_sites.parquet", index=False)

        # labels: A correct at P1; B wrong (at P3), corrected to P1; C wrong (at P3), no correction known
        gpkg = tmp / "training.gpkg"
        orig = {"A": P1, "B": P3, "C": P3}
        classes = gpd.GeoDataFrame({"CWNS_ID": ["A", "B", "C"], "class": ["Correct", "Incorrect", "Incorrect"]},
                                   geometry=[Point(orig[k][1], orig[k][0]) for k in "ABC"], crs=4326)
        classes.to_file(gpkg, layer="classes", driver="GPKG")
        corr = gpd.GeoDataFrame({"CWNS_ID": ["B"], "Corrected_X": [P1[1]], "Corrected_Y": [P1[0]],
                                 "Original_X": [P3[1]], "Original_Y": [P3[0]]},
                                geometry=[Point(P3[1], P3[0])], crs=4326)
        corr.to_file(gpkg, layer="corrections", driver="GPKG")
        C.TRAINING_GPKG = gpkg

        plant_features = pd.DataFrame({
            "CWNS_ID": ["A", "B", "C"], "STATE_CODE": "ZZ", "LATITUDE": 40.0, "LONGITUDE": -100.0,
            "h3_res9": [h3.latlng_to_cell(*orig[k], 9) for k in "ABC"],
            "subdivision": "xville", "place": "xville", "county": "zz", "is_rural": False, "pop_served": 5000})
        parcel_features = pd.DataFrame({
            "ll_uuid": ["P1", "P2", "P3", "P4"], "state": "ZZ", "geoid": "00001", "n_parcels": 100,
            "owner": ["CITY OF XVILLE", "City of Xville.", "ACME LLC", "SOMEONE ELSE"], "log_gisacre": 1.0})

        # label logic on its own
        lab = fe._plant_label_table(con, plant_features).set_index("CWNS_ID")
        assert lab.at["A", "truth_ll_uuid"] == "P1" and lab.at["B", "truth_ll_uuid"] == "P1"
        assert pd.isna(lab.at["C", "truth_ll_uuid"]) and lab.at["C", "reported_ll_uuid"] == "P3"

        s = fe.build_stage1_site_training(con, plant_features, parcel_features, ["ZZ"], allow_no_od=True)
        by = s.set_index(["CWNS_ID", "has_cwns_point"])["class"]
        assert by[("A", True)] == "Correct" and by[("A", False)] == "Incorrect"      # A: CWNS site is the truth
        assert by[("B", True)] == "Incorrect" and by[("B", False)] == "Correct"      # B: the ICIS site is the truth
        assert len(s[s.CWNS_ID == "C"]) == 1 and by[("C", True)] == "Incorrect"       # C: other site unlabelled, dropped
        assert not {"STATE_CODE", "site_lat", "ll_uuids", "owner"} & set(s.columns)
        assert {"site_name_pool_rank", "n_origins", "has_cwns", "name_match_score", "d_to_cwns_point_m"} <= set(s.columns)
        assert s.site_id.is_unique
        leaked = set(fe.SITE_LABEL_COLS) & set(s.columns)
        assert not leaked, f"label-derived columns reached the model table: {leaked}"

        # a reviewer-marked multi-parcel plant: a site holding ANY marked parcel is the plant
        ls = fe.label_sites(
            pd.DataFrame({"CWNS_ID": ["Z", "Z", "Z"], "site_id": ["Z_s1", "Z_s2", "Z_s3"],
                          "ll_uuids": [["P1"], ["P2"], ["P3"]]}),
            pd.DataFrame({"CWNS_ID": ["Z"], "class": ["Incorrect"], "reported_ll_uuid": ["P3"],
                          "corrected_ll_uuid": ["P1"], "truth_ll_uuid": ["P1"], "truth_set": [{"P1", "P2"}]}))
        assert list(ls["class"]) == ["Correct", "Correct", "Incorrect"]
        assert not [c for c in s.columns if c.endswith(("_x", "_y"))], "a site column collides with a parcel/plant column"

        # missing seed-site OD is an error unless asked for
        try:
            fe.build_stage1_site_training(con, plant_features, parcel_features, ["ZZ"])
            raise AssertionError("expected SystemExit for missing seed-site OD")
        except SystemExit:
            pass

        # a state with no NAIP: no error, the other states' od_* columns, every site od_ran False
        yy = C.OD_SEED_SITES_DIR / "candidates" / "state=YY"
        yy.mkdir(parents=True)
        pd.DataFrame({"CWNS_ID": ["Q"], "ll_uuid": ["Q1"], "od_ran": True, "od_has_detection": True,
                      "od_n_objects": [1], "state": "YY", "od_fetch_error": None}).to_parquet(yy / "part-y.parquet")
        C.NO_NAIP_STATES = ["ZZ"]
        s3 = fe.build_stage1_site_training(con, plant_features, parcel_features, ["ZZ"])
        assert len(s3) == len(s) and not s3["od_ran"].any() and (s3["od_n_objects"] == 0).all()
        assert "od_fetch_error" not in s3.columns
        C.NO_NAIP_STATES = []
        import shutil
        shutil.rmtree(yy)

        # with OD present it joins on (CWNS_ID, ll_uuid)
        od_dir = C.OD_SEED_SITES_DIR / "candidates" / "state=ZZ"
        od_dir.mkdir(parents=True)
        pd.DataFrame({"CWNS_ID": ["A", "A", "B", "B", "C"], "ll_uuid": ["P1", "P3", "P3", "P1", "P3"],
                      "od_ran": True, "od_has_detection": [True, False, False, True, False],
                      "od_n_objects": [3, 0, 0, 2, 0], "od_has_clarifier": [True, False, False, True, False],
                      "state": "ZZ", "parcel_area_m2": 1e4, "parcel_capped": False, "parcel_cap_reason": None,
                      "n_tiles_generated": 1, "od_fetch_error": None}).to_parquet(od_dir / "part-x.parquet", index=False)
        s2 = fe.build_stage1_site_training(con, plant_features, parcel_features, ["ZZ"])
        assert len(s2) == len(s) and s2["od_ran"].all()
        # 01e's bookkeeping columns are not Stage 1 features (03 trains on every column it does not drop)
        assert not set(C.OD_BOOKKEEPING_COLS) & set(s2.columns), set(C.OD_BOOKKEEPING_COLS) & set(s2.columns)
        assert s2.set_index(["CWNS_ID", "has_cwns_point"]).at[("A", True), "od_n_objects"] == 3

        # ---- Stage 2 training in multi mode: window from seeds, seed features per candidate ----
        C.NLCD_OUTPUT_DIR = tmp / "nlcd"
        C.NLCD_OUTPUT_DIR.mkdir()
        store = pd.read_parquet(tmp / "parcels" / "state=ZZ" / "all.parquet")
        store[["ll_uuid", "h3_index_9"]].to_parquet(C.NLCD_OUTPUT_DIR / f"nlcd_ZZ_k{C.K_RINGS}.parquet", index=False)
        C.SEEDS_CSV = tmp / "plant_seeds.csv"
        seeds.rename(columns={"CWNS_ID": "cwns_id"}).assign(copy_of="").to_csv(C.SEEDS_CSV, index=False)
        # B's seeds include one 40 km away: it must widen the window without breaking the table
        s2 = fe.build_stage2_training(con, plant_features, parcel_features, ["ZZ"])
        assert set(s2.CWNS_ID) == {"B"}, set(s2.CWNS_ID)                                   # only the corrected plant
        s2 = s2.set_index("ll_uuid")
        assert s2.at["P1", "label"] == 1 and s2["label"].sum() == 1
        assert s2.at["P1", "on_parcel_icis"] and s2.at["P3", "on_parcel_cwns_point"] and not s2.at["P1", "on_parcel_cwns_point"]
        assert s2.at["P1", "d_seed_icis_fac_frs"] < 400 and s2.at["P3", "d_seed_cwns_point"] < 400
        assert s2.at["P3", "distance_m"] < 400 and s2.at["P1", "distance_m"] > 2000          # distance_m is still from the CWNS point
        assert set(fe.__dict__["SS"].SEED_FEATURE_COLS if "SS" in fe.__dict__ else []) <= set(s2.columns) or True
        import seed_sites as _SS
        assert set(_SS.SEED_FEATURE_COLS) <= set(s2.columns)
        assert "name_pool_rank" in s2.columns
        # a state with no labelled plant gives an empty shard rather than a dead task
        none_labelled = plant_features.assign(CWNS_ID=["X1", "X2", "X3"])
        e = fe.build_stage1_site_training(con, none_labelled, parcel_features, ["ZZ"])
        assert len(e) == 0 and {"CWNS_ID", "site_id", "class"} <= set(e.columns)
        print("stage 1 site table tests passed")


if __name__ == "__main__":
    main()
