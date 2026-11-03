"""05's multi-source Stage 1 decision and Stage 2a pool, with stub models, on the synthetic parcel store.
    python correction/tests/test_inference_multi.py"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import h3
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import config as C
import seed_sites as SS
import test_seed_sites as T


class Stub:
    """pipeline.predict_proba(X)[:, 1] from a function of X."""
    def __init__(self, fn):
        self.fn = fn

    def predict_proba(self, X):
        p = np.asarray(self.fn(X), float)
        return np.column_stack([1 - p, p])


def main():
    spec = importlib.util.spec_from_file_location("inf05", HERE.parent / "scripts" / "05_run_inference.py")
    inf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inf)
    inf.NM.load_facility_names = lambda *a, **k: {}
    inf.feat_eng.POP_FLOOR = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        T.write_store(tmp / "parcels")
        C.PARCEL_BASE = tmp / "parcels"
        C.SEED_SITES_DIR = tmp / "seed_sites"
        C.OD_SEED_SITES_DIR = tmp / "od_seed_sites"
        C.NLCD_OUTPUT_DIR = tmp / "nlcd"
        C.NLCD_OUTPUT_DIR.mkdir()
        C.SEEDS_CSV = tmp / "plant_seeds.csv"
        con = C.duckdb_connect()

        P1, P2, P3 = T.centre(0, 0), T.centre(1, 0), T.centre(0, 20)
        sd = T.seed
        seeds = pd.DataFrame([
            sd("A_cwns_point_1", "cwns_point", "cwns", *P1, cid="A"),
            sd("A_icis_fac_frs_1", "icis_fac_frs", "icis_facility", *P2, cid="A"),
            sd("A_icis_feat_INF_1", "icis_feat_INF", "icis_feature", *P3, cid="A"),
            sd("B_cwns_point_1", "cwns_point", "cwns", *P3, cid="B"),
            sd("B_icis_fac_frs_1", "icis_fac_frs", "icis_facility", *P1, cid="B"),
            sd("D_cwns_point_1", "cwns_point", "cwns", *P1, cid="D"),
            sd("D_icis_fac_frs_1", "icis_fac_frs", "icis_facility", *P2, cid="D"),
            sd("D_icis_feat_INF_1", "icis_feat_INF", "icis_feature", *P3, cid="D"),
            sd("D_icis_fac_addr_1", "icis_fac_addr", "icis_facility", *P3, cid="D"),
        ])
        seeds.rename(columns={"CWNS_ID": "cwns_id"}).assign(copy_of="").to_csv(C.SEEDS_CSV, index=False)
        m = SS.match_seeds_to_parcels(con, "ZZ", seeds)
        sp = seeds.merge(m, on="seed_id")
        parcels = SS.fetch_parcels(con, "ZZ", sp["ll_uuid"].dropna().unique())
        sp, sites = SS.build_sites(sp, parcels)
        d = C.SEED_SITES_DIR / "state=ZZ"
        d.mkdir(parents=True)
        sp.to_parquet(d / "seed_parcels.parquet", index=False)
        sites.to_parquet(d / "seed_sites.parquet", index=False)

        plants = pd.DataFrame({"CWNS_ID": ["A", "B", "D", "E"], "STATE_CODE": "ZZ",
                               "LATITUDE": [P1[0], P3[0], P1[0], 40.5], "LONGITUDE": [P1[1], P3[1], P1[1], -100.5]})
        plants["h3_res9"] = [h3.latlng_to_cell(a, b, 9) for a, b in zip(plants.LATITUDE, plants.LONGITUDE)]
        plant_features = pd.DataFrame({"CWNS_ID": plants.CWNS_ID, "STATE_CODE": "ZZ", "LATITUDE": plants.LATITUDE,
                                       "LONGITUDE": plants.LONGITUDE, "subdivision": "x", "place": "x", "county": "x",
                                       "is_rural": False})
        parcel_features = pd.DataFrame({"ll_uuid": ["P1", "P2", "P3", "P4"], "state": "ZZ", "geoid": "1",
                                        "owner": ["CITY OF X", "City of X.", "ACME LLC", "OTHER"], "osm_ww": False})

        # Stage 1: p = origins / 3, accept at >= 0.6
        s1_bundle = dict(feature_columns=["n_origins"], pipeline=Stub(lambda X: X["n_origins"] / 3))
        out, scores = inf.run_stage1_sites(plants, plant_features, parcel_features, s1_bundle, 0.6, ["ZZ"])
        o = out.set_index("CWNS_ID")
        assert o.at["A", "trigger_reason"] == "none" and o.at["A", "accepted_sources"] == "cwns_point+icis_fac_frs"
        assert o.at["A", "reported_ll_uuid"] == "P1" and o.at["A", "alt_site_sources"] == "icis_feat_INF"
        assert not o.at["A", "multi_pass"]                                   # the alternative does not pass
        assert o.at["B", "trigger_reason"] == "low_confidence" and o.at["B", "stage1_accepted"] == False
        assert o.at["B", "best_site_sources"] == "cwns_point"                 # tie at 1/3: the CWNS site wins
        assert pd.isna(o.at["B", "accepted_site_id"])
        assert o.at["D", "trigger_reason"] == "none" and o.at["D", "multi_pass"]  # two 2-origin sites 2.2 km apart
        assert o.at["D", "alt_site_dist_m"] > 2000 and o.at["D", "n_sites"] == 2
        assert o.at["E", "trigger_reason"] == "no_parcel" and o.at["E", "stage1_route"] == "no_parcel"
        assert len(scores) == len(sites) and scores.stage1_prob_correct.between(0, 1).all()

        # OSM rule: a tagged site is accepted whatever its score
        pf_osm = parcel_features.assign(osm_ww=[False, False, True, False])
        out2, _ = inf.run_stage1_sites(plants, plant_features, pf_osm, s1_bundle, 0.99, ["ZZ"])
        o2 = out2.set_index("CWNS_ID")
        assert o2.at["B", "stage1_route"] == "osm_confirmed" and o2.at["B", "reported_ll_uuid"] == "P3"
        assert o2.at["B", "trigger_reason"] == "none"

        # Stage 2a for the flagged plant B: every parcel in the seed window stays, seed parcels included
        store = pd.read_parquet(tmp / "parcels" / "state=ZZ" / "all.parquet")
        store[["ll_uuid", "h3_index_9"]].to_parquet(C.NLCD_OUTPUT_DIR / f"nlcd_ZZ_k{C.K_RINGS}.parquet", index=False)
        flagged = out[out.CWNS_ID == "B"]
        s2_bundle = dict(feature_columns=["d_seed_nearest", "on_seed_parcel"],
                         pipeline=Stub(lambda X: 1 / (1 + X["d_seed_nearest"] / 1000)))
        cands = inf.run_stage2a(flagged, plant_features, parcel_features, s2_bundle, 20)
        c = cands.set_index("ll_uuid")
        assert {"P1", "P3"} <= set(c.index)                                    # B's own CWNS parcel P3 is NOT excluded
        assert c.at["P1", "on_parcel_icis"] and c.at["P3", "on_parcel_cwns_point"]
        assert set(SS.SEED_FEATURE_COLS) <= set(cands.columns) and cands.stage2_prob_correct.between(0, 1).all()
        print("multi-source inference tests passed")


if __name__ == "__main__":
    main()
