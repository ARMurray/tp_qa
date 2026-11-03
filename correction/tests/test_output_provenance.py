"""13_build_corrected_output in multi mode on the synthetic store: where each output location came from.
    python correction/tests/test_output_provenance.py"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import config as C
import seed_sites as SS
import test_seed_sites as T


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        T.write_store(tmp / "parcels")
        C.PARCEL_BASE = tmp / "parcels"
        C.SEED_SITES_DIR = tmp / "seed_sites"
        C.SEEDS_CSV = tmp / "plant_seeds.csv"
        C.INFERENCE_DIR = tmp / "inference"
        C.OUTPUT_DIR = tmp / "output"
        C.OD_SEED_SITES_DIR = tmp / "od_seed_sites"
        C.OD_OUTPUT_DIR = tmp / "od"
        C.OD_OUTPUT_DIR_CORRECTED = tmp / "od_corr"
        C.OD_CAND_DIR = tmp / "od_cand"
        C.OD_CAND_QUEUE_DIR = tmp / "od_cq"
        C.MASTER_GPKG = tmp / "Updates.gpkg"
        C.MASTER_LAYER = "CWNS_Locations_20261103"
        ids = list("ABCDE")
        C.population_ok_ids = lambda *a, **k: set(ids)
        C.population_served = lambda *a, **k: {i: 5000 for i in ids}
        con = C.duckdb_connect()

        P1, P2, P3 = T.centre(0, 0), T.centre(1, 0), T.centre(0, 20)
        sd = T.seed
        seeds = pd.DataFrame([
            sd("A_cwns_point_1", "cwns_point", "cwns", *P1, cid="A"),
            sd("A_icis_fac_frs_1", "icis_fac_frs", "icis_facility", *P2, cid="A"),
            sd("B_cwns_point_1", "cwns_point", "cwns", *P3, cid="B"),
            sd("B_icis_fac_frs_1", "icis_fac_frs", "icis_facility", *P1, cid="B"),
            sd("C_cwns_point_1", "cwns_point", "cwns", *P3, cid="C"),
            sd("C_icis_fac_addr_1", "icis_fac_addr", "icis_facility", *P1, cid="C"),
            sd("D_cwns_point_1", "cwns_point", "cwns", *P1, cid="D"),
            sd("E_cwns_point_1", "cwns_point", "cwns", *P3, cid="E"),
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

        def site_id(cid, src):
            return sites[(sites.CWNS_ID == cid) & sites.sources.str.contains(src)].site_id.iloc[0]

        # plant_summary as 05 + 05b write it
        base = dict(STATE_CODE="ZZ", rerank_top_ll_uuid=None, rerank_score=np.nan, rerank_fallback=True,
                    rerank_top_stage2a_rank=np.nan, n_sites=2, multi_pass=False)
        summ = pd.DataFrame([
            dict(base, CWNS_ID="A", reported_ll_uuid="P1", stage1_prob_correct=.9, stage1_route="model",
                 trigger_reason="none", best_site_id=site_id("A", "cwns_point"),
                 best_site_sources="cwns_point+icis_fac_frs", best_site_n_origins=2, best_site_n_seeds=2,
                 accepted_site_id=site_id("A", "cwns_point"), accepted_sources="cwns_point+icis_fac_frs"),
            dict(base, CWNS_ID="B", reported_ll_uuid="P1", stage1_prob_correct=.8, stage1_route="model",
                 trigger_reason="none", best_site_id=site_id("B", "icis_fac_frs"),
                 best_site_sources="icis_fac_frs", best_site_n_origins=1, best_site_n_seeds=1,
                 accepted_site_id=site_id("B", "icis_fac_frs"), accepted_sources="icis_fac_frs"),
            dict(base, CWNS_ID="C", reported_ll_uuid="P3", stage1_prob_correct=.2, stage1_route="model",
                 trigger_reason="low_confidence", rerank_top_ll_uuid="P1", rerank_score=.99,
                 rerank_fallback=False, rerank_top_stage2a_rank=1, best_site_id=site_id("C", "cwns_point"),
                 best_site_sources="cwns_point", best_site_n_origins=1, best_site_n_seeds=1),
            dict(base, CWNS_ID="D", reported_ll_uuid="P1", stage1_prob_correct=.9, stage1_route="model",
                 trigger_reason="none", best_site_id=site_id("D", "cwns_point"), best_site_sources="cwns_point",
                 best_site_n_origins=1, best_site_n_seeds=1, accepted_site_id=site_id("D", "cwns_point"),
                 accepted_sources="cwns_point"),
            dict(base, CWNS_ID="E", reported_ll_uuid="P3", stage1_prob_correct=.2, stage1_route="model",
                 trigger_reason="low_confidence", rerank_top_ll_uuid="P1", rerank_score=.30,
                 rerank_fallback=False, rerank_top_stage2a_rank=1, best_site_id=site_id("E", "cwns_point"),
                 best_site_sources="cwns_point", best_site_n_origins=1, best_site_n_seeds=1),
        ])
        C.INFERENCE_DIR.mkdir()
        summ.to_parquet(C.INFERENCE_DIR / "plant_summary_reranked.parquet", index=False)
        pd.DataFrame({"CWNS_ID": ["C", "C", "E"], "ll_uuid": ["P1", "P3", "P1"], "STATE_CODE": "ZZ",
                      "rerank_rank": [1, 2, 1], "rerank_score": [.99, .1, .30], "od_ran": True}) \
            .to_parquet(C.INFERENCE_DIR / "stage2_candidates_reranked.parquet", index=False)

        orig = {"A": P1, "B": P3, "C": P3, "D": P1, "E": P3}
        master = gpd.GeoDataFrame({
            "CWNS_ID": ids, "FACILITY_TYPE": "Treatment Plant", "STATE_CODE": "ZZ", "FACILITY_NAME": "Plant",
            "LATITUDE": [orig[i][0] for i in ids], "LONGITUDE": [orig[i][1] for i in ids],
            "Original_Correct": [None, None, None, "Yes", None], "Verified": ["No", "No", "No", "Yes", "No"],
            "Corrected_X": np.nan, "Corrected_Y": np.nan, "How_Corrected": None,
            "ReGrid_UUID": [None, None, None, "P1", None]},
            geometry=[Point(orig[i][1], orig[i][0]) for i in ids], crs=4326)
        master.to_file(C.MASTER_GPKG, layer="CWNS_Locations_20261103", driver="GPKG")

        spec = importlib.util.spec_from_file_location("out13", HERE.parent / "scripts" / "13_build_corrected_output.py")
        o13 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(o13)
        o13.INFER_DIR, o13.OUT_DIR, o13.SHARE_DIR = C.INFERENCE_DIR, C.OUTPUT_DIR, tmp / "share"
        o13.OBJECT_ROOTS = {k: tmp / "none" for k in o13.OBJECT_ROOTS}
        sys.argv = ["13", "--cutoff", "0.9", "--allow-low-od"]
        o13.main()

        out = pd.read_parquet(C.OUTPUT_DIR / "cwns_corrected_locations.parquet").set_index("CWNS_ID")
        # A: Stage 1 accepted the CWNS point's site; two sources agree and stay at the CWNS coordinates
        assert out.at["A", "status"] == "kept_model" and out.at["A", "loc_basis"] == "stage1_accept"
        assert out.at["A", "loc_sources"] == "cwns_point+icis_fac_frs" and out.at["A", "loc_n_origins"] == 2
        assert out.at["A", "moved_m"] == 0 and pd.isna(out.at["A", "led_by"])
        # B: Stage 1 accepted a site that does NOT hold the CWNS point -> placed on it, found by ICIS
        assert out.at["B", "status"] == "moved" and out.at["B", "loc_basis"] == "stage1_accept_seed"
        assert out.at["B", "moved_to_ll_uuid"] == "P1" and out.at["B", "moved_m"] > 2000
        assert out.at["B", "led_by"] == "icis_fac_frs" and "icis_fac_frs" in out.at["B", "loc_sources"]
        assert out.at["B", "cwns_point_ll_uuid"] == "P3" and out.at["B", "accepted_sources"] == "icis_fac_frs"
        assert out.at["B", "dist_to_final_cwns_point_m"] > 2000 and out.at["B", "dist_to_final_icis_fac_frs_m"] < 250
        # C: flagged, Stage 2 moved it to P1, the parcel an ICIS geocode sits on
        assert out.at["C", "status"] == "moved" and out.at["C", "loc_basis"] == "stage2_move"
        assert out.at["C", "led_by"] == "icis_fac_addr" and out.at["C", "loc_sources"] == "icis_fac_addr"
        # D: reviewer; E: flagged, score under the cutoff
        assert out.at["D", "loc_basis"] == "reviewer" and out.at["E", "status"] == "flagged_not_moved"
        assert out.at["E", "loc_basis"] == "flagged_not_moved" and pd.isna(out.at["E", "led_by"])
        assert {"alt_site_id", "multi_pass", "accepted_sources"} <= set(out.columns)
        print("output provenance tests passed")


if __name__ == "__main__":
    main()
