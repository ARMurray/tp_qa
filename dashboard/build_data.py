"""
Pack the candidate tables into two script files the QA dashboard loads with <script> tags
(so index.html works when opened straight from disk):
  dashboard/data/plants.js   window.PLANTS = {cols: [...], rows: [[...], ...]}   one row per plant
  dashboard/data/cands.js    window.CANDS  = {cwns_id: [[src, lat, lon, permit, detail, dist_m], ...]}

Run build_plant_candidates.py first.   Run:  python -I build_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
IN = REPO / "correction/data/candidates"
OUT = REPO / "dashboard/data"

SRC = ["cwns_point", "cwns_addr", "icis_fac_frs", "icis_fac_addr", "icis_feat_INF", "icis_feat_INO",
       "icis_feat_EXO", "iu_facility", "icis_feat_other"]
CATS = ["all_agree", "cwns_ok_outliers", "cwns_isolated", "split", "no_consensus", "thin", "cwns_missing", "no_location"]

COLS = [  # (output name, plant_agreement column, kind)
    ("cat", None, "cat"),
    ("n_origins", "n_origins", "int"), ("n_pts", "n_plant_points", "int"),
    ("cwns_other", "cwns_to_other_m", "m"), ("cwns_to_cons", "cwns_to_consensus_m", "m"),("farthest", "farthest_from_consensus_m", "m"),
    ("max_pair", "max_pair_m", "m"), ("median_pair", "median_pair_m", "m"),
    ("cwns_fac", "d_cwns_point__icis_fac_frs", "m"), ("cwns_addr", "d_cwns_point__cwns_addr", "m"),
    ("cwns_facaddr", "d_cwns_point__icis_fac_addr", "m"), ("cwns_inf", "d_cwns_point__icis_feat_INF", "m"),
    ("iu_n", "iu_n", "int"), ("iu_median", "iu_median_m", "m"), ("exo_n", "exo_n", "int"), ("exo_min", "exo_min_m", "m"),
    ("cwns_truth", "cwns_point_to_truth_m", "m"), ("n_permits", "n_permits", "int"),
]


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))


def num(v, nd=0):
    return None if pd.isna(v) else (round(float(v), nd) if nd else int(round(float(v))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    W = pd.read_csv(IN / "plant_agreement.csv", dtype={"cwns_id": str, "primary_permit": str}, low_memory=False)
    L = pd.read_csv(IN / "plant_candidates_long.csv", dtype={"cwns_id": str, "permit_number": str}, keep_default_na=False,
                    na_values=[""])
    W = W[W.consensus_category != "no_location"].reset_index(drop=True)

    # where the plant dot is drawn: the CWNS point (the thing under QA), else the consensus
    W["lat"] = W.cwns_point_lat.fillna(W.consensus_lat)
    W["lon"] = W.cwns_point_lon.fillna(W.consensus_lon)
    W = W[W.lat.notna()].reset_index(drop=True)

    cols = ["id", "name", "state", "lat", "lon", "cons_lat", "cons_lon", "truth_lat", "truth_lon", "truth_kind",
            "permit"] + [c[0] for c in COLS]
    rows = []
    for r in W.itertuples(index=False):
        d = r._asdict()
        row = [d["cwns_id"], d["name"] if isinstance(d["name"], str) else "", d["state"] if isinstance(d["state"], str) else "",
               num(d["lat"], 5), num(d["lon"], 5), num(d["consensus_lat"], 5), num(d["consensus_lon"], 5),
               num(d["truth_lat"], 5), num(d["truth_lon"], 5),
               d["truth_kind"] if isinstance(d["truth_kind"], str) else "",
               d["primary_permit"] if isinstance(d["primary_permit"], str) else ""]
        for name, col, kind in COLS:
            if kind == "cat":
                row.append(CATS.index(d["consensus_category"]))
            else:
                row.append(num(d[col]))
        rows.append(row)
    (OUT / "plants.js").write_text("window.PLANTS=" + json.dumps({"cols": cols, "cats": CATS, "rows": rows}, separators=(",", ":")) + ";\n",
                                   encoding="utf-8")

    # candidates: every point for every plant, with its distance to the consensus point
    L = L[L.cwns_id.isin(W.cwns_id)].merge(W[["cwns_id", "consensus_lat", "consensus_lon"]], on="cwns_id")
    L["src"] = L.source.where(L.source.isin(SRC), "icis_feat_other").map(SRC.index)
    L["dm"] = hav(L.lat, L.lon, L.consensus_lat, L.consensus_lon)
    L["detail"] = L.detail.fillna("")
    other = ~L.source.isin(SRC)
    L.loc[other, "detail"] = L.loc[other, "source"].str.replace("icis_feat_", "") + " " + L.loc[other, "detail"]
    cands = {}
    for cid, g in L.groupby("cwns_id", sort=False):
        cands[cid] = [[int(s), round(la, 5), round(lo, 5), p if isinstance(p, str) else "", dt, num(dm)]
                      for s, la, lo, p, dt, dm in zip(g.src, g.lat, g.lon, g.permit_number, g.detail, g.dm)]
    (OUT / "cands.js").write_text("window.SRC=" + json.dumps(SRC) + ";window.CANDS=" + json.dumps(cands, separators=(",", ":")) + ";\n",
                                  encoding="utf-8")
    for f in ("plants.js", "cands.js"):
        print(f, round((OUT / f).stat().st_size / 1e6, 1), "MB")
    print(len(rows), "plants,", len(L), "candidate points")


if __name__ == "__main__":
    main()
