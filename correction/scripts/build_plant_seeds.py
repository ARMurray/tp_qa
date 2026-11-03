"""
Build the seed table for the multi-source correction pipeline (docs/PLAN_20261103_multi_source_seeds.md).

A seed is one plant-role location for one plant, from one source. Read from build_plant_candidates.py's
plant_candidates_long.csv, every plant-role row becomes a seed (a plant with several ICIS permits has
several ICIS seeds). Output:
  correction/data/candidates/plant_seeds.csv

Columns
  seed_id           cwns_id_source_k  (k = rank within the plant's rows of that source)
  source            cwns_point, cwns_addr, icis_fac_frs, icis_fac_addr, icis_feat_INF, icis_feat_INO
  source_group      cwns_coords | cwns_geocode | icis_npdes   (the three labels the outputs report)
  origin            cwns | icis_facility | icis_feature   (support is counted in origins, not points)
  lat, lon, permit_number, detail, quality, is_primary
  copy_of           seed_id of the earliest seed within COPY_M of this one, else blank. ICIS facility
                    points are often the CWNS point to the metre; a copy is kept but is not independent
  independent       True when copy_of is blank
  location_group    1 km chained clusters of this plant's seeds, numbered 1.. best first (most
                    independent origins, then most seeds). The pre-parcel stand-in for "same site"
  group_n_seeds, group_n_origins (independent origins only), group_n_origins_all (copies counted:
  the definition the QA dashboard uses), is_best_group

The parcel/site dedupe in the plan happens later, when seeds meet parcels; location_group is the
point-only version used for QA and for the recall check.

Run:  python -I build_plant_seeds.py        (build_plant_candidates.py first)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

REPO = Path(__file__).resolve().parents[2]
IN = REPO / "correction/data/candidates"

SEED_SRC = ["cwns_point", "cwns_addr", "icis_fac_frs", "icis_fac_addr", "icis_feat_INF", "icis_feat_INO"]
ORIGIN = {"cwns_point": "cwns", "cwns_addr": "cwns", "icis_fac_frs": "icis_facility",
          "icis_fac_addr": "icis_facility", "icis_feat_INF": "icis_feature", "icis_feat_INO": "icis_feature"}
GROUP = {"cwns_point": "cwns_coords", "cwns_addr": "cwns_geocode", "icis_fac_frs": "icis_npdes",
         "icis_fac_addr": "icis_npdes", "icis_feat_INF": "icis_npdes", "icis_feat_INO": "icis_npdes"}
CLUSTER_M = 1000   # same as build_plant_candidates.py
COPY_M = 1.0


def hav_matrix(lat, lon):
    p = np.radians(lat)
    dlat = p[:, None] - p[None, :]
    dlon = np.radians(lon)[:, None] - np.radians(lon)[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(p)[:, None] * np.cos(p)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))


def main():
    L = pd.read_csv(IN / "plant_candidates_long.csv", dtype={"cwns_id": str, "permit_number": str}, low_memory=False)
    S = L[(L.role == "plant") & L.source.isin(SEED_SRC)].copy()
    S["order"] = S.source.map(SEED_SRC.index)
    S["is_primary"] = S.is_primary.fillna(False).astype(bool)
    S = S.sort_values(["cwns_id", "order", "is_primary", "permit_number", "lat", "lon"],
                      ascending=[True, True, False, True, True, True], kind="stable").reset_index(drop=True)
    S["k"] = S.groupby(["cwns_id", "source"]).cumcount() + 1
    S["seed_id"] = S.cwns_id + "_" + S.source + "_" + S.k.astype(str)
    S["origin"] = S.source.map(ORIGIN)
    S["source_group"] = S.source.map(GROUP)

    copy_of = np.full(len(S), "", dtype=object)
    group = np.zeros(len(S), dtype=int)
    g_seeds = np.zeros(len(S), dtype=int)
    g_orig = np.zeros(len(S), dtype=int)
    g_orig_all = np.zeros(len(S), dtype=int)

    ids = S.seed_id.to_numpy()
    origin = S.origin.to_numpy()
    for _, idx in S.groupby("cwns_id", sort=False).indices.items():
        n = len(idx)
        if n == 1:
            group[idx] = 1
            g_seeds[idx] = g_orig[idx] = g_orig_all[idx] = 1
            continue
        D = hav_matrix(S.lat.to_numpy()[idx], S.lon.to_numpy()[idx])
        indep = np.ones(n, dtype=bool)
        for i in range(1, n):
            hit = np.flatnonzero(D[:i, i] <= COPY_M)
            if len(hit):
                copy_of[idx[i]] = ids[idx[hit[0]]]
                indep[i] = False
        nc, lab = connected_components(csr_matrix(D <= CLUSTER_M), directed=False)
        info = [(len(set(origin[idx][indep & (lab == c)])), int((lab == c).sum()),
                 len(set(origin[idx][lab == c])), c) for c in range(nc)]
        rank = {c: r + 1 for r, (_, _, _, c) in enumerate(sorted(info, key=lambda t: (-t[0], -t[1], t[3])))}
        for no, ns, na, c in info:
            m = idx[lab == c]
            group[m] = rank[c]
            g_seeds[m], g_orig[m], g_orig_all[m] = ns, no, na

    S["copy_of"] = copy_of
    S["independent"] = S.copy_of == ""
    S["location_group"], S["group_n_seeds"] = group, g_seeds
    S["group_n_origins"], S["group_n_origins_all"] = g_orig, g_orig_all
    S["is_best_group"] = S.location_group == 1

    cols = ["seed_id", "cwns_id", "source", "source_group", "origin", "lat", "lon", "permit_number", "detail",
            "quality", "is_primary", "copy_of", "independent", "location_group", "group_n_seeds",
            "group_n_origins", "group_n_origins_all", "is_best_group"]
    S[cols].to_csv(IN / "plant_seeds.csv", index=False)

    # ---- report
    print(len(S), "seeds,", S.cwns_id.nunique(), "plants")
    print("seeds per plant:", S.groupby("cwns_id").size().describe().round(2).to_dict())
    print("\nshare of seeds that copy an earlier seed within 1 m, by source")
    print(S.groupby("source").independent.agg(n="size", copies=lambda s: int((~s).sum())).assign(
        pct=lambda d: (100 * d.copies / d.n).round(1)).to_string())
    best = S[S.is_best_group].groupby("cwns_id").agg(all_=("group_n_origins_all", "first"), ind=("group_n_origins", "first"))
    print("\nbest group, origins counting copies vs independent only (plants)")
    print(pd.crosstab(best.all_.clip(upper=3), best.ind.clip(upper=3), rownames=["with copies"], colnames=["independent"]).to_string())
    drop = int(((best.all_ >= 2) & (best.ind < 2)).sum())
    print(f"\nplants whose best group looks multi-origin only because of copies: {drop} of {int((best.all_ >= 2).sum())}")
    print("plants with >1 location group:", int((S.groupby("cwns_id").location_group.max() > 1).sum()))


if __name__ == "__main__":
    main()
