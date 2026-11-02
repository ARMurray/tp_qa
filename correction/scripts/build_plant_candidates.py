"""
Per-plant location candidates and source agreement.

Writes (correction/data/candidates/):
  plant_candidates_long.csv   one row per (plant, source point)
  plant_agreement.csv         one row per plant: which sources exist, pairwise
                              distances, spread, draft agreement class, truth check

Sources, by what they say about the PLANT itself or about its surroundings:
  role "plant"      cwns_point      CWNS coordinate (2022 survey / contractor update)
                    cwns_addr       ArcGIS geocode of the CWNS address
                    icis_fac_frs    ICIS facility coordinate (FRS) of a linked permit
                    icis_fac_addr   ArcGIS geocode of that permit's address
                    icis_feat_INF   ICIS influent structure (inside the plant)
                    icis_feat_INO   ICIS internal outfall (inside the plant; long table only)
  role "downstream" icis_feat_EXO   external outfall (legitimately away from the plant)
  role "dispersed"  iu_facility     industrial users that name the plant's permit as receiving POTW
  role "other"      remaining ICIS feature types (receiving water, land application, wells ...)
CWNS and ICIS are separate systems, so cwns_* and icis_* agreement is independent evidence;
cwns_point and cwns_addr (and the two icis_fac_* rows) share an origin and are not.

Draft agreement class (max pairwise distance among the plant-role points in the wide
table): identical <=250 m, close <=1 km, same_town <=5 km, region <=25 km, conflict >25 km,
single_source when fewer than two points. Thresholds are the THRESH list below.

Run:  python -I build_plant_candidates.py [path/to/sewersheds_v2.duckdb]
"""
import sys
from itertools import combinations
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DB = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("C:/Github/sewersheds/V2/data/db/sewersheds_v2.duckdb")
ICIS = REPO / "correction/data/icis_npdes"
GEO = REPO / "correction/data/geocode"
TRUTH = REPO / "correction/data/training/Updates.gpkg"
OUT = REPO / "correction/data/candidates"

THRESH = [(250, "identical"), (1000, "close"), (5000, "same_town"), (25000, "region")]
PLANT_SRC = ["cwns_point", "cwns_addr", "icis_fac_frs", "icis_fac_addr", "icis_feat_INF"]


def hav(lat1, lon1, lat2, lon2):
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def valid(lat, lon):
    return lat.between(17, 72) & lon.between(-180, -60) | lat.between(5, 25) & lon.between(130, 180)


def rows(df, **const):
    out = df.copy()
    for k, v in const.items():
        out[k] = v
    return out


def read_geocode(gdb, layer, id_pad=None):
    g = gpd.read_file(gdb, layer=layer)
    g["ID"] = g.ID.astype("int64").astype(str).str.zfill(id_pad) if id_pad else g.ID.astype(str)
    g = g[g.Status.isin(["M", "T"])].copy()
    g["lat"], g["lon"] = g.geometry.y, g.geometry.x
    return g[["ID", "lat", "lon", "Status", "Score", "Addr_type", "EXPECTED_PRECISION"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB), read_only=True)
    plants = con.execute("""SELECT t.cwns_id, t.facility_name AS name, t.state_code AS state, t.in_universe,
                                   l.latitude AS lat, l.longitude AS lon, l.coord_missing, l.coord_zero, l.coord_source
                            FROM derived.treatment_plant t JOIN staged.location l USING (cwns_id)
                            WHERE NOT t.planned_new""").df()
    links = con.execute("SELECT DISTINCT cwns_id, permit_number FROM staged.permit WHERE permit_number IS NOT NULL").df()
    con.close()
    plants["cwns_id"] = plants.cwns_id.astype(str)
    links["cwns_id"] = links.cwns_id.astype(str)

    # ---- permits / links
    p = pd.read_csv(ICIS / "icis_npdes_permits.csv", dtype=str, keep_default_na=False)
    p["flat"] = pd.to_numeric(p.GEOCODE_LATITUDE, errors="coerce")
    p["flon"] = pd.to_numeric(p.GEOCODE_LONGITUDE, errors="coerce")
    p["flow"] = pd.to_numeric(p.TOTAL_DESIGN_FLOW_NMBR, errors="coerce").fillna(0)
    p["pot"] = p.pot_any_version == "TRUE"
    p = p.rename(columns={"HORIZONTAL_COLLECT_METHOD_CODE": "fac_method", "HORIZONTAL_REF_DATUM_CODE": "fac_datum"})
    pl = links.merge(p[["EXTERNAL_PERMIT_NMBR", "tier", "pot", "flow", "flat", "flon", "fac_method", "fac_datum",
                        "has_cso_component"]],
                     left_on="permit_number", right_on="EXTERNAL_PERMIT_NMBR")
    pl = pl.sort_values(["cwns_id", "pot", "flow", "permit_number"], ascending=[True, False, False, True])
    pl["is_primary"] = ~pl.duplicated("cwns_id")
    n_links, n_in = links.permit_number.nunique(), pl.permit_number.nunique()
    print(f"CWNS-linked permits: {n_links}, found in ICIS universe: {n_in}")

    L = []  # long-table pieces
    # ---- CWNS point
    cp = plants[~plants.coord_missing.astype(bool) & ~plants.coord_zero.astype(bool) & valid(plants.lat, plants.lon)]
    L.append(rows(cp[["cwns_id", "lat", "lon"]], source="cwns_point", role="plant", permit_number="",
                  detail=cp.coord_source.values, quality="", is_primary=True))
    # ---- CWNS address geocode
    cg = read_geocode(GEO / "cwns_results.gdb", "cwns_geocoded", 11).rename(columns={"ID": "cwns_id"})
    L.append(pd.DataFrame({"cwns_id": cg.cwns_id, "lat": cg.lat, "lon": cg.lon, "source": "cwns_addr", "role": "plant",
                           "permit_number": "", "detail": cg.Addr_type + "/" + cg.EXPECTED_PRECISION + "/" + cg.Status,
                           "quality": cg.Score, "is_primary": True}))
    # ---- ICIS facility coordinate (FRS)
    f = pl[pl.flat.notna() & valid(pl.flat, pl.flon)]
    L.append(pd.DataFrame({"cwns_id": f.cwns_id, "lat": f.flat, "lon": f.flon, "source": "icis_fac_frs", "role": "plant",
                           "permit_number": f.permit_number,
                           "detail": "method=" + f.fac_method + ";datum=" + f.fac_datum,
                           "quality": "", "is_primary": f.is_primary}))
    # ---- ICIS address geocode
    ig = read_geocode(GEO / "icis_npdes_results.gdb", "icis_npdes_Geocoded")
    g = pl.merge(ig, left_on="permit_number", right_on="ID")
    L.append(pd.DataFrame({"cwns_id": g.cwns_id, "lat": g.lat, "lon": g.lon, "source": "icis_fac_addr", "role": "plant",
                           "permit_number": g.permit_number, "detail": g.Addr_type + "/" + g.EXPECTED_PRECISION + "/" + g.Status,
                           "quality": g.Score, "is_primary": g.is_primary}))
    # ---- ICIS permitted features
    ft = pd.read_csv(ICIS / "icis_npdes_features.csv", dtype=str, keep_default_na=False)
    ft["lat"] = pd.to_numeric(ft.LATITUDE_MEASURE, errors="coerce")
    ft["lon"] = pd.to_numeric(ft.LONGITUDE_MEASURE, errors="coerce")
    n_bad = int((ft.lat.notna() & ~valid(ft.lat, ft.lon)).sum())
    ft = ft[ft.lat.notna() & valid(ft.lat, ft.lon)]
    fj = pl.merge(ft, left_on="permit_number", right_on="EXTERNAL_PERMIT_NMBR")
    role = fj.PERM_FEATURE_TYPE_CODE.map({"INF": "plant", "INO": "plant", "EXO": "downstream"}).fillna("other")
    L.append(pd.DataFrame({"cwns_id": fj.cwns_id, "lat": fj.lat, "lon": fj.lon,
                           "source": "icis_feat_" + fj.PERM_FEATURE_TYPE_CODE, "role": role,
                           "permit_number": fj.permit_number,
                           "detail": "method=" + fj.HORIZONTAL_COLLECT_METHOD_CODE + ";cso=" + fj.has_cso_component.str.lower(),
                           "quality": "", "is_primary": fj.is_primary}))
    # ---- pretreatment (industrial users) naming the plant's permit as receiving POTW
    iu = pd.read_csv(ICIS / "icis_npdes_pretreatment.csv", dtype=str, keep_default_na=False)
    iu["lat"] = pd.to_numeric(iu.GEOCODE_LATITUDE, errors="coerce")
    iu["lon"] = pd.to_numeric(iu.GEOCODE_LONGITUDE, errors="coerce")
    iu = iu[iu.lat.notna() & valid(iu.lat, iu.lon)]
    ij = links.merge(iu, left_on="permit_number", right_on="RECEIVING_POTW_ID")
    L.append(pd.DataFrame({"cwns_id": ij.cwns_id, "lat": ij.lat, "lon": ij.lon, "source": "iu_facility", "role": "dispersed",
                           "permit_number": ij.EXTERNAL_PERMIT_NMBR, "detail": "receiving=" + ij.permit_number + ";status=" + ij.PERMIT_STATUS_CODE,
                           "quality": "", "is_primary": False}))
    long = pd.concat(L, ignore_index=True)
    long = long[long.cwns_id.isin(plants.cwns_id)]
    print(f"features dropped for out-of-range coordinates: {n_bad}")

    # ---- wide: plant-role points (primary permit) and pairwise distances
    pts = {}
    for s in PLANT_SRC:
        if s == "icis_feat_INF":
            d = long[(long.source == s) & long.is_primary].groupby("cwns_id")[["lat", "lon"]].mean()  # centroid of INF points
        else:
            d = long[(long.source == s) & long.is_primary].drop_duplicates("cwns_id").set_index("cwns_id")[["lat", "lon"]]
        pts[s] = d
    W = plants[["cwns_id", "name", "state", "in_universe"]].set_index("cwns_id")
    for s, d in pts.items():
        W[s + "_lat"] = d.lat
        W[s + "_lon"] = d.lon
    pairs = list(combinations(PLANT_SRC, 2))
    for a, b in pairs:
        W[f"d_{a}__{b}"] = hav(W[a + "_lat"], W[a + "_lon"], W[b + "_lat"], W[b + "_lon"])
    pcols = [f"d_{a}__{b}" for a, b in pairs]
    W["n_plant_points"] = sum(W[s + "_lat"].notna().astype(int) for s in PLANT_SRC)
    W["max_pair_m"] = W[pcols].max(axis=1)
    W["median_pair_m"] = W[pcols].median(axis=1)
    # medoid: the plant-role point with the smallest total distance to the others (ties -> source order)
    lat = np.column_stack([W[s + "_lat"] for s in PLANT_SRC])
    lon = np.column_stack([W[s + "_lon"] for s in PLANT_SRC])
    D = hav(lat[:, :, None], lon[:, :, None], lat[:, None, :], lon[:, None, :])
    tot = np.where(np.isnan(lat), np.inf, np.nansum(D, axis=2))
    k = np.argmin(tot, axis=1)
    has = W.n_plant_points.values > 0
    W["medoid_source"] = np.where(has, np.array(PLANT_SRC)[k], "")
    W["medoid_lat"] = np.where(has, lat[np.arange(len(W)), k], np.nan)
    W["medoid_lon"] = np.where(has, lon[np.arange(len(W)), k], np.nan)
    cls = np.full(len(W), "conflict", dtype=object)
    for t, name in reversed(THRESH):
        cls = np.where(W.max_pair_m.values <= t, name, cls)
    cls = np.where(W.n_plant_points.values < 2, "single_source", cls)
    cls = np.where(W.n_plant_points.values == 0, "no_location", cls)
    W["agreement_class"] = cls

    # downstream / dispersed context, measured from the medoid
    def ctx(src, prefix):
        d = long[long.source == src].merge(W[["medoid_lat", "medoid_lon"]], left_on="cwns_id", right_index=True)
        d = d[d.medoid_lat.notna()]
        d["dm"] = hav(d.lat, d.lon, d.medoid_lat, d.medoid_lon)
        g = d.groupby("cwns_id").dm
        W[prefix + "_n"] = g.size()
        W[prefix + "_min_m"] = g.min()
        W[prefix + "_median_m"] = g.median()
        W[prefix + "_p90_m"] = g.quantile(0.9)
    ctx("iu_facility", "iu")
    ctx("icis_feat_EXO", "exo")
    for c in ("iu_n", "exo_n"):
        W[c] = W[c].fillna(0).astype(int)
    W["has_cso_permit"] = W.index.isin(pl[pl.has_cso_component == "TRUE"].cwns_id)
    W["n_permits"] = pl.groupby("cwns_id").size().reindex(W.index).fillna(0).astype(int)
    W["primary_permit"] = pl[pl.is_primary].set_index("cwns_id").permit_number.reindex(W.index).fillna("")

    # long: distance of each plant-role point to the medoid
    long = long.merge(W[["medoid_lat", "medoid_lon"]], left_on="cwns_id", right_index=True, how="left")
    long["dist_to_medoid_m"] = hav(long.lat, long.lon, long.medoid_lat, long.medoid_lon).round(1)
    long = long.drop(columns=["medoid_lat", "medoid_lon"])

    # ---- truth check (verified treatment plants only)
    m = gpd.read_file(TRUTH)
    m["cwns_id"] = m.CWNS_ID.astype(str).str.zfill(11)
    v = m[(m.Verified == "Yes") & (m.FACILITY_TYPE == "Treatment Plant")].copy()
    corr, okk = v.Corrected == "Yes", v.Original_Correct == "Yes"
    v["truth_lat"] = np.where(corr, v.Corrected_Y, np.where(okk, v.Original_Y.fillna(v.LATITUDE), np.nan))
    v["truth_lon"] = np.where(corr, v.Corrected_X, np.where(okk, v.Original_X.fillna(v.LONGITUDE), np.nan))
    v["truth_kind"] = np.where(corr, "corrected", np.where(okk, "original_correct", ""))
    v = v.dropna(subset=["truth_lat"]).set_index("cwns_id")[["truth_lat", "truth_lon", "truth_kind"]]
    W = W.join(v)
    W["cwns_point_to_truth_m"] = hav(W.truth_lat, W.truth_lon, W.cwns_point_lat, W.cwns_point_lon)
    W["medoid_to_truth_m"] = hav(W.truth_lat, W.truth_lon, W.medoid_lat, W.medoid_lon)

    num = W.select_dtypes("number").columns
    W[num] = W[num].round(1)
    W.reset_index().to_csv(OUT / "plant_agreement.csv", index=False)
    long[["cwns_id", "source", "role", "lat", "lon", "permit_number", "detail", "quality", "is_primary", "dist_to_medoid_m"]] \
        .round({"lat": 6, "lon": 6}).to_csv(OUT / "plant_candidates_long.csv", index=False)

    print(f"\nplants {len(W)}, long rows {len(long)}")
    print(long.groupby(["role", "source"]).size().to_string())
    print("\nagreement_class:\n" + W.agreement_class.value_counts().to_string())
    t = W[W.truth_lat.notna()]
    print(f"\nverified plants with truth: {len(t)}")
    print(t.groupby("agreement_class").agg(n=("cwns_point_to_truth_m", "size"),
                                            cwns_pt_within_500m=("cwns_point_to_truth_m", lambda s: (s <= 500).mean()),
                                            cwns_pt_over_2km=("cwns_point_to_truth_m", lambda s: (s > 2000).mean()),
                                            medoid_within_500m=("medoid_to_truth_m", lambda s: (s <= 500).mean())).round(3).to_string())


if __name__ == "__main__":
    main()
