"""
seed_sites.py
=============
Turns a plant's seed locations into SEED SITES: the unit Stage 1 now judges
(docs/PLAN_20261103_multi_source_seeds.md).

A seed is one plant-role location from correction/data/candidates/plant_seeds.csv (CWNS point, CWNS
geocode, ICIS facility geocodes, ICIS influent feature). Each is put on the parcel it falls in, and
seeds are collapsed so a parcel is judged once however many sources point at it:

  1. Seed -> parcel. Point-in-parcel, lowest ll_uuid when a point sits in several (the rule 01b and
     02 already use). A seed on no parcel snaps to the nearest parcel within SEED_SNAP_M; beyond that
     it stays unparcelled and is kept as point-only evidence (no site).
  2. Parcels -> site. Parcels of one plant merge into one site when they are within SEED_SITE_GAP_M of
     each other and share a normalised owner, the parcel-only part of site_geometry's rule. Detections
     are not used, so sites exist before any detector has run. Oversized parcels (whole-town
     digitisation artifacts) only ever merge with themselves.
  3. Each site carries its sources, its support counted in INDEPENDENT ORIGINS (a seed that copies an
     earlier seed within SEED_COPY_M adds nothing), and plant-level context: how many sites the plant
     has, this site's rank, its distance to the CWNS point and to the nearest other site.

Pure pandas/shapely except match_seeds_to_parcels and fetch_parcels, which take a DuckDB connection
(config.duckdb_connect) onto the Regrid store. build_sites is tested without any store.

Outputs are written by 01f_build_seed_sites.py:
    data/seed_sites/seed_parcels.parquet   one row per seed: its parcel, match kind, site
    data/seed_sites/seed_sites.parquet     one row per site
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import shapely
from shapely import from_wkb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

SOURCES = ["cwns_point", "cwns_addr", "icis_fac_frs", "icis_fac_addr", "icis_feat_INF", "icis_feat_INO"]
ORIGINS = ["cwns", "icis_facility", "icis_feature"]
PROJ_CRS = 5070


# ===========================================================================
# Loading
# ===========================================================================
def load_seeds(cwns_ids=None, path: Path | None = None, plants: pd.DataFrame | None = None) -> pd.DataFrame:
    """plant_seeds.csv, optionally restricted to some plants. Column contract: build_plant_seeds.py.

    plants (CWNS_ID, LATITUDE, LONGITUDE): every plant in it is guaranteed a cwns_point seed. plant_seeds.csv is
    built from the V2 database, which lacks about 340 of the master's 17,874 treatment plants (the 2026-11-03
    check: 18 in OH, 11 of them verified); without a seed such a plant would have no site and no Stage 1
    verdict. The missing seed is the plant's own coordinates, marked detail='plant_coordinates'."""
    path = path or C.SEEDS_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. It is tracked in git (correction/data/candidates/): pull the repo, or run "
            f"build_plant_candidates.py then build_plant_seeds.py locally.")
    s = pd.read_csv(path, dtype={"cwns_id": str, "permit_number": str}, low_memory=False)
    s = s.rename(columns={"cwns_id": "CWNS_ID"})
    s["copy_of"] = s["copy_of"].fillna("")
    if plants is not None:
        cwns_ids = plants["CWNS_ID"] if cwns_ids is None else cwns_ids
    if cwns_ids is not None:
        s = s[s["CWNS_ID"].isin(set(cwns_ids))]
    if plants is not None:
        pl = plants[["CWNS_ID", "LATITUDE", "LONGITUDE"]].drop_duplicates("CWNS_ID")
        pl = pl[pl["CWNS_ID"].isin(set(cwns_ids)) & ~pl["CWNS_ID"].isin(set(s.loc[s["source"] == "cwns_point", "CWNS_ID"]))]
        pl = pl.assign(LATITUDE=pd.to_numeric(pl["LATITUDE"], errors="coerce"),
                       LONGITUDE=pd.to_numeric(pl["LONGITUDE"], errors="coerce")).dropna()
        if len(pl):
            add = pd.DataFrame({
                "seed_id": pl["CWNS_ID"] + "_cwns_point_1", "CWNS_ID": pl["CWNS_ID"], "source": "cwns_point",
                "source_group": "cwns_coords", "origin": "cwns", "lat": pl["LATITUDE"], "lon": pl["LONGITUDE"],
                "permit_number": None, "detail": "plant_coordinates", "quality": np.nan, "is_primary": True,
                "copy_of": "", "independent": True, "location_group": 1, "group_n_seeds": 1, "group_n_origins": 1,
                "group_n_origins_all": 1, "is_best_group": True})
            # an existing seed on the same spot is a copy of the CWNS point, as build_plant_seeds.py would have said
            near = s.merge(add[["CWNS_ID", "seed_id", "lat", "lon"]], on="CWNS_ID", suffixes=("", "_c"))
            near = near[hav_m(near["lat"], near["lon"], near["lat_c"], near["lon_c"]) <= C.SEED_COPY_M]
            s.loc[s["seed_id"].isin(near["seed_id"]), ["independent"]] = False
            s.loc[s["seed_id"].isin(near["seed_id"]), "copy_of"] = s.loc[s["seed_id"].isin(near["seed_id"]), "CWNS_ID"] + "_cwns_point_1"
            s = pd.concat([s, add.reindex(columns=s.columns)], ignore_index=True)
    return s.reset_index(drop=True)


def hav_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(np.asarray(lon2) - lon1) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))


def norm_owner(s) -> str:
    """Owner text reduced to lowercase words, so 'CITY OF X.' and 'City of X' compare equal."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


# ===========================================================================
# Seed -> parcel (DuckDB)
# ===========================================================================
def _store(state: str) -> str:
    return f"read_parquet('{C.PARCEL_BASE.as_posix()}/state={state}/*.parquet')"


def match_seeds_to_parcels(con, state: str, seeds: pd.DataFrame, snap_m: float = C.SEED_SNAP_M) -> pd.DataFrame:
    """seeds: seed_id, lat, lon. Returns seed_id, ll_uuid, match_kind ('inside' | 'snapped' | 'none'),
    snap_m. Inside = the seed point intersects the parcel; ties go to the lowest ll_uuid."""
    if not len(seeds):
        return pd.DataFrame(columns=["seed_id", "ll_uuid", "match_kind", "snap_m"])
    pts = pd.DataFrame({
        "seed_id": seeds["seed_id"].to_numpy(),
        "geom_wkb": [g.wkb for g in shapely.points(seeds["lon"].to_numpy(), seeds["lat"].to_numpy())],
    })
    con.register("seed_pts", pts)
    try:
        # Direct ST_Intersects with no H3 prefilter: see 02's point_in_parcel_lookup for why.
        inside = con.execute(f"""
            SELECT s.seed_id, p.ll_uuid
            FROM {_store(state)} p
            JOIN seed_pts s ON ST_Intersects(ST_GeomFromWKB(p.wkb_geometry), ST_GeomFromWKB(s.geom_wkb))
        """).df()
    finally:
        con.unregister("seed_pts")
    inside = (inside.sort_values(["seed_id", "ll_uuid"]).drop_duplicates("seed_id")
              .assign(match_kind="inside", snap_m=0.0))

    left = seeds[~seeds["seed_id"].isin(inside["seed_id"])]
    snapped = _snap(con, state, left, snap_m) if len(left) else pd.DataFrame(columns=inside.columns)
    out = pd.concat([inside, snapped], ignore_index=True)
    none = seeds.loc[~seeds["seed_id"].isin(out["seed_id"]), ["seed_id"]].assign(
        ll_uuid=None, match_kind="none", snap_m=np.nan)
    return pd.concat([out, none], ignore_index=True)


def _snap(con, state: str, seeds: pd.DataFrame, snap_m: float, k: int = 2, chunk: int = 400) -> pd.DataFrame:
    """Nearest parcel within snap_m for seeds that fall on no parcel. Candidate parcels come from the
    H3 res-9 cells within k rings of the seed (~600 m at k=2). A parcel is indexed by ONE cell, so a
    very large parcel whose index cell is further away is missed; acceptable here, since this only
    rescues points that sit just off a parcel (road, water, a gap in the county data)."""
    rows = []
    cells = {r.seed_id: set(h3.grid_disk(h3.latlng_to_cell(r.lat, r.lon, 9), k)) for r in seeds.itertuples()}
    all_cells = sorted(set().union(*cells.values()))
    frames = []
    for i in range(0, len(all_cells), chunk):
        flt = ", ".join(f"'{c}'" for c in all_cells[i:i + chunk])
        frames.append(con.execute(
            f"SELECT ll_uuid, h3_index_9, wkb_geometry FROM {_store(state)} WHERE h3_index_9 IN ({flt})").df())
    if not frames:
        return pd.DataFrame(columns=["seed_id", "ll_uuid", "match_kind", "snap_m"])
    par = pd.concat(frames, ignore_index=True).drop_duplicates("ll_uuid")
    if not len(par):
        return pd.DataFrame(columns=["seed_id", "ll_uuid", "match_kind", "snap_m"])
    pg = gpd.GeoDataFrame(par[["ll_uuid", "h3_index_9"]],
                          geometry=from_wkb([bytes(g) for g in par["wkb_geometry"]]), crs=4326).to_crs(PROJ_CRS)
    by_cell = pg.groupby("h3_index_9").indices
    sg = gpd.GeoSeries(gpd.points_from_xy(seeds["lon"], seeds["lat"]), crs=4326).to_crs(PROJ_CRS).to_numpy()
    for r, pt in zip(seeds.itertuples(), sg):
        idx = np.concatenate([by_cell[c] for c in cells[r.seed_id] if c in by_cell] or [np.array([], dtype=int)])
        if not len(idx):
            continue
        d = shapely.distance(pg.geometry.to_numpy()[idx], pt)
        j = int(np.argmin(d))
        if d[j] <= snap_m:
            rows.append((r.seed_id, pg["ll_uuid"].iloc[idx[j]], "snapped", float(d[j])))
    return pd.DataFrame(rows, columns=["seed_id", "ll_uuid", "match_kind", "snap_m"])


def fetch_parcels(con, state: str, uuids, chunk: int = 5000) -> gpd.GeoDataFrame:
    """Geometry (projected to metres) and normalised owner for some parcels, indexed by ll_uuid."""
    uuids = sorted(set(u for u in uuids if isinstance(u, str)))
    frames = []
    for i in range(0, len(uuids), chunk):
        flt = ", ".join(f"'{u}'" for u in uuids[i:i + chunk])
        frames.append(con.execute(
            f"SELECT ll_uuid, owner, wkb_geometry FROM {_store(state)} WHERE ll_uuid IN ({flt})").df())
    if not frames:
        return gpd.GeoDataFrame({"owner_norm": [], "area_m2": []}, geometry=[], crs=PROJ_CRS)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("ll_uuid")
    g = gpd.GeoDataFrame(df[["ll_uuid"]], geometry=from_wkb([bytes(x) for x in df["wkb_geometry"]]),
                         crs=4326).to_crs(PROJ_CRS)
    g["owner_norm"] = df["owner"].map(norm_owner).to_numpy()
    g["area_m2"] = g.geometry.area
    return g.set_index("ll_uuid")


# ===========================================================================
# Parcels -> sites (pure)
# ===========================================================================
def _components(n: int, edges) -> np.ndarray:
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


def merge_parcels(uuids: list[str], parcels: gpd.GeoDataFrame, gap_m: float = C.SEED_SITE_GAP_M) -> dict[str, int]:
    """Group a plant's parcels into sites. Two merge when within gap_m and with the same non-empty owner;
    an oversized parcel merges with nothing. Returns ll_uuid -> component id (smallest index in group)."""
    uuids = sorted(uuids)
    edges = []
    for i in range(len(uuids)):
        if uuids[i] not in parcels.index:
            continue
        for j in range(i + 1, len(uuids)):
            if uuids[j] not in parcels.index:
                continue
            a, b = parcels.loc[uuids[i]], parcels.loc[uuids[j]]
            if a["area_m2"] > C.MAX_PARCEL_AREA_M2 or b["area_m2"] > C.MAX_PARCEL_AREA_M2:
                continue
            if a["owner_norm"] and a["owner_norm"] == b["owner_norm"] and a.geometry.distance(b.geometry) <= gap_m:
                edges.append((i, j))
    comp = _components(len(uuids), edges)
    return dict(zip(uuids, comp))


def build_sites(sp: pd.DataFrame, parcels: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """sp: one row per seed -- plant_seeds columns plus ll_uuid (None when unparcelled), match_kind, snap_m.
    parcels: fetch_parcels output (metres, indexed by ll_uuid). Returns (sp with site_id, sites)."""
    sp = sp.copy()
    sp["site_id"] = None
    site_rows = []
    for cid, g in sp.groupby("CWNS_ID", sort=False):
        have = g[g["ll_uuid"].map(lambda u: isinstance(u, str))]
        comp = merge_parcels(list(have["ll_uuid"].unique()), parcels) if len(have) else {}
        plant = _plant_sites(cid, g, have, comp, parcels)
        for site_id, members in plant["members"].items():
            sp.loc[members, "site_id"] = site_id
        site_rows += plant["rows"]
    sites = pd.DataFrame(site_rows)
    return sp, sites


def _plant_sites(cid, g, have, comp, parcels):
    members_by_comp: dict[int, list] = {}
    for idx, u in have["ll_uuid"].items():
        members_by_comp.setdefault(comp[u], []).append(idx)

    cwns_pt = g[g["source"] == "cwns_point"]
    cwns_ll = (float(cwns_pt["lat"].iloc[0]), float(cwns_pt["lon"].iloc[0])) if len(cwns_pt) else None
    plant_origins = g[g["independent"]]["origin"].nunique()
    sites = []
    for c, idx in members_by_comp.items():
        m = g.loc[idx]
        ind = m[m["independent"]]
        uuids = sorted(m["ll_uuid"].unique())
        # rep parcel: the one holding the CWNS point, else the one with most independent seeds, else lowest id
        pick = m[m["source"] == "cwns_point"]["ll_uuid"]
        if len(pick):
            rep = pick.iloc[0]
        else:
            cnt = ind["ll_uuid"].value_counts() if len(ind) else m["ll_uuid"].value_counts()
            rep = sorted(cnt[cnt == cnt.max()].index)[0]
        site = {
            "CWNS_ID": cid, "STATE_CODE": m["STATE_CODE"].iloc[0] if "STATE_CODE" in m else None,
            "rep_ll_uuid": rep, "ll_uuids": uuids, "site_n_parcels": len(uuids),
            "seed_ids": list(m["seed_id"]), "sources": "+".join(sorted(set(m["source"]))),
            "n_seeds": len(m), "n_independent_seeds": len(ind),
            "n_origins": ind["origin"].nunique(), "n_origins_all": m["origin"].nunique(),
            "site_lat": float(m["lat"].mean()), "site_lon": float(m["lon"].mean()),
            "oversized": bool(any(parcels.at[u, "area_m2"] > C.MAX_PARCEL_AREA_M2 for u in uuids if u in parcels.index)),
            "snapped": bool((m["match_kind"] == "snapped").any()),
            "_idx": idx,
        }
        for s in SOURCES:
            site[f"has_{s}"] = bool((m["source"] == s).any())
        site["has_cwns"] = site["has_cwns_point"] or site["has_cwns_addr"]
        site["has_icis"] = any(site[f"has_{s}"] for s in SOURCES if s.startswith("icis"))
        sites.append(site)

    # rank: most independent origins, then most independent seeds, then the CWNS point's site, then id
    sites.sort(key=lambda s: (-s["n_origins"], -s["n_independent_seeds"], not s["has_cwns_point"], s["rep_ll_uuid"]))
    members = {}
    for r, s in enumerate(sites, 1):
        s["site_id"] = f"{cid}_s{r}"
        s["site_rank"] = r
        s["is_best_site"] = r == 1
        s["n_sites"] = len(sites)
        s["plant_n_origins"] = int(plant_origins)
        s["plant_n_seeds"] = len(g)
        s["n_unparcelled_seeds"] = int((~g.index.isin(have.index)).sum())
        s["d_to_cwns_point_m"] = (float(hav_m(cwns_ll[0], cwns_ll[1], s["site_lat"], s["site_lon"]))
                                  if cwns_ll else np.nan)
        others = [hav_m(s["site_lat"], s["site_lon"], o["site_lat"], o["site_lon"]) for o in sites if o is not s]
        s["d_nearest_other_site_m"] = float(min(others)) if others else np.nan
        members[s["site_id"]] = s.pop("_idx")
    return {"rows": sites, "members": members}


# ===========================================================================
# Reading 01f's output
# ===========================================================================
def load_sites(states=None, cwns_ids=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(sites, seed_parcels) from data/seed_sites/state=XX/, one state file at a time (no whole-directory
    schema merge, as everywhere else in this pipeline). states=None reads every state present."""
    root = C.SEED_SITES_DIR
    dirs = sorted(root.glob("state=*")) if root.exists() else []
    if states is not None:
        want = {f"state={s}" for s in states}
        dirs = [d for d in dirs if d.name in want]
    if not dirs:
        raise FileNotFoundError(f"no seed sites under {root} for states={states!r}: run 01f_build_seed_sites.py first")
    sites = pd.concat([pd.read_parquet(d / "seed_sites.parquet") for d in dirs], ignore_index=True)
    sp = pd.concat([pd.read_parquet(d / "seed_parcels.parquet") for d in dirs], ignore_index=True)
    if cwns_ids is not None:
        keep = set(cwns_ids)
        sites, sp = sites[sites["CWNS_ID"].isin(keep)], sp[sp["CWNS_ID"].isin(keep)]
    return sites.reset_index(drop=True), sp.reset_index(drop=True)


# ===========================================================================
# Stage 2: the candidate window and per-candidate seed features
# ===========================================================================
SEED_FEATURE_COLS = (
    [f"d_seed_{s}" for s in SOURCES]
    + ["d_seed_nearest", "d_seed_nearest_indep", "n_origins_500m", "n_origins_1km", "n_origins_2km", "n_seeds_1km",
       "on_seed_parcel", "seed_parcel_n_origins", "on_parcel_cwns_point", "on_parcel_cwns_addr", "on_parcel_icis",
       "in_seed_site", "seed_site_n_origins", "seed_site_is_best"])

_BOOL_FEATURES = {"on_seed_parcel", "on_parcel_cwns_point", "on_parcel_cwns_addr", "on_parcel_icis", "in_seed_site",
                  "seed_site_is_best"}
_COUNT_FEATURES = {"n_origins_500m", "n_origins_1km", "n_origins_2km", "n_seeds_1km", "seed_parcel_n_origins",
                   "seed_site_n_origins"}

_disk_cache: dict = {}


def _disk(cell: str, k: int) -> frozenset:
    key = (cell, k)
    if key not in _disk_cache:
        _disk_cache[key] = frozenset(h3.grid_disk(cell, k))
    return _disk_cache[key]


def window_cells(seeds: pd.DataFrame, plant_cell: str, k: int) -> set:
    """H3 res-9 cells of a plant's candidate window: the k-ring disk around the plant's own location and
    around every seed. Seeds are one per source point, so most plants add a few overlapping disks."""
    cells = {plant_cell}
    for lat, lon in zip(seeds["lat"], seeds["lon"]):
        cells.add(h3.latlng_to_cell(float(lat), float(lon), 9))
    out = set()
    for c in cells:
        out |= _disk(c, k)
    return out


def add_seed_features(cand: pd.DataFrame, seeds: pd.DataFrame, sp: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Per-candidate evidence from ONE plant's seeds (docs/PLAN_20261103_multi_source_seeds.md). cand needs
    centroid_lat, centroid_lng, ll_uuid. seeds/sp/sites are that plant's rows of plant_seeds, 01f's seed_parcels
    and seed_sites (any may be empty). Every column in SEED_FEATURE_COLS is added, so training and inference
    frames have one schema.

    Distances are from the candidate's H3 cell centre (the same convention as distance_m) to each source's
    nearest seed. 'Independent' counts a seed once however many sources copy its coordinate, so n_origins_*
    cannot be inflated by ICIS points that are the CWNS point."""
    cand = cand.copy()
    n = len(cand)
    for c in SEED_FEATURE_COLS:
        cand[c] = False if c in _BOOL_FEATURES else (0 if c in _COUNT_FEATURES else np.nan)
    if n == 0:
        return cand

    if len(seeds):
        D = hav_m(cand["centroid_lat"].to_numpy()[:, None], cand["centroid_lng"].to_numpy()[:, None],
                  seeds["lat"].to_numpy()[None, :], seeds["lon"].to_numpy()[None, :])
        src = seeds["source"].to_numpy()
        ind = seeds["independent"].to_numpy().astype(bool)
        org = seeds["origin"].to_numpy()
        for s in SOURCES:
            m = src == s
            cand[f"d_seed_{s}"] = D[:, m].min(axis=1) if m.any() else np.nan
        cand["d_seed_nearest"] = D.min(axis=1)
        cand["d_seed_nearest_indep"] = D[:, ind].min(axis=1) if ind.any() else np.nan
        for r, name in ((500, "n_origins_500m"), (1000, "n_origins_1km"), (2000, "n_origins_2km")):
            cand[name] = sum((D[:, ind & (org == o)] <= r).any(axis=1).astype(int) for o in ORIGINS if (ind & (org == o)).any()) \
                if ind.any() else 0
        cand["n_seeds_1km"] = (D[:, ind] <= 1000).sum(axis=1) if ind.any() else 0

    if len(sp) and len(seeds):
        j = sp[sp["ll_uuid"].map(lambda u: isinstance(u, str))].merge(
            seeds[["seed_id", "source", "origin", "independent"]], on="seed_id", how="inner", suffixes=("", "_s"))
        if len(j):
            g = j.groupby("ll_uuid")
            per = pd.DataFrame({
                "seed_parcel_n_origins": g.apply(lambda d: d.loc[d["independent"], "origin"].nunique(), include_groups=False),
                "on_parcel_cwns_point": g["source"].apply(lambda s: bool((s == "cwns_point").any())),
                "on_parcel_cwns_addr": g["source"].apply(lambda s: bool((s == "cwns_addr").any())),
                "on_parcel_icis": g["source"].apply(lambda s: bool(s.str.startswith("icis").any())),
            })
            idx = cand["ll_uuid"].map(lambda u: u in per.index).to_numpy()
            cand["on_seed_parcel"] = idx
            for c in per.columns:
                cand.loc[idx, c] = cand.loc[idx, "ll_uuid"].map(per[c]).to_numpy()

    if len(sites):
        e = sites[["ll_uuids", "n_origins", "is_best_site"]].explode("ll_uuids").dropna(subset=["ll_uuids"])
        e = e.sort_values(["n_origins", "is_best_site"], ascending=False).drop_duplicates("ll_uuids").set_index("ll_uuids")
        idx = cand["ll_uuid"].isin(e.index).to_numpy()
        cand["in_seed_site"] = idx
        cand.loc[idx, "seed_site_n_origins"] = cand.loc[idx, "ll_uuid"].map(e["n_origins"]).to_numpy()
        cand.loc[idx, "seed_site_is_best"] = cand.loc[idx, "ll_uuid"].map(e["is_best_site"]).astype(bool).to_numpy()

    for c in _BOOL_FEATURES:
        cand[c] = cand[c].fillna(False).astype(bool)
    for c in _COUNT_FEATURES:
        cand[c] = cand[c].fillna(0).astype(int)
    return cand
