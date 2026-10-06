"""
13_build_corrected_output.py
============================
The product: one row for EVERY CWNS treatment plant, saying where it is and
how we know -- plus the parcel outline(s) of each plant's site and the
detected objects there.

STATUS (first match wins)
    verified_correct      a reviewer confirmed the reported location (master)
    verified_corrected    a reviewer supplied a corrected location (master)
    reviewed_unresolved   the master says the reported location is wrong but
                          holds no corrected one -- kept, flagged
    kept_site             Stage 1 flagged it and the re-ranker's #1 cleared
                          --cutoff, but the reported parcel turned out to be
                          PART OF THE SAME SITE (split parcels) -- not moved
    moved                 Stage 1 flagged it AND the re-ranker's #1 parcel
                          scored >= --cutoff; the plant moves to that site
    flagged_not_moved     Stage 1 flagged it but no candidate cleared the
                          cutoff -- reported location kept, flagged as doubtful
    kept_osm              reported parcel carries an OSM wastewater tag
                          (passed by rule, stage1_route = osm_confirmed)
    kept_model            Stage 1 scored the reported location as correct
    not_assessed          outside the model's scope; reason in status_reason

Human verification always wins: the master (Updates.gpkg, newest
CWNS_Locations layer) is read first and a verified plant is never moved.

SITES AND THE MOVED POINT (decided 2026-10-05, site_geometry.py)
    A moved plant's site starts at the re-ranker's #1 parcel and takes in
    neighbouring candidates (and the reported parcel) within 50 m that hold a
    detected object (conf >= 0.4) or share #1's owner; up to 4 parcels. The
    point is the mean centre of the detected objects (conf >= 0.4) inside
    the site; with none, the #1 parcel's centroid if inside it, else its
    pole of inaccessibility. coord_method records which.

THE MOVE RULE IS A PARAMETER, NOT A CONSTANT
    --cutoff is required (owner's decision 2026-10-04: 0.95).
    --require-detection additionally refuses moves where nothing fired in
    the pool (rerank_fallback: the #1 is Stage 2a's).

REFUSES to run when the re-ranked candidates mostly lack current-detector
results (national 01e not run). --allow-low-od overrides, for testing only.

Outputs (data/output/):
    cwns_corrected_locations.parquet / .csv     one row per plant
    cwns_corrected_locations.gpkg               layers plants, sites, detections
and copies of all three tables in diagnostics/output/ for the viewer.

Usage:
    python 13_build_corrected_output.py --cutoff 0.95
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import name_match as NM
import site_geometry as SG

INFER_DIR = C.DATA_DIR / "inference"
OUT_DIR = C.DATA_DIR / "output"
SHARE_DIR = C.ROOT / "diagnostics" / "output"
MIN_OD_COVERAGE = 0.5

# Per-object detection output, by source. 01b: tiles around the REPORTED
# location; 01c: around the CORRECTED location of verified plants; 01e: around
# each candidate parcel (national/holdout root and the review-queue root).
OBJECT_ROOTS = {
    "reported": C.OD_OUTPUT_DIR / "objects",
    "corrected": C.OD_OUTPUT_DIR_CORRECTED / "objects",
    "candidate": C.DATA_DIR / "od_features_candidates" / "objects",
    "candidate_queue": C.DATA_DIR / "od_features_candidates_queue" / "objects",
}


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6_371_008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_master() -> pd.DataFrame:
    import geopandas as gpd
    layer = C.MASTER_LAYER or C.latest_master_layer()
    g = gpd.read_file(C.MASTER_GPKG, layer=layer)
    print(f"  master: {C.MASTER_GPKG.name} layer {layer}: {len(g):,} facilities")
    g = g[g["FACILITY_TYPE"] == "Treatment Plant"].copy()
    g["CWNS_ID"] = g["CWNS_ID"].astype(str)
    print(f"  treatment plants: {len(g):,}")
    return pd.DataFrame(g.drop(columns="geometry"))


def load_scored() -> tuple[pd.DataFrame, pd.DataFrame]:
    summ_p = INFER_DIR / "plant_summary_reranked.parquet"
    cand_p = INFER_DIR / "stage2_candidates_reranked.parquet"
    for p in (summ_p, cand_p):
        if not p.exists():
            print(f"ERROR: {p} not found. Run 05 (array + merge) and 05b first.")
            sys.exit(2)
    summ = pd.read_parquet(summ_p)
    summ["CWNS_ID"] = summ["CWNS_ID"].astype(str)
    cand = pd.read_parquet(cand_p, columns=["CWNS_ID", "ll_uuid", "STATE_CODE",
                                            "rerank_rank", "rerank_score", "od_ran"])
    cand["CWNS_ID"] = cand["CWNS_ID"].astype(str)
    cand["ll_uuid"] = cand["ll_uuid"].astype(str)
    return summ, cand


def deployed_model_mtime() -> float | None:
    pts = sorted(C.OD_MODEL_DIR.rglob("*.pt"), key=lambda p: p.stat().st_mtime,
                 reverse=True) if C.OD_MODEL_DIR.exists() else []
    return pts[0].stat().st_mtime if pts else None


def load_objects(plant_ids: set) -> pd.DataFrame:
    """Every detected object for these plants, from every source, written by
    the deployed detector (older files are skipped, as in 05b). Columns:
    CWNS_ID, lon, lat, class_name, max_confidence, source."""
    model_t = deployed_model_mtime()
    want = ["CWNS_ID", "lon", "lat", "class_name", "max_confidence"]
    frames = []
    for source, root in OBJECT_ROOTS.items():
        if not root.exists():
            continue
        files = [f for f in root.rglob("*.parquet")
                 if model_t is None or f.stat().st_mtime >= model_t]
        n = 0
        for f in files:
            try:
                d = pd.read_parquet(f, columns=want)
            except Exception:
                continue
            d["CWNS_ID"] = d["CWNS_ID"].astype(str)
            d = d[d["CWNS_ID"].isin(plant_ids)]
            if len(d):
                d["source"] = source
                frames.append(d)
                n += len(d)
        print(f"  objects from {source:<16} {n:>9,}  ({len(files)} current file(s))")
    if not frames:
        return pd.DataFrame(columns=want + ["source"])
    return pd.concat(frames, ignore_index=True)


def load_parcels(wanted: pd.DataFrame) -> pd.DataFrame:
    """Geometry (EPSG:4326 WKB in the mirror) and owner for (STATE_CODE,
    ll_uuid) pairs. Returns ll_uuid, owner, geometry (shapely, 4326)."""
    from shapely import wkb
    con = C.duckdb_connect()
    rows = []
    for state, grp in wanted.dropna(subset=["ll_uuid"]).groupby("STATE_CODE"):
        con.register("want", pd.DataFrame({"ll_uuid": grp["ll_uuid"].astype(str).unique()}))
        try:
            res = con.execute(f"""
                SELECT p.{C.PARCEL_ID_FIELD} AS ll_uuid, p.owner AS owner,
                       p.{C.PARCEL_WKB_FIELD} AS parcel_wkb
                FROM read_parquet('{C.PARCEL_BASE.as_posix()}/state={state}/*.parquet') p
                JOIN want ON want.ll_uuid = p.{C.PARCEL_ID_FIELD}
            """).df().drop_duplicates(subset="ll_uuid")
        except Exception as e:
            print(f"    [{state}] parcel lookup failed: {str(e)[:160]}")
            continue
        finally:
            con.unregister("want")
        res["geometry"] = [wkb.loads(bytes(b)) for b in res["parcel_wkb"]]
        rows.append(res.drop(columns="parcel_wkb"))
    con.close()
    if not rows:
        return pd.DataFrame(columns=["ll_uuid", "owner", "geometry"])
    out = pd.concat(rows, ignore_index=True)
    out["ll_uuid"] = out["ll_uuid"].astype(str)
    return out


def main():
    import geopandas as gpd
    from shapely.ops import unary_union

    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, required=True,
                    help="re-rank score the #1 parcel must reach to move a "
                         "flagged plant (from calibrate_move_rule.py)")
    ap.add_argument("--require-detection", action="store_true",
                    help="never move when nothing fired in the pool")
    ap.add_argument("--allow-low-od", action="store_true",
                    help="build even when national candidate OD is missing "
                         "(testing only -- moves then rest on Stage 2a)")
    args = ap.parse_args()

    print("=== 13_build_corrected_output.py ===")
    print(f"move rule: Stage 1 flagged AND re-rank #1 >= {args.cutoff}"
          + ("  AND a detection fired in the pool" if args.require_detection else ""))
    print(f"site rule: neighbours within {SG.SITE_GAP_M:.0f} m with a detection "
          f"(conf >= {SG.MIN_OBJ_CONF}) or #1's owner (re-rank >= "
          f"{SG.OWNER_MIN_RERANK}); max {SG.SITE_MAX_PARCELS} parcels")

    master = load_master()
    summ, cand = load_scored()

    top = cand[cand["rerank_rank"] == 1]
    cov = top["od_ran"].fillna(False).astype(bool).mean() if len(top) else 0.0
    print(f"  re-ranked #1 candidates with a current-detector result: {cov:.1%}")
    if cov < MIN_OD_COVERAGE and not args.allow_low_od:
        print(f"\nRefusing: under {MIN_OD_COVERAGE:.0%} of #1 candidates were "
              f"examined by the detector, so the re-rank is mostly Stage 2a "
              f"order. Run the national 01e array, then 05b, then this.")
        sys.exit(1)

    pop_ok = C.population_ok_ids()
    scored_states = set(summ["STATE_CODE"].dropna().astype(str))

    if "Site_UUIDs" not in master.columns:   # masters written before 2026-10-05
        master["Site_UUIDs"] = None
    df = master[["CWNS_ID", "STATE_CODE", "FACILITY_NAME", "LATITUDE", "LONGITUDE",
                 "Original_Correct", "Verified", "Corrected_X", "Corrected_Y",
                 "How_Corrected", "ReGrid_UUID", "Site_UUIDs"]].rename(
        columns={"LATITUDE": "reported_lat", "LONGITUDE": "reported_lon"})
    keep = ["CWNS_ID", "stage1_route", "trigger_reason", "stage1_prob_correct",
            "reported_ll_uuid", "rerank_top_ll_uuid", "rerank_score",
            "rerank_fallback", "rerank_top_stage2a_rank"]
    df = df.merge(summ[[c for c in keep if c in summ.columns]], on="CWNS_ID", how="left")

    df["status"] = pd.NA
    df["status_reason"] = ""
    df["out_lat"] = df["reported_lat"]
    df["out_lon"] = df["reported_lon"]
    df["moved_to_ll_uuid"] = pd.NA
    df["coord_method"] = "reported"
    df["site_parcels"] = ""
    df["n_site_parcels"] = 0
    df["n_objects"] = 0
    df["coord_in_site"] = pd.NA

    verified = df["Verified"] == "Yes"
    v_ok = verified & (df["Original_Correct"] == "Yes")
    v_fix = verified & (df["Original_Correct"] == "No") & df["Corrected_X"].notna()
    df.loc[v_ok, "status"] = "verified_correct"
    df.loc[v_fix, "status"] = "verified_corrected"
    df.loc[v_fix, "out_lat"] = df.loc[v_fix, "Corrected_Y"]
    df.loc[v_fix, "out_lon"] = df.loc[v_fix, "Corrected_X"]
    df.loc[v_fix, "coord_method"] = "reviewer"
    df.loc[v_fix, "status_reason"] = df.loc[v_fix, "How_Corrected"].fillna("")
    unresolved = df["status"].isna() & (df["Original_Correct"] == "No")
    df.loc[unresolved, "status"] = "reviewed_unresolved"

    open_ = df["status"].isna()
    in_scope = df["stage1_route"].notna()
    na = open_ & ~in_scope
    df.loc[na, "status"] = "not_assessed"
    df.loc[na, "coord_method"] = "reported"
    df.loc[na & df["reported_lat"].isna(), "status_reason"] = "no_coordinates"
    df.loc[na & (df["status_reason"] == "") & ~df["STATE_CODE"].isin(scored_states),
           "status_reason"] = "no_imagery_state"
    df.loc[na & (df["status_reason"] == "") & ~df["CWNS_ID"].isin(pop_ok),
           "status_reason"] = f"population_le_{C.MIN_POP_SERVED}_or_unknown"
    df.loc[na & (df["status_reason"] == ""), "status_reason"] = "not_scored"

    open_ = df["status"].isna()
    df.loc[open_ & (df["stage1_route"] == "osm_confirmed"), "status"] = "kept_osm"
    open_ = df["status"].isna()
    df.loc[open_ & (df["trigger_reason"] == "none"), "status"] = "kept_model"

    open_ = df["status"].isna()
    flagged = open_ & df["trigger_reason"].isin(["low_confidence", "no_parcel"])
    can_move = flagged & df["rerank_top_ll_uuid"].notna() & (df["rerank_score"] >= args.cutoff)
    if args.require_detection:
        can_move &= ~df["rerank_fallback"].fillna(True).astype(bool)
    df.loc[flagged, "status"] = "flagged_not_moved"
    df.loc[flagged, "status_reason"] = df.loc[flagged, "trigger_reason"]
    print(f"\n  flagged plants: {int(flagged.sum()):,}  |  clearing the move rule: "
          f"{int(can_move.sum()):,}")

    # ---- the one parcel each non-moving plant is drawn on ------------------
    # Verified plants: the master's parcel when it has one (ReGrid_UUID), else
    # the reported parcel. Everyone else in scope: the reported parcel.
    df["final_parcel"] = df["reported_ll_uuid"]
    has_rg = verified & df["ReGrid_UUID"].notna() & (df["ReGrid_UUID"].astype(str) != "")
    df.loc[has_rg, "final_parcel"] = df.loc[has_rg, "ReGrid_UUID"]
    # A reviewer's split-parcel site (master Site_UUIDs, primary first) wins.
    has_site_lbl = verified & df["Site_UUIDs"].fillna("").astype(str).str.len().gt(0)
    df.loc[has_site_lbl, "final_parcel"] = df.loc[has_site_lbl, "Site_UUIDs"].str.split(";").str[0]
    verified_sites = {cw: s.split(";") for cw, s in
                      zip(df.loc[has_site_lbl, "CWNS_ID"], df.loc[has_site_lbl, "Site_UUIDs"])}

    movers = df.loc[can_move, ["CWNS_ID", "STATE_CODE", "rerank_top_ll_uuid",
                               "reported_ll_uuid"]]
    pool = cand[cand["CWNS_ID"].isin(movers["CWNS_ID"])][
        ["CWNS_ID", "STATE_CODE", "ll_uuid", "rerank_score"]]
    rep = movers.dropna(subset=["reported_ll_uuid"]).rename(
        columns={"reported_ll_uuid": "ll_uuid"})[["CWNS_ID", "STATE_CODE", "ll_uuid"]]
    rep["rerank_score"] = np.nan
    pool = pd.concat([pool, rep], ignore_index=True)
    pool["ll_uuid"] = pool["ll_uuid"].astype(str)

    singles = df.loc[df["final_parcel"].notna() & ~can_move,
                     ["CWNS_ID", "STATE_CODE", "final_parcel"]] \
                .rename(columns={"final_parcel": "ll_uuid"})
    st_of = dict(zip(df["CWNS_ID"], df["STATE_CODE"]))
    extra = pd.DataFrame([(st_of.get(cw), u) for cw, us in verified_sites.items() for u in us],
                         columns=["STATE_CODE", "ll_uuid"])
    wanted = pd.concat([pool[["STATE_CODE", "ll_uuid"]], singles[["STATE_CODE", "ll_uuid"]], extra])
    wanted = wanted.drop_duplicates()
    print(f"\n  loading {len(wanted):,} parcel geometries...")
    parcels = load_parcels(wanted)
    print(f"  found {len(parcels):,}")
    pg = gpd.GeoDataFrame(parcels, geometry="geometry", crs=4326).to_crs(C.PROJECTED_CRS)
    geom_m = dict(zip(pg["ll_uuid"], pg.geometry))
    geom_ll = dict(zip(parcels["ll_uuid"], parcels["geometry"]))
    owner_norm = {u: NM.normalise(o) if isinstance(o, str) else ""
                  for u, o in zip(parcels["ll_uuid"], parcels["owner"])}

    print("\n  loading detected objects...")
    plant_ids = set(df.loc[df["final_parcel"].notna() | can_move, "CWNS_ID"])
    objs = load_objects(plant_ids)
    if len(objs):
        og = gpd.GeoSeries(gpd.points_from_xy(objs["lon"], objs["lat"]), crs=4326) \
               .to_crs(C.PROJECTED_CRS)
        objs["x"], objs["y"] = og.x.to_numpy(), og.y.to_numpy()
    objs_by_plant = {k: SG.dedupe_objects(g) for k, g in objs.groupby("CWNS_ID")} \
        if len(objs) else {}
    empty_objs = pd.DataFrame(columns=["x", "y", "class_name", "max_confidence",
                                       "lon", "lat", "source"])

    # ---- moves: assemble the site, place the point ---------------------------
    site_of: dict[str, list[str]] = {}
    n_lost = n_site = 0
    pool_by_plant = dict(tuple(pool.groupby("CWNS_ID")))
    for r in movers.itertuples(index=False):
        cw, seed = r.CWNS_ID, str(r.rerank_top_ll_uuid)
        if seed not in geom_m:
            n_lost += 1
            df.loc[df["CWNS_ID"] == cw, "status_reason"] = "move_parcel_not_found"
            continue
        p = pool_by_plant[cw]
        p = p[p["ll_uuid"].isin(geom_m)].copy()
        p["geometry"] = p["ll_uuid"].map(geom_m)
        p["owner_norm"] = p["ll_uuid"].map(owner_norm)
        o = objs_by_plant.get(cw, empty_objs)
        site = SG.assemble_site(seed, p, o)
        site_of[cw] = site
        i = df.index[df["CWNS_ID"] == cw]
        df.loc[i, "site_parcels"] = ";".join(site)
        df.loc[i, "n_site_parcels"] = len(site)
        if len(site) > 1:
            n_site += 1

        if pd.notna(r.reported_ll_uuid) and str(r.reported_ll_uuid) in site:
            df.loc[i, "status"] = "kept_site"
            df.loc[i, "status_reason"] = "reported parcel is part of the #1 site"
            continue

        x, y, method, n = SG.site_point([geom_m[u] for u in site], geom_m[seed], o)
        pt = gpd.GeoSeries(gpd.points_from_xy([x], [y]), crs=C.PROJECTED_CRS).to_crs(4326).iloc[0]
        union = unary_union([geom_m[u] for u in site])
        df.loc[i, "out_lat"] = pt.y
        df.loc[i, "out_lon"] = pt.x
        df.loc[i, "moved_to_ll_uuid"] = seed
        df.loc[i, "coord_method"] = method
        df.loc[i, "n_objects"] = n
        df.loc[i, "coord_in_site"] = bool(union.buffer(1.0).contains(
            gpd.points_from_xy([x], [y])[0]))
        df.loc[i, "status"] = "moved"
    print(f"\n  moves: {int((df['status'] == 'moved').sum()):,} moved, "
          f"{int((df['status'] == 'kept_site').sum()):,} kept_site (reported parcel "
          f"in the site), {n_site:,} multi-parcel site(s), {n_lost:,} parcel(s) not found")

    leftover = df["status"].isna()
    if leftover.any():
        df.loc[leftover, "status"] = "not_assessed"
        df.loc[leftover, "status_reason"] = "unrouted"
        print(f"  WARNING: {int(leftover.sum())} plant(s) matched no rule (unrouted)")

    # ---- site of every other plant: the reviewer's parcels, else its one ----
    for r in df.loc[df["final_parcel"].notna() & ~df["CWNS_ID"].isin(site_of),
                    ["CWNS_ID", "final_parcel"]].itertuples(index=False):
        parts = [u for u in verified_sites.get(r.CWNS_ID, [str(r.final_parcel)]) if u in geom_m]
        if parts:
            site_of[r.CWNS_ID] = parts
    has_site = df["CWNS_ID"].isin(site_of) & (df["n_site_parcels"] == 0)
    df.loc[has_site, "site_parcels"] = df.loc[has_site, "CWNS_ID"].map(lambda c: ";".join(site_of[c]))
    df.loc[has_site, "n_site_parcels"] = df.loc[has_site, "CWNS_ID"].map(lambda c: len(site_of[c]))

    # Population served (the same table and first-row-wins rule as the
    # population floor), so downstream readers can slice by band without the
    # HPC-only CWNS export.
    df["pop_served"] = df["CWNS_ID"].map(C.population_served())
    df["pop_band"] = df["pop_served"].map(C.pop_band)

    # How far to trust a location (decided 2026-10-05): a human said so >
    # model decision on a plant serving > 1,000 > model decision on a newly
    # admitted 100-1,000 plant, whose precision has not been measured yet.
    model_decided = df["status"].isin(["moved", "kept_site", "kept_model",
                                       "kept_osm", "flagged_not_moved"])
    df["confidence_tier"] = "none"
    df.loc[df["status"].isin(["verified_correct", "verified_corrected"]),
           "confidence_tier"] = "verified"
    df.loc[model_decided, "confidence_tier"] = "model"
    df.loc[model_decided & (df["pop_band"] != ">1k"), "confidence_tier"] = "model_small_plant"

    df["moved_m"] = np.where(
        df["status"].isin(["moved", "verified_corrected"]),
        haversine_m(df["reported_lat"], df["reported_lon"], df["out_lat"], df["out_lon"]),
        0.0)
    df["move_cutoff"] = args.cutoff
    df["require_detection"] = args.require_detection
    df["built"] = date.today().isoformat()

    # ---- sites + detections layers -------------------------------------------
    status_of = dict(zip(df["CWNS_ID"], df["status"]))
    site_rows, det_rows = [], []
    for cw, site in site_of.items():
        geoms = [geom_ll[u] for u in site if u in geom_ll]
        if not geoms:
            continue
        site_rows.append({"CWNS_ID": cw, "status": status_of.get(cw),
                          "site_parcels": ";".join(site), "n_parcels": len(site),
                          "geometry": unary_union(geoms)})
        o = objs_by_plant.get(cw)
        if o is None or o.empty:
            continue
        union_m = unary_union([geom_m[u] for u in site if u in geom_m])
        pts = gpd.points_from_xy(o["x"], o["y"])
        inside = o[[union_m.contains(p) for p in pts]]
        for d in inside.itertuples(index=False):
            det_rows.append({
                "CWNS_ID": cw, "class_name": d.class_name,
                "confidence": float(d.max_confidence), "source": d.source,
                "used_for_coord": bool(status_of.get(cw) == "moved"
                                       and d.max_confidence >= SG.MIN_OBJ_CONF),
                "lon": float(d.lon), "lat": float(d.lat)})
    sites = gpd.GeoDataFrame(site_rows, geometry="geometry", crs=4326) if site_rows else \
        gpd.GeoDataFrame(columns=["CWNS_ID", "status", "site_parcels", "n_parcels", "geometry"],
                         geometry="geometry", crs=4326)
    dets = pd.DataFrame(det_rows, columns=["CWNS_ID", "class_name", "confidence", "source",
                                           "used_for_coord", "lon", "lat"])
    print(f"\n  sites layer: {len(sites):,} plant site(s)  |  detections layer: "
          f"{len(dets):,} object(s)")

    cols = ["CWNS_ID", "STATE_CODE", "FACILITY_NAME", "pop_served", "pop_band",
            "confidence_tier", "status", "status_reason",
            "reported_lat", "reported_lon", "out_lat", "out_lon", "moved_m",
            "coord_method", "coord_in_site", "n_objects", "site_parcels",
            "n_site_parcels", "moved_to_ll_uuid", "reported_ll_uuid", "stage1_route",
            "stage1_prob_correct", "rerank_top_ll_uuid", "rerank_score",
            "rerank_fallback", "move_cutoff", "require_detection", "built"]
    out = df[cols].sort_values(["STATE_CODE", "CWNS_ID"]).reset_index(drop=True)
    out["coord_in_site"] = out["coord_in_site"].astype("boolean")

    print("\n=== status ===")
    vc = out["status"].value_counts()
    for s, n in vc.items():
        print(f"  {s:<22} {n:>7,}  ({n / len(out):.1%})")
    print(f"  {'TOTAL':<22} {len(out):>7,}")
    print("\n  by population band:")
    print(pd.crosstab(out["status"], out["pop_band"]).to_string())
    na_r =out.loc[out["status"] == "not_assessed", "status_reason"].value_counts()
    if len(na_r):
        print("  not_assessed by reason: " + ", ".join(f"{k} {v:,}" for k, v in na_r.items()))
    mv = out[out["status"] == "moved"]
    if len(mv):
        print(f"  moved distance: median {mv['moved_m'].median():,.0f} m, "
              f"90th pct {mv['moved_m'].quantile(0.9):,.0f} m, max {mv['moved_m'].max():,.0f} m")
        print("  moved point from: " + ", ".join(
            f"{k} {v:,}" for k, v in mv["coord_method"].value_counts().items()))
        off = int((mv["coord_in_site"] == False).sum())  # noqa: E712
        print(f"  moved points outside their site (mean centre fell between "
              f"parcels): {off:,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / "cwns_corrected_locations"
    out.to_parquet(base.with_suffix(".parquet"), index=False)
    out.to_csv(base.with_suffix(".csv"), index=False)
    gpkg = base.with_suffix(".gpkg")
    gpd.GeoDataFrame(out, geometry=gpd.points_from_xy(out["out_lon"], out["out_lat"]),
                     crs=4326).to_file(gpkg, layer="plants", driver="GPKG")
    sites.to_file(gpkg, layer="sites", driver="GPKG")
    gpd.GeoDataFrame(dets, geometry=gpd.points_from_xy(dets["lon"], dets["lat"]),
                     crs=4326).to_file(gpkg, layer="detections", driver="GPKG")
    print(f"\nWritten: {base}.parquet / .csv / .gpkg (layers plants, sites, detections)")

    # Copies where git can carry them (data/ is ignored, diagnostics/ is not):
    # viewer/build_data.py on the work computer reads exactly these paths.
    # Site outlines are simplified to ~1 m for the map.
    SHARE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SHARE_DIR / "cwns_corrected_locations.parquet", index=False)
    s = sites.copy()
    s["geometry"] = s.geometry.simplify(0.00001, preserve_topology=True)
    s.to_parquet(SHARE_DIR / "cwns_sites.parquet", index=False)
    dets.to_parquet(SHARE_DIR / "cwns_detections.parquet", index=False)
    print(f"Copied for the viewer: {SHARE_DIR}/  (git add correction/diagnostics/output/)")


if __name__ == "__main__":
    main()
