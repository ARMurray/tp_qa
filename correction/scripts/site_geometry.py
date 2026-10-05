"""
site_geometry.py
================
Pure geometry for 13_build_corrected_output.py: which parcels make up a
plant's site, and where its output point goes. No I/O -- 13 loads parcels
and detected objects and hands them in, so this can be tested anywhere.

Decided 2026-10-05 (docs/PLAN_20261005_detector_population_sites.md):

SITE  Plants often span several parcels (roads split them). Starting from the
      seed parcel (the re-ranker's #1), a pool parcel joins when it lies
      within SITE_GAP_M of the site so far AND either
        - a detected object (confidence >= MIN_OBJ_CONF) lies inside it, or
        - it has the seed's normalised owner and re-rank >= OWNER_MIN_RERANK
          (the reported parcel has no re-rank score; owner alone suffices).
      Growth repeats until nothing joins or the site holds SITE_MAX_PARCELS.

POINT detections (>= MIN_OBJ_CONF) inside the site -> their mean centre;
      none -> the seed parcel's centroid if it falls inside the parcel, else
      shapely.polylabel (pole of inaccessibility: the interior point farthest
      from the edges -- right for L-shaped, ring and multi-part parcels).

All geometry here is in a projected CRS in metres (EPSG:5070).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point
from shapely.ops import polylabel, unary_union

SITE_GAP_M = 50.0
SITE_MAX_PARCELS = 4
MIN_OBJ_CONF = 0.4
OWNER_MIN_RERANK = 0.5
DEDUPE_M = 12.0   # = config.NMS_DISTANCE_M: one physical object seen from two tiles


def interior_point(geom) -> tuple[Point, str]:
    """Centroid when it is inside the polygon, else the pole of
    inaccessibility. For a MultiPolygon, the largest part."""
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    c = geom.centroid
    if geom.contains(c):
        return c, "centroid"
    tol = max(0.5, np.sqrt(geom.area) / 100.0)
    return polylabel(geom, tolerance=tol), "polylabel"


def dedupe_objects(objs: pd.DataFrame) -> pd.DataFrame:
    """One row per physical object. Neighbouring candidates' tiles overlap, so
    the same clarifier is detected once per candidate; keep the most
    confident of any same-class objects within DEDUPE_M. Expects x, y
    (metres), class_name, max_confidence."""
    if objs.empty:
        return objs
    objs = objs.sort_values("max_confidence", ascending=False).reset_index(drop=True)
    keep = np.ones(len(objs), dtype=bool)
    xy = objs[["x", "y"]].to_numpy()
    cls = objs["class_name"].to_numpy()
    for i in range(len(objs)):
        if not keep[i]:
            continue
        d = np.hypot(xy[i + 1:, 0] - xy[i, 0], xy[i + 1:, 1] - xy[i, 1])
        dup = (d <= DEDUPE_M) & (cls[i + 1:] == cls[i])
        keep[i + 1:][dup] = False
    return objs[keep].reset_index(drop=True)


def _objects_inside(geom, objs: pd.DataFrame) -> pd.DataFrame:
    if objs.empty:
        return objs
    pts = shapely.points(objs["x"].to_numpy(), objs["y"].to_numpy())
    return objs[shapely.contains(geom, pts)]


def assemble_site(seed: str, pool: pd.DataFrame, objs: pd.DataFrame) -> list[str]:
    """pool: one row per parcel the plant could own -- ll_uuid, geometry
    (shapely, metres), rerank_score (NaN for the reported parcel), owner_norm.
    objs: deduped objects with x, y, max_confidence. Returns the site's
    ll_uuids, seed first."""
    pool = pool.drop_duplicates(subset="ll_uuid").set_index("ll_uuid")
    if seed not in pool.index:
        return [seed]
    strong = objs[objs["max_confidence"] >= MIN_OBJ_CONF] if not objs.empty else objs
    seed_owner = pool.at[seed, "owner_norm"] or ""

    fired = {u: len(_objects_inside(g, strong)) > 0 for u, g in pool["geometry"].items()}
    site = [seed]
    site_geom = pool.at[seed, "geometry"]
    while len(site) < SITE_MAX_PARCELS:
        added = False
        for u, r in pool.iterrows():
            if u in site or r["geometry"].distance(site_geom) > SITE_GAP_M:
                continue
            same_owner = bool(seed_owner) and (r["owner_norm"] or "") == seed_owner
            score = r["rerank_score"]
            owner_ok = same_owner and (pd.isna(score) or score >= OWNER_MIN_RERANK)
            if fired[u] or owner_ok:
                site.append(u)
                site_geom = unary_union([site_geom, r["geometry"]])
                added = True
                if len(site) >= SITE_MAX_PARCELS:
                    break
        if not added:
            break
    return site


def site_point(site_geoms: list, seed_geom, objs: pd.DataFrame) -> tuple[float, float, str, int]:
    """(x, y, coord_method, n_objects) in the input CRS."""
    union = unary_union(site_geoms)
    strong = objs[objs["max_confidence"] >= MIN_OBJ_CONF] if not objs.empty else objs
    inside = _objects_inside(union, strong)
    if len(inside):
        return float(inside["x"].mean()), float(inside["y"].mean()), "detections", len(inside)
    pt, method = interior_point(seed_geom)
    return pt.x, pt.y, method, 0
