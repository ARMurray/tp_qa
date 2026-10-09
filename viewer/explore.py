"""
explore.py -- candidates, detected objects and parcels for the viewer
=====================================================================
Everything behind "why was this plant (not) moved, and what else is there?"

DATA, and where each piece lives
    viewer_candidates.parquet, viewer_objects.parquet
        correction/diagnostics/output/, written on the HPC by
        export_viewer_data.py (run by 13_build_corrected_output.slurm) and
        committed. Every scored candidate of every flagged plant -- not only
        the #1 or those above the move cutoff -- and every detected object.
        A few MB each; loaded into memory once.
    Parcels (geometry + Regrid attributes)
        NOT in git. Read live from the local Regrid mirror the review app
        uses (review_app/config.py REGRID_ROOT), with DuckDB, only for what
        is asked for: a plant's candidates by ll_uuid, or every parcel in a
        small map window by H3 cell (h3_index_9, the column 01a fetches by).
        Results are cached in memory per (state, cell) and per parcel, so
        panning back is instant. A state's files are scanned per query, so
        the first view in a state can take a few seconds.

Nothing here writes anything.
"""
import json
import math
import os
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SHARE = REPO / "correction" / "diagnostics" / "output"
CANDS_PARQUET = Path(os.environ.get("VIEWER_CANDIDATES", SHARE / "viewer_candidates.parquet"))
OBJECTS_PARQUET = Path(os.environ.get("VIEWER_OBJECTS", SHARE / "viewer_objects.parquet"))

ATTR_COLS = ["owner", "ll_gisacre", "ll_bldg_count", "lbcs_activity_desc",
             "lbcs_function_desc", "lbcs_structure_desc", "lbcs_site_desc",
             "lbcs_ownership_desc", "zoning_type", "zoning_subtype"]
OPTIONAL_COLS = ["address", "usedesc", "scity"]   # used when the mirror has them

MAX_VIEW_KM = 14.0         # widest window parcels are fetched for (zoom 14 on a wide screen)
MAX_VIEW_PARCELS = 15000   # cap per response
H3_RES = 9


def _regrid_config():
    """review_app/config.py's REGRID_ROOT / glob / column names, unless
    VIEWER_REGRID_ROOT overrides the root."""
    sys.path.insert(0, str(REPO / "review_app"))
    try:
        import config as RC          # review_app's config (the viewer has none)
    finally:
        sys.path.pop(0)
    root = Path(os.environ.get("VIEWER_REGRID_ROOT", RC.REGRID_ROOT))
    return root, RC.REGRID_STATE_GLOB, RC.REGRID_GEOM_COL, RC.REGRID_UUID_COL


# ---------------------------------------------------------------------------
# Candidates and objects (from the HPC export)
# ---------------------------------------------------------------------------
class Scored:
    def __init__(self):
        self.cands = pd.DataFrame()
        self.by_plant: dict[str, pd.DataFrame] = {}
        self.by_parcel: dict[str, list[dict]] = {}
        self.obj = None          # dict of numpy arrays
        self.source = ""

    def load(self):
        if CANDS_PARQUET.exists():
            c = pd.read_parquet(CANDS_PARQUET)
            c["CWNS_ID"] = c["CWNS_ID"].astype(str)
            c["ll_uuid"] = c["ll_uuid"].astype(str)
            sort = "rerank_rank" if "rerank_rank" in c.columns else "stage2a_rank"
            c = c.sort_values(["CWNS_ID", sort], na_position="last")
            self.cands = c
            self.by_plant = {k: g for k, g in c.groupby("CWNS_ID", sort=False)}
            idx: dict[str, list[dict]] = {}
            for r in c[["ll_uuid", "CWNS_ID"] + [x for x in ("rerank_rank", "rerank_score",
                                                             "od_has_detection") if x in c.columns]
                       ].itertuples(index=False):
                idx.setdefault(r.ll_uuid, []).append({
                    "id": r.CWNS_ID,
                    "rank": _int(getattr(r, "rerank_rank", None)),
                    "rr": _num(getattr(r, "rerank_score", None), 3),
                    "od": _bool(getattr(r, "od_has_detection", None)),
                })
            self.by_parcel = idx
        if OBJECTS_PARQUET.exists():
            o = pd.read_parquet(OBJECTS_PARQUET)
            o = o[o["lon"].notna() & o["lat"].notna()]
            self.obj = {
                "lon": o["lon"].to_numpy("float64"), "lat": o["lat"].to_numpy("float64"),
                "cls": o["class_name"].astype(str).to_numpy(),
                "conf": o["confidence"].to_numpy("float64"),
                "src": o["source"].astype(str).to_numpy(),
                "id": o["CWNS_ID"].astype(str).to_numpy(),
                "parcel": o["ll_uuid"].astype("string").fillna("").to_numpy(),
            }
        self.source = (f"{len(self.cands):,} candidates / {len(self.by_plant):,} plants"
                       if len(self.cands) else "no viewer_candidates.parquet")
        if self.obj is not None:
            self.source += f", {len(self.obj['lon']):,} detected objects"

    def candidates_of(self, plant_id: str) -> list[dict]:
        g = self.by_plant.get(plant_id)
        if g is None:
            return []
        out = []
        for r in g.to_dict(orient="records"):
            out.append({
                "parcel": r["ll_uuid"],
                "state": r.get("STATE_CODE"),
                "rank": _int(r.get("rerank_rank")),
                "rank2a": _int(r.get("stage2a_rank")),
                "rr": _num(r.get("rerank_score"), 3),
                "s2a": _num(r.get("stage2_prob_correct"), 3),
                "fb": _bool(r.get("rerank_fallback")),
                "dist": _num(r.get("distance_m"), 0),
                "lat": _num(r.get("centroid_lat"), 6), "lon": _num(r.get("centroid_lng"), 6),
                "od_ran": _bool(r.get("od_ran")),
                "od": _bool(r.get("od_has_detection")),
                "nobj": _int(r.get("od_n_objects")),
                "odconf": _num(r.get("od_max_confidence"), 2),
                "odcls": _txt(r.get("od_dominant_class")),
                "name": _num(r.get("name_match_score"), 2),
                "kw": _bool(r.get("has_ww_keyword")),
                "osm": _bool(r.get("osm_ww")),
                "util": _bool(r.get("utility_owner")),
                "acres": _num(r.get("ll_gisacre"), 2),
                "bldg": _int(r.get("ll_bldg_count")),
                "land": _txt(r.get("dominant_class_group")),
            })
        return out

    def objects_in(self, w, s, e, n, limit=20000) -> list[dict]:
        if self.obj is None:
            return []
        o = self.obj
        m = (o["lon"] >= w) & (o["lon"] <= e) & (o["lat"] >= s) & (o["lat"] <= n)
        ix = np.flatnonzero(m)[:limit]
        return [{"lon": round(float(o["lon"][i]), 6), "lat": round(float(o["lat"][i]), 6),
                 "cls": o["cls"][i], "conf": round(float(o["conf"][i]), 3),
                 "src": o["src"][i], "id": o["id"][i], "parcel": o["parcel"][i] or None}
                for i in ix]


# ---------------------------------------------------------------------------
# Parcels (live, local Regrid mirror)
# ---------------------------------------------------------------------------
class Parcels:
    def __init__(self):
        self._con = None
        self._lock = threading.Lock()
        self._cols: dict[str, list[str]] = {}
        self._cells: OrderedDict = OrderedDict()     # (state, cell) -> [feature]
        self._by_uuid: OrderedDict = OrderedDict()   # uuid -> feature
        self.cfg = None
        self.spatial = True

    def _connect(self):
        if self._con is None:
            import duckdb
            self.cfg = _regrid_config()
            self._con = duckdb.connect()
            try:
                self._con.execute("INSTALL spatial; LOAD spatial;")
            except Exception as e:
                # No spatial extension (offline, blocked download): read the
                # raw WKB and convert it in Python with shapely instead.
                print(f"[viewer] DuckDB spatial unavailable ({str(e)[:80]}); decoding WKB with shapely")
                self.spatial = False
            try:
                self._con.execute("SET enable_geoparquet_conversion = false;")
            except Exception:
                pass
        return self._con

    def _glob(self, state):
        root, pattern, _, _ = self.cfg
        return (root / pattern.format(state=state)).as_posix()

    def _select(self, state) -> str:
        """Columns to read: the confirmed attribute set plus optional extras
        this state's files actually have."""
        if state not in self._cols:
            have = set(self._con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{self._glob(state)}')").df()["column_name"])
            self._cols[state] = [c for c in ATTR_COLS + OPTIONAL_COLS if c in have]
        _, _, geom, uid = self.cfg
        cols = ", ".join(self._cols[state])
        g = f"ST_AsGeoJSON(ST_GeomFromWKB({geom})) AS geojson" if self.spatial else f"{geom} AS wkb"
        return f"{uid} AS ll_uuid, {g}" + (f", {cols}" if cols else "")

    def _features(self, df: pd.DataFrame, state: str) -> list[dict]:
        feats = []
        for r in df.to_dict(orient="records"):
            if "wkb" in r:
                raw = r.pop("wkb")
                if raw is None:
                    continue
                from shapely import wkb as _wkb
                from shapely.geometry import mapping
                geom = mapping(_wkb.loads(bytes(raw)))
            else:
                g = r.pop("geojson", None)
                if not g:
                    continue
                geom = json.loads(g)
            props = {k: _clean(v) for k, v in r.items()}
            props["state"] = state
            feats.append({"type": "Feature", "geometry": _round_coords(geom), "properties": props})
        return feats

    def by_uuids(self, state: str, uuids: list[str]) -> list[dict]:
        want = [u for u in dict.fromkeys(uuids) if u]
        with self._lock:
            have = {u: self._by_uuid[u] for u in want if u in self._by_uuid}
            need = [u for u in want if u not in have]
            if need:
                con = self._connect()
                con.register("want", pd.DataFrame({"u": need}))
                try:
                    df = con.execute(f"""
                        SELECT {self._select(state)}
                        FROM read_parquet('{self._glob(state)}') p
                        JOIN want ON want.u = p.{self.cfg[3]}
                    """).df().drop_duplicates(subset="ll_uuid")
                finally:
                    con.unregister("want")
                for f in self._features(df, state):
                    self._remember(f)
                    have[f["properties"]["ll_uuid"]] = f
        return [have[u] for u in want if u in have]

    def in_view(self, states: list[str], w, s, e, n) -> dict:
        width_km = (e - w) * 111.32 * math.cos(math.radians((s + n) / 2))
        height_km = (n - s) * 110.57
        if width_km > MAX_VIEW_KM or height_km > MAX_VIEW_KM:
            return {"features": [], "too_big": True}
        cells = _cells_for_bbox(w, s, e, n)
        out, truncated, missing = [], False, []
        with self._lock:
            con = self._connect()
            for state in states:
                need = [c for c in cells if (state, c) not in self._cells]
                if need:
                    flt = ", ".join(f"'{c}'" for c in need)
                    try:
                        df = con.execute(f"""
                            SELECT h3_index_9 AS _cell, {self._select(state)}
                            FROM read_parquet('{self._glob(state)}')
                            WHERE h3_index_9 IN ({flt})
                        """).df()
                    except Exception as e:
                        # A neighbouring state with no folder in the local
                        # mirror: skip it rather than fail the whole window.
                        missing.append(f"{state}: {str(e)[:120]}")
                        continue
                    got = {c: [] for c in need}
                    for cell, grp in df.groupby("_cell"):
                        got.setdefault(cell, []).extend(
                            self._features(grp.drop(columns="_cell"), state))
                    for c, feats in got.items():
                        self._cells[(state, c)] = feats
                        for f in feats:
                            self._remember(f)
                    while len(self._cells) > 20000:
                        self._cells.popitem(last=False)
                seen = set()
                for c in cells:
                    for f in self._cells.get((state, c), []):
                        u = f["properties"]["ll_uuid"]
                        if u not in seen:
                            seen.add(u)
                            out.append(f)
            if len(out) > MAX_VIEW_PARCELS:
                out, truncated = out[:MAX_VIEW_PARCELS], True
        if missing:
            print("[viewer] parcels skipped for " + "; ".join(missing))
        return {"features": out, "too_big": False, "truncated": truncated,
                "skipped": [m.split(":")[0] for m in missing]}

    def _remember(self, f):
        self._by_uuid[f["properties"]["ll_uuid"]] = f
        while len(self._by_uuid) > 200000:
            self._by_uuid.popitem(last=False)


def _round_coords(geom: dict, nd: int = 6) -> dict:
    """~0.1 m precision is plenty for drawing; halves the JSON for a zoom-14
    window of thousands of parcels."""
    def rnd(c):
        if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
            return [round(float(x), nd) for x in c]
        return [rnd(x) for x in c]
    if "coordinates" in geom:
        geom = {**geom, "coordinates": rnd(geom["coordinates"])}
    elif geom.get("type") == "GeometryCollection":
        geom = {**geom, "geometries": [_round_coords(g, nd) for g in geom.get("geometries", [])]}
    return geom


def _cells_for_bbox(w, s, e, n) -> list[str]:
    """H3 res-9 cells covering the box, plus one ring: h3_index_9 is the cell of
    a parcel's centroid, so a parcel overlapping the edge can sit just outside.
    Sampled on a ~80 m grid (res-9 cells are ~170 m across), which needs no
    polygon API and so works on any h3 4.x."""
    import h3
    lat_step = 80 / 110_570
    lon_step = 80 / (111_320 * max(0.2, math.cos(math.radians((s + n) / 2))))
    cells = set()
    lat = s
    while lat <= n + lat_step:
        lon = w
        while lon <= e + lon_step:
            cells.add(h3.latlng_to_cell(min(lat, n), min(lon, e), H3_RES))
            lon += lon_step
        lat += lat_step
    ring = set()
    for c in cells:
        ring |= set(h3.grid_disk(c, 1))
    return sorted(ring)


def _num(v, nd=None):
    if v is None or v is pd.NA:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return round(f, nd) if nd is not None else f


def _int(v):
    f = _num(v)
    return None if f is None else int(f)


def _bool(v):
    if v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)):
        return None
    return bool(v)


def _txt(v):
    return None if v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)) or v == "" else str(v)


def _clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) else round(float(v), 3)
    return v
