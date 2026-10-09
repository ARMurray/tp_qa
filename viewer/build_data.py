"""
build_data.py
=============
Assembles the one table the viewer draws: every CWNS treatment plant, where
it is, and its location status.

SOURCE, in order of preference
    1. correction/diagnostics/output/cwns_corrected_locations.parquet
       -- 13_build_corrected_output.py's product, committed from the HPC.
    2. PREVIEW: the master alone (correction/data/training/Updates.gpkg).
       Gives the human statuses (verified_correct / verified_corrected /
       reviewed_unresolved); every other plant shows as 'pending'.

Either way the master supplies names and places, and the committed review
logs (review_app/data/outgoing/review_log_round*.parquet) supply each
plant's review history.

No tiles: ~18k points is a few MB of JSON, which deck.gl draws directly.
That is the whole reason this viewer needs no tippecanoe (and so no WSL).

    python build_data.py            # prints a status summary
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUTPUT_PARQUET = Path(os.environ.get(
    "VIEWER_OUTPUT",
    REPO / "correction" / "diagnostics" / "output" / "cwns_corrected_locations.parquet"))
MASTER_GPKG = REPO / "correction" / "data" / "training" / "Updates.gpkg"
REVIEW_DIR = REPO / "review_app" / "data" / "outgoing"
MASTER_LAYER_RE = r"^CWNS_Locations_(\d{8})(_v\d+)?$"

# Draw order: first is drawn first (bottom). Status the eye should find --
# moves and doubts -- goes on top.
SITES_PARQUET = OUTPUT_PARQUET.parent / "cwns_sites.parquet"
DETECTIONS_PARQUET = OUTPUT_PARQUET.parent / "cwns_detections.parquet"

STATUS_ORDER = ["not_assessed", "pending", "kept_model", "kept_osm", "kept_site",
                "verified_correct", "verified_corrected", "reviewed_unresolved",
                "flagged_not_moved", "moved"]


def latest_master_layer() -> str:
    import pyogrio
    names = [str(n) for n in pyogrio.list_layers(MASTER_GPKG)[:, 0]]
    dated = sorted((m.group(1), n) for n in names
                   if (m := re.match(MASTER_LAYER_RE, n)))
    if not dated:
        raise SystemExit(f"No CWNS_Locations_YYYYMMDD layer in {MASTER_GPKG}")
    return dated[-1][1]


def load_master() -> tuple[pd.DataFrame, str]:
    import geopandas as gpd
    layer = latest_master_layer()
    g = gpd.read_file(MASTER_GPKG, layer=layer)
    g = g[g["FACILITY_TYPE"] == "Treatment Plant"].copy()
    g["CWNS_ID"] = g["CWNS_ID"].astype(str)
    return pd.DataFrame(g.drop(columns="geometry")), layer


def preview_from_master(m: pd.DataFrame) -> pd.DataFrame:
    """The human statuses only -- the same rules 13 applies first."""
    df = m[["CWNS_ID", "STATE_CODE", "FACILITY_NAME", "LATITUDE", "LONGITUDE",
            "Original_Correct", "Verified", "Corrected_X", "Corrected_Y",
            "How_Corrected"]].rename(columns={"LATITUDE": "reported_lat",
                                              "LONGITUDE": "reported_lon"})
    df["status"] = "pending"
    df["status_reason"] = ""
    df["out_lat"], df["out_lon"] = df["reported_lat"], df["reported_lon"]
    v = df["Verified"] == "Yes"
    ok = v & (df["Original_Correct"] == "Yes")
    fix = v & (df["Original_Correct"] == "No") & df["Corrected_X"].notna()
    df.loc[ok, ["status", "status_reason"]] = ["verified_correct", ""]
    df.loc[fix, "status"] = "verified_corrected"
    df.loc[fix, "status_reason"] = df.loc[fix, "How_Corrected"].fillna("")
    df.loc[fix, "out_lat"] = df.loc[fix, "Corrected_Y"]
    df.loc[fix, "out_lon"] = df.loc[fix, "Corrected_X"]
    un = (df["status"] == "pending") & (df["Original_Correct"] == "No")
    df.loc[un, ["status", "status_reason"]] = ["reviewed_unresolved", ""]
    return df


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6_371_008.8 * np.arcsin(np.sqrt(a))


def review_history() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for f in sorted(REVIEW_DIR.glob("review_log_round*.parquet")):
        if not re.fullmatch(r"review_log_round\d+\.parquet", f.name):
            continue
        d = pd.read_parquet(f)
        if "reviewed" in d.columns:
            d = d[d["reviewed"] == 1]
        for r in d.itertuples(index=False):
            out.setdefault(str(r.cwns_id), []).append({
                "round": int(r.review_round),
                "verdict": r.plant_verdict,
                "task": r.review_task,
                "notes": r.reviewer_notes if isinstance(r.reviewer_notes, str) else "",
            })
    return out


def build() -> dict:
    master, layer = load_master()
    if OUTPUT_PARQUET.exists():
        df = pd.read_parquet(OUTPUT_PARQUET)
        df["CWNS_ID"] = df["CWNS_ID"].astype(str)
        source = f"13 output ({OUTPUT_PARQUET.name}, built {df['built'].iloc[0]})" \
            if "built" in df.columns and len(df) else f"13 output ({OUTPUT_PARQUET.name})"
        cutoff = float(df["move_cutoff"].iloc[0]) if "move_cutoff" in df.columns and len(df) else None
    else:
        df = preview_from_master(master)
        source = f"PREVIEW: master layer {layer} only (no 13 output yet)"
        cutoff = None

    info = master[["CWNS_ID", "CITY", "COUNTY_NAME", "OWNER_TYPE", "Has_OSM"]]
    df = df.merge(info, on="CWNS_ID", how="left")
    if "moved_m" not in df.columns:
        df["moved_m"] = haversine_m(df["reported_lat"], df["reported_lon"],
                                    df["out_lat"], df["out_lon"])
    df = df[df["out_lat"].notna() & df["out_lon"].notna()].copy()

    rank = {s: i for i, s in enumerate(STATUS_ORDER)}
    df["_o"] = df["status"].map(rank).fillna(-1)
    df = df.sort_values("_o", kind="stable")

    hist = review_history()

    def num(v, nd=None):
        if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
            return None
        return round(float(v), nd) if nd is not None else float(v)

    def txt(v):
        return None if v is None or pd.isna(v) or v == "" else str(v)

    plants = []
    for r in df.to_dict(orient="records"):
        plants.append({
            "id": r["CWNS_ID"],
            "name": txt(r.get("FACILITY_NAME")),
            "state": txt(r.get("STATE_CODE")),
            "city": txt(r.get("CITY")),
            "county": txt(r.get("COUNTY_NAME")),
            "owner": txt(r.get("OWNER_TYPE")),
            "status": r["status"],
            "reason": txt(r.get("status_reason")),
            "lat": num(r["out_lat"], 6), "lon": num(r["out_lon"], 6),
            "rlat": num(r.get("reported_lat"), 6), "rlon": num(r.get("reported_lon"), 6),
            "moved_m": num(r.get("moved_m"), 0),
            "s1": num(r.get("stage1_prob_correct"), 3),
            "route": txt(r.get("stage1_route")),
            "rr": num(r.get("rerank_score"), 3),
            "fb": None if pd.isna(r.get("rerank_fallback")) else bool(r.get("rerank_fallback")),
            "parcel": txt(r.get("moved_to_ll_uuid")),
            "rparcel": txt(r.get("reported_ll_uuid")),
            "osm": bool(r.get("Has_OSM")) if not pd.isna(r.get("Has_OSM")) else None,
            "pop": num(r.get("pop_served"), 0),
            "tier": txt(r.get("confidence_tier")),
            "coord": txt(r.get("coord_method")),
            "nobj": None if pd.isna(r.get("n_objects")) else int(r.get("n_objects")),
            "site": txt(r.get("site_parcels")),
            "nsite": None if pd.isna(r.get("n_site_parcels")) else int(r.get("n_site_parcels")),
            "insite": None if pd.isna(r.get("coord_in_site")) else bool(r.get("coord_in_site")),
            "reviews": hist.get(r["CWNS_ID"], []),
        })

    counts = df["status"].value_counts().to_dict()
    return {"source": source, "cutoff": cutoff, "counts": counts, "plants": plants}


def build_sites() -> dict:
    """Site outlines (13's sites layer) as GeoJSON; empty when absent."""
    if not SITES_PARQUET.exists():
        return {"type": "FeatureCollection", "features": []}
    import geopandas as gpd
    g = gpd.read_parquet(SITES_PARQUET)
    g = g[["CWNS_ID", "status", "n_parcels", "geometry"]]
    return json.loads(g.to_json(drop_id=True))


def build_detections() -> list[dict]:
    """One record per detected object at each plant's final site."""
    if not DETECTIONS_PARQUET.exists():
        return []
    d = pd.read_parquet(DETECTIONS_PARQUET)
    return [{"id": r.CWNS_ID, "cls": r.class_name, "conf": round(float(r.confidence), 3),
             "src": r.source, "used": bool(r.used_for_coord),
             "lon": round(float(r.lon), 6), "lat": round(float(r.lat), 6)}
            for r in d.itertuples(index=False)]


if __name__ == "__main__":
    d = build()
    print(d["source"])
    for s in reversed(STATUS_ORDER):
        if s in d["counts"]:
            print(f"  {s:<22} {d['counts'][s]:>7,}")
    print(f"  {'TOTAL':<22} {len(d['plants']):>7,}")
    print(f"  JSON size ~{len(json.dumps(d)) / 1e6:.1f} MB")
    print(f"  sites: {len(build_sites()['features']):,}  detections: {len(build_detections()):,}")
