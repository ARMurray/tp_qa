"""
13_build_corrected_output.py
============================
The product: one row for EVERY CWNS treatment plant, saying where it is and
how we know.

STATUS (first match wins)
    verified_correct      a reviewer confirmed the reported location (master)
    verified_corrected    a reviewer supplied a corrected location (master)
    reviewed_unresolved   the master says the reported location is wrong but
                          holds no corrected one -- kept, flagged
    moved                 Stage 1 flagged the reported location AND the
                          re-ranker's #1 parcel scored >= --cutoff; the new
                          point is a point INSIDE that parcel
    flagged_not_moved     Stage 1 flagged it but no candidate cleared the
                          cutoff -- reported location kept, flagged as doubtful
    kept_osm              reported parcel carries an OSM wastewater tag
                          (passed by rule, stage1_route = osm_confirmed)
    kept_model            Stage 1 scored the reported location as correct
    not_assessed          outside the model's scope; reason in status_reason
                          (no NAIP state, population <= 1,000 or unknown,
                          no coordinates)

Human verification always wins: the master (Updates.gpkg, newest
CWNS_Locations layer) is read first and a verified plant is never moved.

THE MOVE RULE IS A PARAMETER, NOT A CONSTANT
    --cutoff is required. It must come from out-of-sample review precision
    (calibrate_move_rule.py), and choosing it is the project owner's call.
    --require-detection additionally refuses moves where nothing fired in
    the pool (rerank_fallback: the #1 is Stage 2a's).

REFUSES to run when the re-ranked candidates mostly lack current-detector
results (national 01e not run): every move would then rest on Stage 2a
ordering alone. --allow-low-od overrides, for testing only.

Outputs (data/output/):
    cwns_corrected_locations.parquet / .csv / .gpkg (layer 'plants')

Usage:
    python 13_build_corrected_output.py --cutoff 0.95
    python 13_build_corrected_output.py --cutoff 0.95 --require-detection
"""
import argparse
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

INFER_DIR = C.DATA_DIR / "inference"
OUT_DIR = C.DATA_DIR / "output"
MIN_OD_COVERAGE = 0.5


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
                                            "rerank_rank", "od_ran"])
    cand["CWNS_ID"] = cand["CWNS_ID"].astype(str)
    return summ, cand


def parcel_points(moves: pd.DataFrame) -> pd.DataFrame:
    """A point guaranteed INSIDE each moved-to parcel (shapely
    representative_point, not the centroid, which can fall outside an
    L-shaped or multi-part parcel). Parcel WKB is EPSG:4326."""
    from shapely import wkb
    con = C.duckdb_connect()
    rows = []
    for state, grp in moves.groupby("STATE_CODE"):
        con.register("want", pd.DataFrame({"ll_uuid": grp["ll_uuid"].unique()}))
        try:
            res = con.execute(f"""
                SELECT p.{C.PARCEL_ID_FIELD} AS ll_uuid,
                       p.{C.PARCEL_WKB_FIELD} AS parcel_wkb
                FROM read_parquet('{C.PARCEL_BASE.as_posix()}/state={state}/*.parquet') p
                JOIN want ON want.ll_uuid = p.{C.PARCEL_ID_FIELD}
            """).df().drop_duplicates(subset="ll_uuid")
        finally:
            con.unregister("want")
        for _, r in res.iterrows():
            pt = wkb.loads(bytes(r["parcel_wkb"])).representative_point()
            rows.append((r["ll_uuid"], pt.y, pt.x))
    out = pd.DataFrame(rows, columns=["ll_uuid", "new_lat", "new_lon"])
    lost = set(moves["ll_uuid"]) - set(out["ll_uuid"])
    if lost:
        print(f"  WARNING: {len(lost)} moved-to parcel(s) not found in the parcel "
              f"mirror -- those plants fall back to flagged_not_moved")
    return out


def main():
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

    df = master[["CWNS_ID", "STATE_CODE", "FACILITY_NAME", "LATITUDE", "LONGITUDE",
                 "Original_Correct", "Verified", "Corrected_X", "Corrected_Y",
                 "How_Corrected"]].rename(columns={"LATITUDE": "reported_lat",
                                                   "LONGITUDE": "reported_lon"})
    keep = ["CWNS_ID", "stage1_route", "trigger_reason", "stage1_prob_correct",
            "reported_ll_uuid", "rerank_top_ll_uuid", "rerank_score",
            "rerank_fallback", "rerank_top_stage2a_rank"]
    df = df.merge(summ[[c for c in keep if c in summ.columns]], on="CWNS_ID", how="left")

    df["status"] = pd.NA
    df["status_reason"] = ""
    df["out_lat"] = df["reported_lat"]
    df["out_lon"] = df["reported_lon"]
    df["moved_to_ll_uuid"] = pd.NA

    verified = df["Verified"] == "Yes"
    v_ok = verified & (df["Original_Correct"] == "Yes")
    v_fix = verified & (df["Original_Correct"] == "No") & df["Corrected_X"].notna()
    df.loc[v_ok, "status"] = "verified_correct"
    df.loc[v_fix, "status"] = "verified_corrected"
    df.loc[v_fix, "out_lat"] = df.loc[v_fix, "Corrected_Y"]
    df.loc[v_fix, "out_lon"] = df.loc[v_fix, "Corrected_X"]
    df.loc[v_fix, "status_reason"] = df.loc[v_fix, "How_Corrected"].fillna("")
    unresolved = df["status"].isna() & (df["Original_Correct"] == "No")
    df.loc[unresolved, "status"] = "reviewed_unresolved"

    open_ = df["status"].isna()
    in_scope = df["stage1_route"].notna()
    na = open_ & ~in_scope
    df.loc[na, "status"] = "not_assessed"
    df.loc[na & df["reported_lat"].isna(), "status_reason"] = "no_coordinates"
    df.loc[na & (df["status_reason"] == "") & ~df["STATE_CODE"].isin(scored_states),
           "status_reason"] = "no_imagery_state"
    df.loc[na & (df["status_reason"] == "") & ~df["CWNS_ID"].isin(pop_ok),
           "status_reason"] = "population_le_1000_or_unknown"
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

    moves = df.loc[can_move, ["CWNS_ID", "STATE_CODE", "rerank_top_ll_uuid"]] \
              .rename(columns={"rerank_top_ll_uuid": "ll_uuid"})
    print(f"\n  flagged plants: {int(flagged.sum()):,}  |  clearing the move rule: "
          f"{len(moves):,}")
    if len(moves):
        pts = parcel_points(moves)
        lost = ~moves["ll_uuid"].isin(pts["ll_uuid"])
        df.loc[df["CWNS_ID"].isin(moves.loc[lost, "CWNS_ID"]),
               "status_reason"] = "move_parcel_not_found"
        moves = moves.merge(pts, on="ll_uuid", how="inner")
        idx = df["CWNS_ID"].isin(moves["CWNS_ID"])
        m = df.loc[idx, ["CWNS_ID"]].merge(moves, on="CWNS_ID", how="left")
        df.loc[idx, "out_lat"] = m["new_lat"].to_numpy()
        df.loc[idx, "out_lon"] = m["new_lon"].to_numpy()
        df.loc[idx, "moved_to_ll_uuid"] = m["ll_uuid"].to_numpy()
        df.loc[idx, "status"] = "moved"

    leftover = df["status"].isna()
    if leftover.any():
        df.loc[leftover, "status"] = "not_assessed"
        df.loc[leftover, "status_reason"] = "unrouted"
        print(f"  WARNING: {int(leftover.sum())} plant(s) matched no rule (unrouted)")

    df["moved_m"] = np.where(
        df["status"].isin(["moved", "verified_corrected"]),
        haversine_m(df["reported_lat"], df["reported_lon"], df["out_lat"], df["out_lon"]),
        0.0)
    df["move_cutoff"] = args.cutoff
    df["require_detection"] = args.require_detection
    df["built"] = date.today().isoformat()

    cols = ["CWNS_ID", "STATE_CODE", "FACILITY_NAME", "status", "status_reason",
            "reported_lat", "reported_lon", "out_lat", "out_lon", "moved_m",
            "moved_to_ll_uuid", "reported_ll_uuid", "stage1_route",
            "stage1_prob_correct", "rerank_score", "rerank_fallback",
            "move_cutoff", "require_detection", "built"]
    out = df[cols].sort_values(["STATE_CODE", "CWNS_ID"]).reset_index(drop=True)

    print("\n=== status ===")
    vc = out["status"].value_counts()
    for s, n in vc.items():
        print(f"  {s:<22} {n:>7,}  ({n / len(out):.1%})")
    print(f"  {'TOTAL':<22} {len(out):>7,}")
    na_r = out.loc[out["status"] == "not_assessed", "status_reason"].value_counts()
    if len(na_r):
        print("  not_assessed by reason: " + ", ".join(f"{k} {v:,}" for k, v in na_r.items()))
    mv = out.loc[out["status"] == "moved", "moved_m"]
    if len(mv):
        print(f"  moved distance: median {mv.median():,.0f} m, "
              f"90th pct {mv.quantile(0.9):,.0f} m, max {mv.max():,.0f} m")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / "cwns_corrected_locations"
    out.to_parquet(base.with_suffix(".parquet"), index=False)
    out.to_csv(base.with_suffix(".csv"), index=False)
    import geopandas as gpd
    gdf = gpd.GeoDataFrame(out, geometry=gpd.points_from_xy(out["out_lon"], out["out_lat"]),
                           crs=4326)
    gdf.to_file(base.with_suffix(".gpkg"), layer="plants", driver="GPKG")
    print(f"\nWritten: {base}.parquet / .csv / .gpkg")


if __name__ == "__main__":
    main()
