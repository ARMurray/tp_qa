"""
08_diagnose_candidate_coverage.py
===================================
Answers the questions that gate every Stage 2b data decision:

  Q1. What is the k=18 search window's ACTUAL radius in km?
      config.py calls it "~10km", but H3 res-9 cells are ~174 m on edge, so
      adjacent cell centers sit ~302 m apart and k=18 is a radius of roughly
      5.4 km -- the ~10 km figure looks like the DIAMETER. This script
      measures it empirically instead of arguing from geometry: for every
      correction it computes both the true reported->corrected distance and
      the H3 grid distance, and reports the observed meters-per-ring.

  Q2. For what fraction of corrections does the candidate pool actually
      CONTAIN the true parcel? This is the hard ceiling on the entire
      pipeline -- no amount of Stage 2a/2b tuning can recover a plant whose
      right answer was never a candidate.

  Q3. WHY is each missing one missing? Four distinct causes with four
      different fixes, currently all collapsed into "46 dropped":
        (a) outside the ring          -> raise K_RINGS (expensive, k^2)
        (b) no parcel at that point   -> Regrid coverage gap, unfixable here
        (c) parcel exists but not in 01a's NLCD output -> 01d top-up
        (d) in NLCD output but absent from 10_parcel_features -> join loss
                                          in 02 PART 2 (a real bug, cheap fix)

  Q4. The mirror question on the REPORTED side: 06 produced only 162 negative
      rows against 266 positives, so ~130 plants contribute no negative at
      all. Is that because the reported point hits no parcel (expected and
      fine -- those plants are Stage 1's job, not Stage 2b's), or because
      the reported parcel exists and got lost in a join (a bug worth fixing,
      and ~130 free hard negatives if so)?

Note on Q1/Q2: 01a builds its H3 window as the UNION of grid_disks around
EVERY plant in the state, so a corrected parcel further than k rings from
its OWN plant can still be captured if it happens to sit near some other
plant. This script reports both measures separately -- "own-plant ring
distance" and "inside the state union window" -- because they imply
different fixes.

Writes:
    data/diagnostics/correction_coverage.parquet   one row per correction
    data/diagnostics/missing_parcels_topup.csv     state,ll_uuid -- feeds 01d

Usage:
    python 08_diagnose_candidate_coverage.py [--states OH,PA]
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import h3
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

# ---- Reuse 01a's plant loader / window builder rather than re-deriving them
# (filename starts with a digit, so importlib -- same pattern as 01c). ----
_spec = importlib.util.spec_from_file_location(
    "extract_lib", Path(__file__).resolve().parent / "01a_extract_parcels.py")
extract_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract_lib)


def lookup_ll_uuid(con, state: str, cwns_ids, lons, lats, label: str) -> pd.DataFrame:
    """Point-in-parcel lookup. Identical pattern to 06's, kept local so this
    diagnostic doesn't depend on 06 having been run."""
    pts = pd.DataFrame({"CWNS_ID": list(cwns_ids), "LON": list(lons), "LAT": list(lats)})
    con.register("pts", pts)
    try:
        result = con.execute(f"""
            SELECT pts.CWNS_ID, p.{C.PARCEL_ID_FIELD} AS ll_uuid
            FROM read_parquet('{C.PARCEL_BASE.as_posix()}/state={state}/*.parquet') p
            JOIN pts ON ST_Intersects(
                ST_GeomFromWKB(p.{C.PARCEL_WKB_FIELD}),
                ST_Point(pts.LON, pts.LAT)
            )
        """).df()
    except Exception as e:
        print(f"    [{state}] {label} point-in-parcel lookup failed: {e}")
        return pd.DataFrame(columns=["CWNS_ID", "ll_uuid"])
    finally:
        con.unregister("pts")
    if not len(result):
        return pd.DataFrame(columns=["CWNS_ID", "ll_uuid"])
    result["CWNS_ID"] = result["CWNS_ID"].astype(str)
    return result.drop_duplicates(subset="CWNS_ID", keep="first")


def h3_cell(lat, lon):
    try:
        return h3.latlng_to_cell(float(lat), float(lon), 9)
    except Exception:
        return None


def grid_distance_safe(a, b):
    """h3.grid_distance raises across icosahedron face boundaries for
    very long distances -- return NaN rather than killing the run."""
    if a is None or b is None:
        return np.nan
    try:
        return h3.grid_distance(a, b)
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=str, default=None,
                    help="comma-separated STATE_CODE filter (default: all)")
    ap.add_argument("--k-rings", type=int, default=C.K_RINGS)
    args = ap.parse_args()
    states_filter = [s.strip() for s in args.states.split(",")] if args.states else None

    C.ensure_dirs()
    diag_dir = C.DATA_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    print("=== 08_diagnose_candidate_coverage.py ===")
    print(f"K rings: {args.k_rings}\n")

    # ---- Corrections + state codes ----
    corr = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CORRECTIONS)
    corr["CWNS_ID"] = corr["CWNS_ID"].astype(str)
    corr = corr.dropna(subset=["Original_X", "Original_Y", "Corrected_X", "Corrected_Y"])
    loc = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt",
                      dtype={"CWNS_ID": str}, encoding="latin1")
    loc = loc[["CWNS_ID", "STATE_CODE"]].drop_duplicates(subset="CWNS_ID")
    corr = corr.merge(loc, on="CWNS_ID", how="left").dropna(subset=["STATE_CODE"])
    if states_filter:
        corr = corr[corr["STATE_CODE"].isin(states_filter)]
    corr = corr[["CWNS_ID", "STATE_CODE", "Original_X", "Original_Y",
                 "Corrected_X", "Corrected_Y"]].reset_index(drop=True)
    print(f"Corrections with both coordinate pairs: {len(corr)}\n")

    # ---- Q1: distance + H3 ring distance ----
    rep_pts = gpd.GeoSeries(gpd.points_from_xy(corr["Original_X"], corr["Original_Y"]),
                            crs=C.EXPORT_CRS).to_crs(C.PROJECTED_CRS)
    cor_pts = gpd.GeoSeries(gpd.points_from_xy(corr["Corrected_X"], corr["Corrected_Y"]),
                            crs=C.EXPORT_CRS).to_crs(C.PROJECTED_CRS)
    corr["distance_m"] = rep_pts.distance(cor_pts).to_numpy()
    corr["reported_h3"] = [h3_cell(y, x) for x, y in
                           zip(corr["Original_X"], corr["Original_Y"])]
    corr["corrected_h3"] = [h3_cell(y, x) for x, y in
                            zip(corr["Corrected_X"], corr["Corrected_Y"])]
    corr["ring_distance"] = [grid_distance_safe(a, b) for a, b in
                             zip(corr["reported_h3"], corr["corrected_h3"])]
    corr["within_own_ring"] = corr["ring_distance"] <= args.k_rings

    d = corr["distance_m"] / 1000.0
    print("--- Q1: correction distance distribution (km) ---")
    print(f"  n={len(d)}  mean={d.mean():.2f}  median={d.median():.2f}  "
          f"p90={d.quantile(0.90):.2f}  p95={d.quantile(0.95):.2f}  max={d.max():.2f}")
    ok = corr["ring_distance"].notna() & (corr["ring_distance"] > 0)
    if ok.any():
        m_per_ring = (corr.loc[ok, "distance_m"] / corr.loc[ok, "ring_distance"]).median()
        print(f"  Observed median meters per H3 ring step: {m_per_ring:.0f} m")
        print(f"  => effective k={args.k_rings} search RADIUS is about "
              f"{m_per_ring * args.k_rings / 1000:.1f} km")
        print(f"     (config.py's comment says '~10km' -- if the number above is "
              f"~5.4, that comment is describing the diameter.)")
    print(f"  Corrections beyond their own k={args.k_rings} ring: "
          f"{(~corr['within_own_ring']).sum()} / {len(corr)} "
          f"({100 * (~corr['within_own_ring']).mean():.1f}%)")
    print()

    # ---- Per-state lookups ----
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; SET enable_geoparquet_conversion = false;")

    rep_ll, cor_ll, window_rows = [], [], []
    all_plants = extract_lib.load_treatment_plants(states_filter, training_only=True)

    for state, grp in corr.groupby("STATE_CODE"):
        print(f"--- {state}: {len(grp)} corrections ---")
        rep_ll.append(lookup_ll_uuid(con, state, grp["CWNS_ID"],
                                     grp["Original_X"], grp["Original_Y"], "reported")
                      .rename(columns={"ll_uuid": "reported_ll_uuid"}))
        cor_ll.append(lookup_ll_uuid(con, state, grp["CWNS_ID"],
                                     grp["Corrected_X"], grp["Corrected_Y"], "corrected")
                      .rename(columns={"ll_uuid": "corrected_ll_uuid"}))

        # Union window across ALL of this state's training plants -- the
        # thing 01a actually queried, which can rescue a far-away parcel.
        plants_state = all_plants[all_plants["STATE_CODE"] == state]
        if len(plants_state):
            window = extract_lib.build_h3_search_window(plants_state, args.k_rings)
            window_rows.append(pd.DataFrame({
                "CWNS_ID": grp["CWNS_ID"].to_numpy(),
                "in_state_window": [c in window for c in grp["corrected_h3"]],
            }))

    con.close()

    def cat(frames, cols):
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)

    corr = corr.merge(cat(rep_ll, ["CWNS_ID", "reported_ll_uuid"]), on="CWNS_ID", how="left")
    corr = corr.merge(cat(cor_ll, ["CWNS_ID", "corrected_ll_uuid"]), on="CWNS_ID", how="left")
    corr = corr.merge(cat(window_rows, ["CWNS_ID", "in_state_window"]), on="CWNS_ID", how="left")

    # ---- Membership in 01a's NLCD output and 02's parcel features ----
    nlcd_uuids = set()
    for state in corr["STATE_CODE"].unique():
        p = C.NLCD_OUTPUT_DIR / f"nlcd_{state}_k{args.k_rings}.parquet"
        if p.exists():
            nlcd_uuids |= set(pd.read_parquet(p, columns=["ll_uuid"])["ll_uuid"])
        else:
            print(f"  WARNING: {p.name} not found -- {state} will look 100% missing")

    pf_path = C.FEATURES_OUTPUT_DIR / "10_parcel_features.parquet"
    pf_uuids = set(pd.read_parquet(pf_path, columns=["ll_uuid"])["ll_uuid"]) \
        if pf_path.exists() else set()
    if not pf_uuids:
        print(f"  WARNING: {pf_path.name} not found -- run 02 first for the join-loss check")

    corr["corrected_in_nlcd"] = corr["corrected_ll_uuid"].isin(nlcd_uuids)
    corr["corrected_in_parcel_features"] = corr["corrected_ll_uuid"].isin(pf_uuids)
    corr["reported_in_parcel_features"] = corr["reported_ll_uuid"].isin(pf_uuids)

    # ---- Q3: classify every correction ----
    def classify(r):
        if pd.isna(r["corrected_ll_uuid"]):
            return "b_no_parcel_at_corrected_point"
        if r["corrected_in_parcel_features"]:
            return "ok_usable"
        if r["corrected_in_nlcd"]:
            return "d_lost_in_02_join"
        if not r["within_own_ring"] and not bool(r.get("in_state_window", False)):
            return "a_outside_search_window"
        return "c_in_window_but_no_nlcd_row"

    corr["status"] = corr.apply(classify, axis=1)

    print("\n--- Q2/Q3: corrected-parcel coverage ---")
    counts = corr["status"].value_counts()
    for k, v in counts.items():
        print(f"  {k:34s} {v:5d}  ({100 * v / len(corr):5.1f}%)")
    usable = counts.get("ok_usable", 0)
    print(f"\n  CEILING: {usable}/{len(corr)} ({100 * usable / len(corr):.1f}%) of known "
          f"corrections can currently contribute a positive Stage 2b row.")

    outside = corr[corr["status"] == "a_outside_search_window"]
    if len(outside):
        need_k = int(np.nanpercentile(outside["ring_distance"], 95))
        print(f"  Of the {len(outside)} outside the window, ring distances run "
              f"{outside['ring_distance'].min():.0f}-{outside['ring_distance'].max():.0f}; "
              f"k={need_k} would capture 95% of them "
              f"(candidate count scales ~k^2, so k={need_k} is roughly "
              f"{(need_k / args.k_rings) ** 2:.1f}x the current pool).")

    # ---- Q4: the reported/negative side ----
    print("\n--- Q4: reported-side (negative row) availability ---")
    n_no_parcel = corr["reported_ll_uuid"].isna().sum()
    n_lost = ((~corr["reported_ll_uuid"].isna()) &
              (~corr["reported_in_parcel_features"])).sum()
    n_ok = corr["reported_in_parcel_features"].sum()
    print(f"  reported point hits NO parcel        : {n_no_parcel}  "
          f"(expected -- these plants legitimately have no negative row)")
    print(f"  reported parcel exists but not in 10_parcel_features: {n_lost}  "
          f"(join loss -- these are FREE hard negatives if recovered)")
    print(f"  usable negative rows                 : {n_ok}")

    # ---- Outputs ----
    out_path = diag_dir / "correction_coverage.parquet"
    corr.drop(columns=["reported_h3", "corrected_h3"]).to_parquet(out_path, index=False)
    print(f"\nWritten: {out_path}")

    topup = corr[corr["status"].isin(["c_in_window_but_no_nlcd_row", "d_lost_in_02_join"])]
    topup_rows = topup[["STATE_CODE", "corrected_ll_uuid"]].rename(
        columns={"STATE_CODE": "state", "corrected_ll_uuid": "ll_uuid"}).dropna()
    # Recoverable reported-side parcels belong in the top-up too.
    rep_topup = corr[(~corr["reported_ll_uuid"].isna()) &
                     (~corr["reported_in_parcel_features"])]
    topup_rows = pd.concat([topup_rows, rep_topup[["STATE_CODE", "reported_ll_uuid"]].rename(
        columns={"STATE_CODE": "state", "reported_ll_uuid": "ll_uuid"})],
        ignore_index=True).drop_duplicates()

    topup_path = diag_dir / "missing_parcels_topup.csv"
    topup_rows.to_csv(topup_path, index=False)
    print(f"Written: {topup_path}  ({len(topup_rows)} parcels for 01d)")
    print("\nNOTE: parcels classified 'a_outside_search_window' are NOT in that "
          "file -- a top-up can compute their NLCD stats, but they would still "
          "be absent from Stage 2a's candidate pool at inference, so the honest "
          "fix is K_RINGS, not a top-up. Decide that from the numbers above.")

    print("\n=== complete ===")


if __name__ == "__main__":
    main()
