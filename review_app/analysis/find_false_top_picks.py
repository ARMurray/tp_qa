"""
find_false_top_picks.py
=========================
Pulls every reviewed plant where the model's TOP-RANKED candidate (rank 1)
was NOT the correct site -- the reviewer picked a lower-ranked candidate
instead. Rank 1's parcel is exactly the kind of site that's confusing the
object detection model (per 2026-08-28's review findings: baseball fields,
cul-de-sacs, and similar false-detect sites that scored HIGHER than the
actual treatment plant). These are the sites worth adding as explicit hard
negatives to the next OD annotation round -- see
detection/pipeline/01_sample_sites.py, which currently only draws hard
negatives from TRI facilities and has no "visually confusable" category at
all.

Also flags reported_correct verdicts where n_candidates > 0 -- these mean
the model ranked something above the ACTUAL reported location, so rank 1's
site is a false positive here too, even though there's no "correct rank" to
compare against.

Does NOT touch anything on HPC. Runs entirely against the local app.db and
the local Regrid mirror (same parcels.py live lookup the review app itself
uses) -- both already on this machine.

Usage:
    python find_false_top_picks.py
    python find_false_top_picks.py --out false_top_picks.csv
    python find_false_top_picks.py --min-rank-gap 2   # only cases where the
                                                        # correct candidate was
                                                        # at least this much
                                                        # lower-ranked than 1
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from backend import parcels, facilities


def load_false_top_picks(min_rank_gap: int) -> pd.DataFrame:
    conn = sqlite3.connect(C.APP_DB_PATH)
    conn.row_factory = sqlite3.Row

    # Case 1: candidate_correct, but NOT rank 1 -- the model's top guess was
    # wrong, and we know exactly which parcel it wrongly favored (rank 1's
    # own ll_uuid, looked up separately below).
    case1 = pd.read_sql_query("""
        SELECT p.cwns_id, p.state_code, p.plant_verdict,
               p.candidate_rank AS correct_rank, p.reviewer, p.reviewed_at,
               p.reviewer_notes, p.latitude AS reported_lat, p.longitude AS reported_lon,
               c1.ll_uuid AS rank1_ll_uuid, c1.stage2a_score AS rank1_score,
               c_correct.ll_uuid AS correct_ll_uuid, c_correct.stage2a_score AS correct_score
        FROM plants p
        JOIN candidates c1 ON c1.cwns_id = p.cwns_id AND c1.candidate_rank = 1
        JOIN candidates c_correct ON c_correct.cwns_id = p.cwns_id
             AND c_correct.candidate_rank = p.candidate_rank
        WHERE p.reviewed = 1
          AND p.plant_verdict = 'candidate_correct'
          AND p.candidate_rank >= 1 + ?
    """, conn, params=[min_rank_gap])
    case1["case"] = "wrong_top_pick"

    # Case 2: reported_correct, but the model generated candidates anyway
    # (i.e. Stage 1 flagged it, Stage 2a ranked something above the actual
    # correct answer, and the reviewer confirmed the REPORTED point was
    # right all along). Rank 1 here is a pure false positive with nothing
    # to its credit at all.
    case2 = pd.read_sql_query("""
        SELECT p.cwns_id, p.state_code, p.plant_verdict,
               NULL AS correct_rank, p.reviewer, p.reviewed_at, p.reviewer_notes,
               p.latitude AS reported_lat, p.longitude AS reported_lon,
               c1.ll_uuid AS rank1_ll_uuid, c1.stage2a_score AS rank1_score,
               NULL AS correct_ll_uuid, NULL AS correct_score
        FROM plants p
        JOIN candidates c1 ON c1.cwns_id = p.cwns_id AND c1.candidate_rank = 1
        WHERE p.reviewed = 1 AND p.plant_verdict = 'reported_correct'
    """, conn)
    case2["case"] = "reported_beat_top_pick"

    conn.close()
    combined = pd.concat([case1, case2], ignore_index=True)
    combined["facility_name"] = combined["cwns_id"].map(
        lambda c: facilities.get_facility_info(c)["facility_name"])
    return combined


def enrich_with_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Adds rank-1 parcel centroid lat/lon + owner/LBCS context, live from
    the local Regrid mirror -- same source the review app itself reads."""
    out_rows = []
    for state, grp in df.groupby("state_code"):
        context = parcels.get_parcel_context(state, grp["rank1_ll_uuid"].dropna().unique().tolist())
        for _, row in grp.iterrows():
            entry = context.get(row["rank1_ll_uuid"], {})
            geom = entry.get("geometry")
            lat = lon = None
            if geom and geom.get("type") == "Point":
                lon, lat = geom["coordinates"]
            elif geom:
                # polygon -- use a rough bbox-center fallback; good enough for
                # "go look at this general area" site selection, not for
                # anything requiring precision.
                coords = _flatten_coords(geom)
                if coords:
                    lon = sum(c[0] for c in coords) / len(coords)
                    lat = sum(c[1] for c in coords) / len(coords)
            row = dict(row)
            row["rank1_lat"] = lat
            row["rank1_lon"] = lon
            row["rank1_owner"] = entry.get("owner")
            row["rank1_lbcs_activity_desc"] = entry.get("lbcs_activity_desc")
            row["rank1_ll_gisacre"] = entry.get("ll_gisacre")
            out_rows.append(row)
    return pd.DataFrame(out_rows)


def _flatten_coords(geom):
    """Crude coordinate flattener across Polygon/MultiPolygon GeoJSON, for
    the bbox-center fallback above only -- not a real centroid."""
    coords = []
    def walk(x):
        if isinstance(x, (list, tuple)):
            if len(x) == 2 and all(isinstance(v, (int, float)) for v in x):
                coords.append(x)
            else:
                for item in x:
                    walk(item)
    walk(geom.get("coordinates", []))
    return coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="false_top_picks.csv")
    ap.add_argument("--min-rank-gap", type=int, default=1,
                     help="only include candidate_correct cases where the correct "
                          "candidate's rank is at least this much worse than 1 "
                          "(default 1 = any non-top-1 correct pick)")
    args = ap.parse_args()

    print("=== find_false_top_picks.py ===")
    df = load_false_top_picks(args.min_rank_gap)
    print(f"Found {len(df)} false-top-pick case(s): "
          f"{int((df['case']=='wrong_top_pick').sum())} wrong_top_pick, "
          f"{int((df['case']=='reported_beat_top_pick').sum())} reported_beat_top_pick")

    if df.empty:
        print("Nothing to write.")
        return

    df = enrich_with_geometry(df)
    df = df.sort_values(["state_code", "cwns_id"])

    cols = ["cwns_id", "facility_name", "state_code", "case", "correct_rank",
            "rank1_score", "correct_score", "rank1_lat", "rank1_lon",
            "rank1_owner", "rank1_lbcs_activity_desc", "rank1_ll_gisacre",
            "reported_lat", "reported_lon", "rank1_ll_uuid", "correct_ll_uuid",
            "reviewer", "reviewed_at", "reviewer_notes"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(args.out, index=False)
    print(f"\nWritten: {args.out}")
    print(f"\nEach row's rank1_lat/rank1_lon is the site that OUTSCORED the "
          f"correct answer -- these are your candidate hard-negative locations "
          f"for the next OD annotation round.")


if __name__ == "__main__":
    main()
