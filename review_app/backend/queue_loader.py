"""
queue_loader.py
================
Imports data/incoming/review_queue_round{N}.parquet (synced down from HPC,
written by 10_build_review_queue.py) into the local SQLite store.

Idempotent per round: re-running for a round you've already loaded will
refuse rather than duplicate or silently overwrite in-progress reviews.

Two ways to re-run against an already-loaded round:
  --force            wipes and reloads EVERYTHING for that round, including
                      any submitted verdicts. Destructive -- only for
                      genuinely starting a round over.
  --update-metadata   backfills descriptive columns (address/city/pop_served/
                      owner/lbcs_*/etc, added 2026-08-26) WITHOUT touching
                      reviewed/plant_verdict/selected_ll_uuid/etc on plants
                      already reviewed. Use this to pick up a queue-builder
                      change that only added columns -- the plant SELECTION
                      itself is unchanged (same --seed, same rows), only
                      what's known ABOUT each plant grew.

Usage:
    python -m backend.queue_loader --round 1
    python -m backend.queue_loader --round 1 --update-metadata
    python -m backend.queue_loader --round 1 --force
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from backend import db

PLANT_COLS = ["CWNS_ID", "STATE_CODE", "LATITUDE", "LONGITUDE",
              "reported_ll_uuid", "stage1_prob_correct", "trigger_reason",
              "review_task", "queue_slice", "model_version",
              "review_round", "is_holdout"]

# Descriptive-only -- added 2026-08-26. Deliberately separate from PLANT_COLS
# above: these are the columns --update-metadata is allowed to touch on an
# already-reviewed plant. Never include reviewed/plant_verdict/etc here.
PLANT_DISPLAY_COLS = ["ADDRESS", "CITY", "COUNTY_NAME", "ZIP_CODE",
                      "pop_served", "subdivision", "place", "county", "is_rural",
                      "surface_water_discharge", "requires_npdes", "any_reuse"]

CAND_COLS = ["CWNS_ID", "candidate_rank", "ll_uuid", "stage2a_score",
             "score_margin", "distance_m", "within_1km", "within_5km",
             "owner", "lbcs_activity", "lbcs_ownership", "lbcs_function",
             "lbcs_structure", "lbcs_site", "ll_gisacre", "ll_bldg_count",
             "dominant_class_group", "has_ww_keyword", "osm_ww",
             "data_quality_score"]


def _insert_plant(conn, r):
    conn.execute("""
        INSERT INTO plants (cwns_id, state_code, latitude, longitude,
            reported_ll_uuid, stage1_prob_correct, trigger_reason,
            review_task, queue_slice, model_version, review_round, is_holdout,
            address, city, county_name, zip_code, pop_served, subdivision,
            place, county, is_rural, surface_water_discharge, requires_npdes,
            any_reuse)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (str(r["CWNS_ID"]), r["STATE_CODE"], r["LATITUDE"], r["LONGITUDE"],
          r["reported_ll_uuid"], r["stage1_prob_correct"], r["trigger_reason"],
          r["review_task"], r["queue_slice"], r["model_version"],
          int(r["review_round"]), int(bool(r["is_holdout"])),
          r.get("ADDRESS"), r.get("CITY"), r.get("COUNTY_NAME"), r.get("ZIP_CODE"),
          r.get("pop_served"), r.get("subdivision"), r.get("place"), r.get("county"),
          _bool_or_none(r.get("is_rural")), _bool_or_none(r.get("surface_water_discharge")),
          _bool_or_none(r.get("requires_npdes")), _bool_or_none(r.get("any_reuse"))))


def _bool_or_none(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return int(bool(v))


def _insert_candidate(conn, r):
    conn.execute("""
        INSERT INTO candidates (cwns_id, candidate_rank, ll_uuid,
            stage2a_score, score_margin, distance_m, within_1km, within_5km,
            owner, lbcs_activity, lbcs_ownership, lbcs_function,
            lbcs_structure, lbcs_site, ll_gisacre, ll_bldg_count,
            dominant_class_group, has_ww_keyword, osm_ww, data_quality_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (str(r["CWNS_ID"]), int(r["candidate_rank"]), r["ll_uuid"],
          r.get("stage2a_score"), r.get("score_margin"), r.get("distance_m"),
          _bool_or_none(r.get("within_1km")), _bool_or_none(r.get("within_5km")),
          r.get("owner"), r.get("lbcs_activity"), r.get("lbcs_ownership"),
          r.get("lbcs_function"), r.get("lbcs_structure"), r.get("lbcs_site"),
          r.get("ll_gisacre"), r.get("ll_bldg_count"), r.get("dominant_class_group"),
          _bool_or_none(r.get("has_ww_keyword")), _bool_or_none(r.get("osm_ww")),
          r.get("data_quality_score")))


def load_round(round_num: int, force: bool = False, update_metadata: bool = False):
    path = C.INCOMING_DIR / f"review_queue_round{round_num}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Sync it down from HPC first -- see sync/pull_round.sh "
            f"-- or copy data/review_queue/review_queue_round{round_num}.parquet "
            f"from HPC into {C.INCOMING_DIR} manually.")

    df = pd.read_parquet(path)
    print(f"Loaded {path.name}: {len(df)} rows, "
          f"{df['CWNS_ID'].nunique()} plants")

    for col in PLANT_DISPLAY_COLS + [c for c in CAND_COLS if c not in
                                     ("CWNS_ID", "candidate_rank", "ll_uuid")]:
        if col not in df.columns:
            df[col] = None

    db.init_db()
    with db.get_conn() as conn:
        existing = conn.execute(
            "SELECT COUNT(*) FROM plants WHERE review_round = ?", (round_num,)
        ).fetchone()[0]

        if existing and not force and not update_metadata:
            raise RuntimeError(
                f"Round {round_num} already has {existing} plant(s) loaded. "
                f"Pass --update-metadata to backfill new descriptive columns "
                f"without touching submitted verdicts, or --force to wipe and "
                f"reload everything (destroys verdicts already submitted).")

        if existing and force:
            print(f"  --force: deleting {existing} existing round {round_num} "
                  f"plant(s) and their candidates first")
            conn.execute("DELETE FROM candidates WHERE cwns_id IN "
                        "(SELECT cwns_id FROM plants WHERE review_round = ?)",
                        (round_num,))
            conn.execute("DELETE FROM plants WHERE review_round = ?", (round_num,))
            existing = 0

        plants = df[PLANT_COLS + PLANT_DISPLAY_COLS].drop_duplicates(subset="CWNS_ID")

        if existing and update_metadata:
            print(f"  --update-metadata: backfilling descriptive columns for "
                  f"{len(plants)} plant(s), leaving reviewed/verdict fields untouched")
            n_updated = 0
            for _, r in plants.iterrows():
                conn.execute("""
                    UPDATE plants SET
                        address = ?, city = ?, county_name = ?, zip_code = ?,
                        pop_served = ?, subdivision = ?, place = ?, county = ?,
                        is_rural = ?, surface_water_discharge = ?,
                        requires_npdes = ?, any_reuse = ?
                    WHERE cwns_id = ?
                """, (r.get("ADDRESS"), r.get("CITY"), r.get("COUNTY_NAME"), r.get("ZIP_CODE"),
                      r.get("pop_served"), r.get("subdivision"), r.get("place"), r.get("county"),
                      _bool_or_none(r.get("is_rural")), _bool_or_none(r.get("surface_water_discharge")),
                      _bool_or_none(r.get("requires_npdes")), _bool_or_none(r.get("any_reuse")),
                      str(r["CWNS_ID"])))
                n_updated += 1
            # Candidates carry no verdict state -- safe to fully replace.
            conn.execute("DELETE FROM candidates WHERE cwns_id IN "
                        "(SELECT cwns_id FROM plants WHERE review_round = ?)",
                        (round_num,))
            n_cands = 0
            cand_rows = df[df["review_task"] == "candidate_pick"]
            for _, r in cand_rows.iterrows():
                if pd.isna(r["candidate_rank"]):
                    continue
                _insert_candidate(conn, r)
                n_cands += 1
            print(f"Updated {n_updated} plant(s), reloaded {n_cands} candidate row(s)")
        else:
            n_plants = 0
            for _, r in plants.iterrows():
                _insert_plant(conn, r)
                n_plants += 1

            cand_rows = df[df["review_task"] == "candidate_pick"]
            n_cands = 0
            for _, r in cand_rows.iterrows():
                if pd.isna(r["candidate_rank"]):
                    continue
                _insert_candidate(conn, r)
                n_cands += 1
            print(f"Imported {n_plants} plant(s), {n_cands} candidate row(s)")

    print(f"\nBy slice:")
    with db.get_conn() as conn:
        for row in db.counts_by_slice(conn):
            print(f"  {row['queue_slice']:<10} {row['total']} total, "
                  f"{row['reviewed']} already reviewed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--update-metadata", action="store_true")
    args = ap.parse_args()
    if args.force and args.update_metadata:
        raise SystemExit("--force and --update-metadata are mutually exclusive")
    load_round(args.round, force=args.force, update_metadata=args.update_metadata)


if __name__ == "__main__":
    main()
