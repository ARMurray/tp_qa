"""
push_review_log.py
===================
Exports reviewed verdicts from app.db to parquet, split per
REVIEW_LOOP_PLAN.md Phase 4 requirement #3 (holdout routing):

    outgoing/review_log_round{N}.parquet
        Non-holdout PLANT-level verdicts. One row per reviewed plant.

    outgoing/review_log_candidates_round{N}.parquet
        Non-holdout CANDIDATE-level rows -- every candidate shown to the
        reviewer for candidate_pick plants, not just the one selected.
        REQUIRED for 11_ingest_review_log.py: per REVIEW_LOOP_PLAN.md Phase 5,
        Stage 2b training needs the selected candidate as label=1 and every
        OTHER shown candidate as label=0 -- the rejected candidates are the
        distribution-matched hard negatives the whole review loop exists to
        produce. Without this file there is nothing to build negatives from.
        (Missing entirely from an earlier version of this script -- confirmed
        2026-08-26 while writing 11_ingest_review_log.py and realizing the
        plant-level export alone couldn't supply what it needed.)

    outgoing/holdout_truth_round{N}.parquet
        Holdout verdicts. Needs manual merging into holdout_truth.parquet on
        HPC. NOT fed into training, never touched by 11_ingest_review_log.py.

Only exports plants reviewed since the last export for that round (tracked
via an export-log table), so re-running mid-session doesn't re-export
everything -- safe to run repeatedly as you review.

Usage:
    python -m sync.push_review_log --round 1
    python -m sync.push_review_log --round 1 --all    # re-export everything,
                                                        # not just new since last push
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from backend import db, parcels

EXPORT_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS export_log (
    cwns_id      TEXT NOT NULL,
    review_round INTEGER NOT NULL,
    exported_at  TEXT NOT NULL,
    PRIMARY KEY (cwns_id, review_round)
);
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--all", action="store_true",
                     help="re-export every reviewed plant for this round, "
                          "not just ones not yet exported")
    args = ap.parse_args()

    with db.get_conn() as conn:
        conn.executescript(EXPORT_LOG_SCHEMA)

        query = """
            SELECT * FROM plants
            WHERE review_round = ? AND reviewed = 1
        """
        params = [args.round]
        if not args.all:
            query += " AND cwns_id NOT IN (SELECT cwns_id FROM export_log WHERE review_round = ?)"
            params.append(args.round)

        df = pd.read_sql_query(query, conn, params=params)

        cwns_ids = df["cwns_id"].tolist()
        cand_df = pd.DataFrame()
        if cwns_ids:
            placeholders = ",".join("?" * len(cwns_ids))
            cand_df = pd.read_sql_query(
                f"SELECT * FROM candidates WHERE cwns_id IN ({placeholders})",
                conn, params=cwns_ids)

    if df.empty:
        print(f"Nothing new to export for round {args.round} "
              f"(pass --all to re-export everything already reviewed)")
        return

    print(f"Exporting {len(df)} reviewed plant(s) for round {args.round}, "
          f"{len(cand_df)} associated candidate row(s)")

    is_holdout = df["is_holdout"].astype(bool)
    holdout_ids = set(df.loc[is_holdout, "cwns_id"])
    holdout_df = df[is_holdout].copy()
    training_df = df[~is_holdout].copy()
    # Candidates only ever matter for Stage 2b training -- holdout candidates
    # aren't exported at all, since holdout plants never feed training.
    training_cand_df = cand_df[~cand_df["cwns_id"].isin(holdout_ids)].copy()

    # Re-fetch live parcel attributes for exported candidates -- the
    # `owner`/`lbcs_activity`/etc columns already in the candidates TABLE are
    # from the original queue load (10_build_review_queue.py's engineered
    # 10_parcel_features.parquet columns), almost certainly still NULL since
    # the app switched to live local lookups (parcels.py) for DISPLAY without
    # ever writing those fetched values back into SQLite. Exporting the stale
    # DB columns would silently ship empty/wrong parcel context into Stage 2b
    # training -- not what the reviewer actually saw on screen. Re-fetching
    # here, grouped by state for one query per state rather than one per row,
    # is what makes the export match reality.
    if len(training_cand_df):
        training_cand_df = training_cand_df.merge(
            training_df[["cwns_id", "state_code"]], on="cwns_id", how="left")
        enriched = []
        for state, grp in training_cand_df.groupby("state_code"):
            context = parcels.get_parcel_context(state, grp["ll_uuid"].tolist())
            grp = grp.copy()
            for col in parcels.ATTR_COLS:
                grp[col] = grp["ll_uuid"].map(lambda u: context.get(u, {}).get(col))
            enriched.append(grp)
        training_cand_df = pd.concat(enriched, ignore_index=True) if enriched else training_cand_df
        # Old engineered-schema columns (owner/lbcs_activity/etc from the
        # candidates table itself) are superseded by the live raw ones above
        # -- drop them rather than exporting two conflicting versions of the
        # same information under overlapping names.
        stale_cols = [c for c in ("owner", "lbcs_activity", "lbcs_ownership",
                                   "lbcs_function", "lbcs_structure", "lbcs_site",
                                   "dominant_class_group") if c in training_cand_df.columns]
        training_cand_df = training_cand_df.drop(columns=stale_cols)

    if len(training_df):
        out_path = C.OUTGOING_DIR / f"review_log_round{args.round}.parquet"
        if out_path.exists() and not args.all:
            existing = pd.read_parquet(out_path)
            training_df = pd.concat([existing, training_df], ignore_index=True)
            training_df = training_df.drop_duplicates(subset="cwns_id", keep="last")
        training_df.to_parquet(out_path, index=False)
        print(f"  {out_path.name}: {len(training_df)} row(s) total")

    if len(training_cand_df):
        cand_out_path = C.OUTGOING_DIR / f"review_log_candidates_round{args.round}.parquet"
        if cand_out_path.exists() and not args.all:
            existing = pd.read_parquet(cand_out_path)
            training_cand_df = pd.concat([existing, training_cand_df], ignore_index=True)
            # A plant can only be reviewed once (see routes/verdicts.py's 409
            # guard), so (cwns_id, candidate_rank) is a stable key even across
            # repeated pushes -- no risk of two different reviews' candidate
            # sets for the same plant colliding.
            training_cand_df = training_cand_df.drop_duplicates(
                subset=["cwns_id", "candidate_rank"], keep="last")
        training_cand_df.to_parquet(cand_out_path, index=False)
        print(f"  {cand_out_path.name}: {len(training_cand_df)} row(s) total")

    if len(holdout_df):
        # Shape roughly matches 09_build_holdout.py's holdout_truth.parquet
        # (CWNS_ID + true_ll_uuid, with a coordinate fallback for truth that
        # falls on no parcel) -- kept nullable/broad here since this gets
        # merged on HPC, not blindly appended; exact merge logic is a
        # decision for whoever does that merge, not this export step.
        holdout_out = holdout_df.rename(columns={
            "cwns_id": "CWNS_ID",
            "selected_ll_uuid": "true_ll_uuid",
            "truth_latitude": "true_lat",
            "truth_longitude": "true_lon",
        })[["CWNS_ID", "plant_verdict", "true_ll_uuid", "true_lat", "true_lon",
            "candidate_rank", "reviewer", "reviewed_at"]]

        out_path = C.OUTGOING_DIR / f"holdout_truth_round{args.round}.parquet"
        if out_path.exists() and not args.all:
            existing = pd.read_parquet(out_path)
            holdout_out = pd.concat([existing, holdout_out], ignore_index=True)
            holdout_out = holdout_out.drop_duplicates(subset="CWNS_ID", keep="last")
        holdout_out.to_parquet(out_path, index=False)
        print(f"  {out_path.name}: {len(holdout_out)} row(s) total")
        print(f"  NOTE: this needs to be merged into holdout_truth.parquet on HPC "
              f"manually -- it is NOT training data and must not go through "
              f"11_ingest_review_log.py.")

    with db.get_conn() as conn:
        now = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            "INSERT OR REPLACE INTO export_log (cwns_id, review_round, exported_at) VALUES (?, ?, ?)",
            [(cid, args.round, now) for cid in df["cwns_id"]]
        )

    print(f"\nSync these files back to HPC via MobaXterm (or however you're "
          f"transferring) -- {C.OUTGOING_DIR}")


if __name__ == "__main__":
    main()
