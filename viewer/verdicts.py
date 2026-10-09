"""
verdicts.py -- verify / correct a plant's location from the viewer
==================================================================
Writes into the review app's own database (review_app/data/app.db), as review
round VIEWER_ROUND (900), so viewer verdicts reach the master and training
exactly like review-round verdicts do (decided 2026-10-09):

    python -m sync.close_round --round 900        (from review_app/)

which runs push_review_log -> update_master_locations -> build_training_bins
-> extract_review_tiles for them. Decided with the project owner: they feed
TRAINING (the master), but not the move-rule calibration or the audit --
those need randomly chosen plants, and a plant someone chose to look at is
not random. calibrate_move_rule.py only reads the rounds it is given, so
round 900 stays out unless asked for.

VERDICTS (same meanings as the review app; update_master_locations.py turns
them into master columns)
    reported_correct          the reported location is right
    candidate_correct         this parcel is the plant -- ANY parcel, scored
                              candidate or not; candidate_rank is its re-rank
                              rank when it is a candidate, else NULL.
                              site_ll_uuids: other parcels of the same plant.
    truth_outside_candidates  a clicked point (no parcel)
    needs_info                undecided; notes only, master untouched

LOCKS: a plant already in app.db under another round -- reviewed there, or
waiting unreviewed in an open queue (e.g. round 6's) -- is not written from
the viewer. Overwriting it would rewrite that round's record, and a viewer
verdict on a queued plant would bias the round's random/audit samples.
Change those in the review app.

A re-verdict on a viewer plant replaces the earlier one and clears its
export_log entry so push_review_log exports the new one.
"""
import getpass
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VIEWER_ROUND = 900
VERDICTS = {"reported_correct", "candidate_correct", "truth_outside_candidates", "needs_info"}


def _app_db_path() -> Path:
    """review_app/config.py's APP_DB_PATH, or VIEWER_APP_DB (testing on a copy)."""
    if os.environ.get("VIEWER_APP_DB"):
        return Path(os.environ["VIEWER_APP_DB"])
    sys.path.insert(0, str(REPO / "review_app"))
    try:
        import config as RC
    finally:
        sys.path.pop(0)
    return Path(RC.APP_DB_PATH)


def _connect():
    """app.db with the review app's schema and migrations applied (adds
    site_ll_uuids etc. to an older file), via the review app's own init."""
    path = _app_db_path()
    sys.path.insert(0, str(REPO / "review_app"))
    try:
        from backend import db as RDB
        RDB.C.APP_DB_PATH = path          # the same file this module writes
        RDB.init_db()
    finally:
        sys.path.pop(0)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def status() -> dict:
    """{cwns_id: {...}} for every plant app.db knows: viewer verdicts and the
    locked rows of other rounds. Read-only."""
    p = _app_db_path()
    if not p.exists():
        return {}
    conn = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT cwns_id, review_round, reviewed, plant_verdict, selected_ll_uuid, "
            "site_ll_uuids, truth_latitude, truth_longitude, reviewer_notes, "
            "reviewed_at, reviewer FROM plants").fetchall()
    except sqlite3.OperationalError:
        rows = conn.execute(
            "SELECT cwns_id, review_round, reviewed, plant_verdict, selected_ll_uuid, "
            "NULL AS site_ll_uuids, truth_latitude, truth_longitude, reviewer_notes, "
            "reviewed_at, reviewer FROM plants").fetchall()
    finally:
        conn.close()
    out = {}
    for r in rows:
        d = dict(r)
        d["viewer"] = d["review_round"] == VIEWER_ROUND
        d["locked"] = not d["viewer"]
        out[str(d["cwns_id"])] = d
    return out


def save(payload: dict, plant: dict, cands: list[dict]) -> dict:
    """payload: cwns_id, verdict, parcel, site_parcels, lat, lon, notes.
    plant: the viewer's plant record (state, rlat/rlon, rparcel, s1, reason).
    cands: the plant's scored candidates (for candidate_rank).
    Returns the stored row, or raises ValueError with a message for the user."""
    pid = str(payload.get("cwns_id") or "")
    verdict = payload.get("verdict")
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}")
    if not plant:
        raise ValueError(f"no plant {pid}")
    parcel = payload.get("parcel") or None
    sites = [u for u in dict.fromkeys(payload.get("site_parcels") or []) if u and u != parcel]
    lat, lon = payload.get("lat"), payload.get("lon")
    notes = (payload.get("notes") or "").strip() or None

    rank = None
    if verdict == "candidate_correct":
        if not parcel:
            raise ValueError("pick the parcel that is the plant")
        rank = next((c.get("rank") for c in cands if c.get("parcel") == parcel), None)
    elif verdict == "reported_correct":
        parcel = None
    if verdict == "truth_outside_candidates":
        if lat is None or lon is None:
            raise ValueError("click the plant's location on the map first")
        lat, lon = float(lat), float(lon)
    else:
        lat = lon = None
    if verdict not in ("candidate_correct", "reported_correct"):
        sites = []

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn = _connect()
    try:
        cur = conn.execute("SELECT review_round, reviewed FROM plants WHERE cwns_id = ?",
                           (pid,)).fetchone()
        if cur is not None and cur["review_round"] != VIEWER_ROUND:
            what = "reviewed" if cur["reviewed"] else "queued, not yet reviewed,"
            raise ValueError(f"{pid} is {what} in review round {cur['review_round']}; "
                             f"change it in the review app")
        row = {
            "cwns_id": pid, "state_code": plant.get("state"),
            "latitude": plant.get("rlat"), "longitude": plant.get("rlon"),
            "reported_ll_uuid": plant.get("rparcel"),
            "stage1_prob_correct": plant.get("s1"),
            "trigger_reason": plant.get("reason") or plant.get("status"),
            "review_task": "viewer", "queue_slice": "viewer",
            "model_version": "viewer", "review_round": VIEWER_ROUND, "is_holdout": 0,
            "reviewed": 1, "plant_verdict": verdict,
            "selected_ll_uuid": parcel if verdict == "candidate_correct" else None,
            "site_ll_uuids": ";".join(sites) or None,
            "candidate_rank": rank, "truth_rank": rank,
            "truth_latitude": lat, "truth_longitude": lon,
            "confirmation_type": "independent" if verdict == "candidate_correct" else None,
            "reviewer_notes": notes, "reviewed_at": now, "reviewer": getpass.getuser(),
            "pop_served": plant.get("pop"),
        }
        cols = ", ".join(row)
        conn.execute(f"INSERT OR REPLACE INTO plants ({cols}) VALUES "
                     f"({', '.join('?' * len(row))})", list(row.values()))
        try:
            conn.execute("DELETE FROM export_log WHERE cwns_id = ? AND review_round = ?",
                         (pid, VIEWER_ROUND))
        except sqlite3.OperationalError:
            pass                     # no export_log yet: nothing exported
        conn.commit()
    finally:
        conn.close()
    return row


def undo(pid: str) -> bool:
    """Remove a viewer verdict (only round-900 rows)."""
    conn = _connect()
    try:
        n = conn.execute("DELETE FROM plants WHERE cwns_id = ? AND review_round = ?",
                         (pid, VIEWER_ROUND)).rowcount
        try:
            conn.execute("DELETE FROM export_log WHERE cwns_id = ? AND review_round = ?",
                         (pid, VIEWER_ROUND))
        except sqlite3.OperationalError:
            pass
        conn.commit()
    finally:
        conn.close()
    return bool(n)
