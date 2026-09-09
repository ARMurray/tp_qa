"""
db.py
=====
SQLite is the live store while reviewing -- parquet is a poor fit for
row-by-row writes/updates/resume, and this pipeline has already been bitten
all week by parquet append/schema-merge problems (see TPQA_MASTER_REFERENCE.md).
Parquet is the right format again at the export boundary (sync/push_review_log.py).

Two tables:
  plants     one row per CWNS_ID. Queue metadata (queue_slice, review_task,
             model_version, etc.) plus verdict fields, NULL until reviewed.
  candidates one row per (CWNS_ID, candidate_rank), only for plants whose
             review_task == 'candidate_pick'. Read-only reference data from
             the queue -- never written to after import.

Schema follows REVIEW_LOOP_PLAN.md Phase 4's four requirements directly:
  1. truth_outside_candidates verdict path -> plant_verdict + truth_lat/lon
  2. rank capture             -> candidate_rank, truth_rank (NULL iff #1)
  3. holdout routing          -> is_holdout, checked at export time to split
                                 output into training-feed vs holdout_truth
  4. provenance                -> model_version, queue_slice, review_round,
                                 reviewed_at, reviewer, confirmation_type
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import config as C

SCHEMA = """
CREATE TABLE IF NOT EXISTS plants (
    cwns_id             TEXT PRIMARY KEY,
    state_code          TEXT,
    latitude            REAL,
    longitude           REAL,
    reported_ll_uuid    TEXT,
    stage1_prob_correct REAL,
    trigger_reason      TEXT,
    review_task         TEXT NOT NULL,   -- 'candidate_pick' | 'confirm_reported'
    queue_slice         TEXT NOT NULL,   -- 'holdout' | 'uncertain' | 'random'
    model_version        TEXT,
    review_round         INTEGER,
    is_holdout           INTEGER NOT NULL DEFAULT 0,

    -- Verdict fields, NULL until reviewed
    reviewed              INTEGER NOT NULL DEFAULT 0,
    plant_verdict          TEXT,    -- 'reported_correct' | 'candidate_correct'
                                    -- | 'truth_outside_candidates' | 'needs_info'
    selected_ll_uuid        TEXT,   -- set iff plant_verdict == 'candidate_correct'
    candidate_rank           INTEGER, -- rank of selected candidate, NULL otherwise
    truth_rank                INTEGER, -- == candidate_rank unless truth_outside_candidates (NULL)
    truth_latitude             REAL,   -- set iff truth_outside_candidates
    truth_longitude             REAL,
    confirmation_type            TEXT, -- 'independent' | 'confirmed_proposal'
    reviewer_notes                 TEXT,
    reviewed_at                     TEXT,
    reviewer                         TEXT,

    -- Descriptive info for the plant-info panel (added 2026-08-26).
    -- No facility NAME field is confirmed to exist in any CWNS table --
    -- address/city/county are the identifying info used instead. See
    -- 10_build_review_queue.py's load_plant_display_info() docstring.
    address         TEXT,
    city            TEXT,
    county_name     TEXT,
    zip_code        TEXT,
    pop_served       REAL,
    subdivision       TEXT,
    place              TEXT,
    county              TEXT,
    is_rural              INTEGER,
    surface_water_discharge INTEGER,
    requires_npdes            INTEGER,
    any_reuse                    INTEGER
);

CREATE TABLE IF NOT EXISTS candidates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cwns_id         TEXT NOT NULL,
    candidate_rank  INTEGER NOT NULL,
    ll_uuid         TEXT NOT NULL,
    stage2a_score   REAL,
    score_margin    REAL,
    distance_m      REAL,
    within_1km      INTEGER,
    within_5km      INTEGER,

    -- Parcel-info added 2026-08-26 (see 10_build_review_queue.py's
    -- PARCEL_INFO_COLS) -- was present in stage2_candidates.parquet from
    -- day one but silently dropped by an earlier version of the queue
    -- builder's narrow column select.
    owner            TEXT,
    lbcs_activity     TEXT,
    lbcs_ownership     TEXT,
    lbcs_function        TEXT,
    lbcs_structure         TEXT,
    lbcs_site                TEXT,
    ll_gisacre                 REAL,
    ll_bldg_count                REAL,
    dominant_class_group           TEXT,
    has_ww_keyword                    INTEGER,
    osm_ww                               INTEGER,
    data_quality_score                    REAL,

    FOREIGN KEY (cwns_id) REFERENCES plants(cwns_id)
);

CREATE INDEX IF NOT EXISTS idx_candidates_cwns ON candidates(cwns_id);
CREATE INDEX IF NOT EXISTS idx_plants_reviewed ON plants(reviewed);
CREATE INDEX IF NOT EXISTS idx_plants_slice ON plants(queue_slice);
"""

# Columns added after the original schema shipped. SQLite's ALTER TABLE
# supports simple ADD COLUMN, so existing databases (with real, already-
# submitted verdicts) get these added additively rather than needing a
# wipe-and-reload. Keep this in sync with the CREATE TABLE columns above --
# it exists ONLY to bring an already-created table up to date.
PLANTS_MIGRATION_COLS = [
    ("address", "TEXT"), ("city", "TEXT"), ("county_name", "TEXT"),
    ("zip_code", "TEXT"), ("pop_served", "REAL"), ("subdivision", "TEXT"),
    ("place", "TEXT"), ("county", "TEXT"), ("is_rural", "INTEGER"),
    ("surface_water_discharge", "INTEGER"), ("requires_npdes", "INTEGER"),
    ("any_reuse", "INTEGER"),
]
CANDIDATES_MIGRATION_COLS = [
    ("owner", "TEXT"), ("lbcs_activity", "TEXT"), ("lbcs_ownership", "TEXT"),
    ("lbcs_function", "TEXT"), ("lbcs_structure", "TEXT"), ("lbcs_site", "TEXT"),
    ("ll_gisacre", "REAL"), ("ll_bldg_count", "REAL"),
    ("dominant_class_group", "TEXT"), ("has_ww_keyword", "INTEGER"),
    ("osm_ww", "INTEGER"), ("data_quality_score", "REAL"),
]


@contextmanager
def get_conn():
    conn = sqlite3.connect(C.APP_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def _migrate(conn):
    """Additive-only: adds any column from *_MIGRATION_COLS that doesn't
    already exist. Never drops or alters existing columns/data -- safe to
    call every startup, safe against a database with real submitted
    verdicts already in it."""
    for table, cols in (("plants", PLANTS_MIGRATION_COLS),
                        ("candidates", CANDIDATES_MIGRATION_COLS)):
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for col_name, col_type in cols:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                print(f"  [migration] added {table}.{col_name} ({col_type})")


def counts_by_slice(conn) -> list[dict]:
    rows = conn.execute("""
        SELECT queue_slice,
               COUNT(*) AS total,
               SUM(reviewed) AS reviewed
        FROM plants
        GROUP BY queue_slice
        ORDER BY queue_slice
    """).fetchall()
    return [dict(r) for r in rows]


def next_unreviewed(conn) -> dict | None:
    """Holdout plants first (round-1 priority: establish the benchmark before
    burning review time elsewhere), then everything else in whatever order
    they were inserted (queue builder already randomized within slices)."""
    row = conn.execute("""
        SELECT * FROM plants
        WHERE reviewed = 0
        ORDER BY (queue_slice = 'holdout') DESC, rowid
        LIMIT 1
    """).fetchone()
    return dict(row) if row else None


def get_plant(conn, cwns_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM plants WHERE cwns_id = ?", (cwns_id,)).fetchone()
    return dict(row) if row else None


def list_all_plants(conn, review_round: int) -> list[dict]:
    """Every plant in the given round, for the navigation list -- ordered
    the same way next_unreviewed serves them (holdout first) so the list
    order matches what "Next" would actually give you."""
    rows = conn.execute("""
        SELECT cwns_id, state_code, queue_slice, review_task, reviewed, plant_verdict
        FROM plants
        WHERE review_round = ?
        ORDER BY (queue_slice = 'holdout') DESC, queue_slice, rowid
    """, (review_round,)).fetchall()
    return [dict(r) for r in rows]


def get_candidates(conn, cwns_id: str) -> list[dict]:
    rows = conn.execute("""
        SELECT * FROM candidates WHERE cwns_id = ? ORDER BY candidate_rank
    """, (cwns_id,)).fetchall()
    return [dict(r) for r in rows]


def submit_verdict(conn, cwns_id: str, verdict: dict):
    """verdict keys: plant_verdict, selected_ll_uuid, candidate_rank,
    truth_rank, truth_latitude, truth_longitude, confirmation_type,
    reviewer_notes, reviewer, reviewed_at (all optional except plant_verdict/
    reviewer/reviewed_at -- validated in routes/verdicts.py before this is
    called, not re-validated here)."""
    conn.execute("""
        UPDATE plants SET
            reviewed = 1,
            plant_verdict = :plant_verdict,
            selected_ll_uuid = :selected_ll_uuid,
            candidate_rank = :candidate_rank,
            truth_rank = :truth_rank,
            truth_latitude = :truth_latitude,
            truth_longitude = :truth_longitude,
            confirmation_type = :confirmation_type,
            reviewer_notes = :reviewer_notes,
            reviewed_at = :reviewed_at,
            reviewer = :reviewer
        WHERE cwns_id = :cwns_id
    """, {**verdict, "cwns_id": cwns_id})
