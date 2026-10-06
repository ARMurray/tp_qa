"""
smoke_site_checkbox.py
======================
End-to-end smoke test of the "also part of this plant" checkbox (2026-10-05)
WITHOUT touching the real app.db: copies it to a temp file, points the app at
the copy, re-opens two already-reviewed candidate_pick plants there, and
submits verdicts with site_ll_uuids through the real route + validation.

Checks:
  1. candidate_correct + extras  -> plants.site_ll_uuids stored, main parcel
     and duplicates stripped
  2. reported_correct + extras   -> stored
  3. needs_info + extras         -> rejected (422 / ValueError)
  4. sync.update_master_locations.build_updates -> Site_UUIDs primary-first

Run from review_app/ on the work PC (the review app's Python):

    python -m analysis.smoke_site_checkbox

The browser checkbox itself still needs one manual click-through (start the
app, tick "also part", watch the status line) -- this covers everything from
the POST onward.
"""
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

tmp_db = Path(tempfile.mkdtemp()) / "app_smoke.db"
shutil.copy2(C.APP_DB_PATH, tmp_db)
C.APP_DB_PATH = tmp_db          # db.get_conn() reads this at call time
print(f"working on a copy: {tmp_db}")

from backend import db           # noqa: E402  (after the path patch)

db.init_db()                     # runs the additive migration on the copy

with db.get_conn() as conn:
    picks = conn.execute("""
        SELECT p.cwns_id, p.reported_ll_uuid FROM plants p
        JOIN candidates c ON c.cwns_id = p.cwns_id
        WHERE p.review_task = 'candidate_pick'
        GROUP BY p.cwns_id HAVING COUNT(*) >= 3
        ORDER BY p.rowid DESC LIMIT 3
    """).fetchall()
    if len(picks) < 3:
        sys.exit("need 3 candidate_pick plants with >= 3 candidates in app.db")
    ids = [r["cwns_id"] for r in picks]
    conn.execute(f"UPDATE plants SET reviewed = 0 WHERE cwns_id IN ({','.join('?' * 3)})", ids)
    cands = {i: [r["ll_uuid"] for r in db.get_candidates(conn, i)] for i in ids}

try:
    from fastapi.testclient import TestClient
    from backend.app import app
    client = TestClient(app)

    def post(body):
        r = client.post("/api/verdict", json=body)
        return r.status_code, r.text
except ImportError as e:         # TestClient needs httpx
    print(f"(no TestClient: {e}; calling the route function directly)")
    from backend.models import VerdictSubmit
    from backend.routes.verdicts import submit_verdict

    def post(body):
        try:
            return 200, str(submit_verdict(VerdictSubmit(**body)))
        except Exception as ex:  # pydantic ValidationError / HTTPException
            return 422, str(ex)

fails = 0


def check(label, ok, detail=""):
    global fails
    fails += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {detail}")


# 1. candidate_correct: main = rank 1, extras = rank 2 twice + the main itself
a, ca = ids[0], cands[ids[0]]
code, txt = post(dict(cwns_id=a, plant_verdict="candidate_correct", reviewer="smoke",
                      selected_ll_uuid=ca[0], candidate_rank=1,
                      site_ll_uuids=[ca[1], ca[1], ca[0]],
                      confirmation_type="independent"))
check("candidate_correct accepted", code == 200, f"({code} {txt[:120]})")

# 2. reported_correct with one extra
b, cb = ids[1], cands[ids[1]]
code, txt = post(dict(cwns_id=b, plant_verdict="reported_correct", reviewer="smoke",
                      site_ll_uuids=[cb[2]], confirmation_type="independent"))
check("reported_correct accepted", code == 200, f"({code} {txt[:120]})")

# 3. needs_info with extras must be refused
c, cc = ids[2], cands[ids[2]]
code, txt = post(dict(cwns_id=c, plant_verdict="needs_info", reviewer="smoke",
                      site_ll_uuids=[cc[0]]))
check("needs_info + extras refused", code == 422, f"({code})")

with sqlite3.connect(tmp_db) as conn:
    conn.row_factory = sqlite3.Row
    rows = {r["cwns_id"]: dict(r) for r in conn.execute(
        f"SELECT * FROM plants WHERE cwns_id IN ({','.join('?' * 3)})", ids)}
check("A site_ll_uuids == rank-2 only", rows[a]["site_ll_uuids"] == ca[1],
      repr(rows[a]["site_ll_uuids"]))
check("B site_ll_uuids stored", rows[b]["site_ll_uuids"] == cb[2],
      repr(rows[b]["site_ll_uuids"]))
check("C left unreviewed", rows[c]["reviewed"] == 0)

# 4. master Site_UUIDs (build_updates only; no master is written)
try:
    import pandas as pd
    from sync.update_master_locations import build_updates
    df = pd.DataFrame([rows[a], rows[b]])
    up = build_updates(df, {a: (0.0, 0.0)}).set_index("CWNS_ID")
    check("master A Site_UUIDs primary first",
          up.loc[a, "Site_UUIDs"] == f"{ca[0]};{ca[1]}", repr(up.loc[a, "Site_UUIDs"]))
    want_b = f"{rows[b]['reported_ll_uuid']};{cb[2]}" if rows[b]["reported_ll_uuid"] else cb[2]
    check("master B Site_UUIDs reported first",
          up.loc[b, "Site_UUIDs"] == want_b, repr(up.loc[b, "Site_UUIDs"]))
except ImportError as e:
    print(f"  [SKIP] build_updates ({e})")

print(f"\n{'ALL PASSED' if not fails else f'{fails} FAILED'}  (real app.db untouched)")
sys.exit(1 if fails else 0)
