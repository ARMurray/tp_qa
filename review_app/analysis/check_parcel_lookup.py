"""
check_parcel_lookup.py
======================
Why are parcel outlines missing in the review app? Runs the app's own parcel
lookup (backend.parcels.get_parcel_context, the local Regrid mirror) for
every plant of a review round and reports, per state, whether the files are
there and how many reported / candidate parcels came back.

Usage (from review_app/, the review app's Python):
    python -m analysis.check_parcel_lookup --round 6
"""
import argparse
import glob
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from backend import parcels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    args = ap.parse_args()

    print("=== check_parcel_lookup ===")
    print(f"REGRID_ROOT : {C.REGRID_ROOT}")
    print(f"  exists    : {Path(C.REGRID_ROOT).exists()}")
    try:
        con = parcels.get_connection()
        print(f"DuckDB      : {con.execute('SELECT version()').fetchone()[0]}, spatial loaded")
    except Exception as e:
        print(f"DuckDB      : FAILED to start / load spatial: {e}")
        return

    db = sqlite3.connect(C.APP_DB_PATH)
    plants = db.execute("SELECT cwns_id, state_code, reported_ll_uuid FROM plants "
                        "WHERE review_round = ?", (args.round,)).fetchall()
    cands = db.execute("SELECT c.cwns_id, c.ll_uuid FROM candidates c JOIN plants p "
                       "USING (cwns_id) WHERE p.review_round = ?", (args.round,)).fetchall()
    db.close()
    if not plants:
        print(f"No plants for round {args.round} in {C.APP_DB_PATH} -- run queue_loader first.")
        return
    state_of = {p[0]: p[1] for p in plants}

    by_state: dict[str, dict] = {}
    for cw, st, rep in plants:
        s = by_state.setdefault(st, {"rep": [], "cand": []})
        if rep:
            s["rep"].append(rep)
    for cw, uuid in cands:
        by_state.setdefault(state_of.get(cw), {"rep": [], "cand": []})["cand"].append(uuid)

    print(f"\nround {args.round}: {len(plants)} plants, {len(cands)} candidate rows, "
          f"{len(by_state)} states\n")
    print(f"  {'state':<6}{'files':>7}  {'reported found':>16}  {'candidates found':>18}")
    tot = [0, 0, 0, 0]
    for st in sorted(by_state, key=str):
        s = by_state[st]
        pattern = str(Path(C.REGRID_ROOT) / C.REGRID_STATE_GLOB.format(state=st))
        n_files = len(glob.glob(pattern))
        want = list(dict.fromkeys(s["rep"] + s["cand"]))
        found = parcels.get_parcel_context(st, want) if (n_files and want) else {}
        got = {u for u, e in found.items() if e.get("geometry")}
        r = (sum(u in got for u in s["rep"]), len(s["rep"]))
        c = (sum(u in got for u in s["cand"]), len(s["cand"]))
        tot = [tot[0] + r[0], tot[1] + r[1], tot[2] + c[0], tot[3] + c[1]]
        flag = "" if n_files and c[0] == c[1] and r[0] == r[1] else "   <--"
        print(f"  {str(st):<6}{n_files:>7}  {r[0]:>7} / {r[1]:<6}  {c[0]:>8} / {c[1]:<7}{flag}")
    print(f"\n  TOTAL         {tot[0]:>7} / {tot[1]:<6}  {tot[2]:>8} / {tot[3]}")
    print("\nfiles = 0     -> no state folder matching REGRID_STATE_GLOB under REGRID_ROOT "
          "(OneDrive not synced / folder moved / different layout)")
    print("files > 0 but few found -> the parcels are not in those files (older Regrid "
          "vintage than the HPC's) -- the server log shows the exact DuckDB error, if any")


if __name__ == "__main__":
    main()
