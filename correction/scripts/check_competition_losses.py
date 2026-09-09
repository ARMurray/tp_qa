"""
check_competition_losses.py  (revised)
=====================================
Original version asked: of the plants Stage 1 flagged, how many had raw
candidates but ended up with no ranked top pick after resolve_competition()?

That question is only answerable if `top_stage2_prob` in plant_summary.parquet
is written AFTER competition resolution. If it is written at scoring time (raw
max over the plant's candidates), it is non-null for every plant with any
candidate and the "lost to competition" count is pinned at 0 by construction.

This version does not assume. It:
  1. Rebuilds candidate counts and raw max scores directly from
     stage2_candidates.parquet and reconciles them against the summary columns.
  2. Compares top_stage2_prob against the raw max. If they are identical for
     every plant, top_stage2_prob is pre-competition and the original check
     was vacuous.
  3. Looks for duplicate parcel assignments across plants, which cannot occur
     if competition actually ran.
  4. Uses mutually exclusive buckets and reports unclassified rows explicitly.
  5. Always reads the candidate table, whether or not losses are detected.

Usage:
    python check_competition_losses.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

# Adjust if your summary uses a different name for the winning-parcel column.
PARCEL_ID_CANDIDATES = ["top_parcel_id", "top_ll_uuid", "ll_uuid", "parcel_id"]


def find_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def main():
    print("=== check_competition_losses.py (revised) ===")

    summary_path = C.DATA_DIR / "inference" / "plant_summary.parquet"
    cand_path = C.DATA_DIR / "inference" / "stage2_candidates.parquet"

    if not cand_path.exists():
        print(f"FATAL: {cand_path} not found. This check requires the raw "
              f"candidate table; the summary alone cannot answer the question.")
        return 1

    df = pd.read_parquet(summary_path)
    cand = pd.read_parquet(cand_path)

    print(f"\nplant_summary rows      : {len(df)}")
    print(f"stage2_candidates rows  : {len(cand)}")
    print(f"summary columns         : {sorted(df.columns.tolist())}")

    flagged = df[df["trigger_reason"] != "none"].copy()
    n_flagged = len(flagged)
    print(f"\nFlagged plants (trigger_reason != 'none'): {n_flagged}")
    print("  trigger_reason breakdown:")
    for reason, n in flagged["trigger_reason"].value_counts().items():
        print(f"    {reason:<40} {n}")

    # ---------------------------------------------------------------
    # 1. Rebuild ground truth from the raw candidate table
    # ---------------------------------------------------------------
    raw = (cand.groupby("CWNS_ID")
               .agg(raw_n_candidates=("stage2_prob_correct", "size"),
                    raw_max_prob=("stage2_prob_correct", "max"))
               .reset_index())

    flagged = flagged.merge(raw, on="CWNS_ID", how="left")
    flagged["raw_n_candidates"] = flagged["raw_n_candidates"].fillna(0).astype(int)

    print("\n--- Reconciling summary columns against raw candidate table ---")

    mismatch_n = flagged[flagged["n_candidates"] != flagged["raw_n_candidates"]]
    print(f"  plants where n_candidates != raw count : {len(mismatch_n)}")
    if len(mismatch_n):
        print("    ^ summary n_candidates is NOT the raw candidate count. "
              "It may already be post-filter. Sample:")
        print(mismatch_n[["CWNS_ID", "n_candidates", "raw_n_candidates"]].head())

    # ---------------------------------------------------------------
    # 2. The decisive test: is top_stage2_prob pre- or post-competition?
    # ---------------------------------------------------------------
    has_both = flagged["top_stage2_prob"].notna() & flagged["raw_max_prob"].notna()
    sub = flagged[has_both]
    identical = np.isclose(sub["top_stage2_prob"], sub["raw_max_prob"],
                           rtol=1e-9, atol=1e-12)
    n_identical = int(identical.sum())
    n_lower = int((sub["top_stage2_prob"] < sub["raw_max_prob"] - 1e-12).sum())
    n_higher = int((sub["top_stage2_prob"] > sub["raw_max_prob"] + 1e-12).sum())

    print("\n--- Provenance of top_stage2_prob ---")
    print(f"  comparable plants                      : {len(sub)}")
    print(f"  top_stage2_prob == raw max             : {n_identical}")
    print(f"  top_stage2_prob <  raw max (displaced) : {n_lower}")
    print(f"  top_stage2_prob >  raw max (!!)        : {n_higher}")

    if len(sub) and n_identical == len(sub):
        print("\n  >>> top_stage2_prob equals the raw max for EVERY plant.")
        print("  >>> Competition never displaced a single top pick, which means")
        print("  >>> either resolve_competition() is a no-op on this column, or")
        print("  >>> the column is written before competition runs. Either way,")
        print("  >>> the original zero-loss result proves nothing.")
    elif n_lower:
        print(f"\n  Competition demonstrably displaced {n_lower} top picks. "
              f"The column is post-competition and the loss count is real.")
    if n_higher:
        print("\n  WARNING: top_stage2_prob exceeds the raw max for some plants. "
              "The summary and candidate tables are out of sync (stale run?).")

    # ---------------------------------------------------------------
    # 3. Independent evidence: does any parcel win twice?
    # ---------------------------------------------------------------
    pid = find_col(flagged, PARCEL_ID_CANDIDATES)
    print("\n--- Parcel exclusivity ---")
    if pid is None:
        print(f"  No winning-parcel column found (looked for "
              f"{PARCEL_ID_CANDIDATES}). Cannot verify exclusivity; add the "
              f"correct column name to PARCEL_ID_CANDIDATES.")
    else:
        won = flagged[flagged[pid].notna()]
        dup = won[pid].duplicated(keep=False)
        n_dup_rows = int(dup.sum())
        n_dup_parcels = int(won.loc[dup, pid].nunique())
        print(f"  using column: {pid}")
        print(f"  plants with an assigned parcel        : {len(won)}")
        print(f"  plants sharing a parcel with another  : {n_dup_rows}")
        print(f"  distinct contested parcels            : {n_dup_parcels}")
        if n_dup_rows:
            print("  >>> The same parcel is assigned to multiple plants. "
                  "resolve_competition() did not enforce exclusivity.")
            print(won.loc[dup, ["CWNS_ID", pid, "top_stage2_prob"]]
                     .sort_values(pid).head(10).to_string(index=False))
        else:
            print("  No parcel is assigned twice. Consistent with competition "
                  "having run (though not proof on its own).")

    # ---------------------------------------------------------------
    # 4. Mutually exclusive buckets, with leftovers surfaced
    # ---------------------------------------------------------------
    has_pick = flagged["top_stage2_prob"].notna()
    has_cands = flagged["raw_n_candidates"] > 0

    b_pick = has_pick & has_cands
    b_no_cands = ~has_cands & ~has_pick
    b_lost = has_cands & ~has_pick
    b_impossible = ~has_cands & has_pick
    b_unclassified = ~(b_pick | b_no_cands | b_lost | b_impossible)

    print("\n--- Outcome buckets (mutually exclusive, raw counts) ---")
    print(f"  candidates and a top pick              : {int(b_pick.sum())}")
    print(f"  zero raw candidates, no pick           : {int(b_no_cands.sum())}")
    print(f"  candidates but lost ALL to competition : {int(b_lost.sum())}")
    print(f"  IMPOSSIBLE (pick with zero candidates) : {int(b_impossible.sum())}")
    print(f"  unclassified                           : {int(b_unclassified.sum())}")
    total = int((b_pick | b_no_cands | b_lost | b_impossible | b_unclassified).sum())
    print(f"\n  partition total: {total}  (should equal {n_flagged})")

    if int(b_impossible.sum()):
        print("  >>> Plants hold a top pick with no raw candidates. The summary "
              "and candidate tables disagree about which plants were scored.")

    # ---------------------------------------------------------------
    # 5. Always show detail, not only on failure
    # ---------------------------------------------------------------
    if int(b_lost.sum()):
        lost_ids = flagged.loc[b_lost, "CWNS_ID"]
        print(f"\n  Sample of plants that lost every candidate "
              f"(up to 5 of {len(lost_ids)}):")
        for cwns_id in lost_ids.head(5):
            pc = cand[cand["CWNS_ID"] == cwns_id]
            top = sorted(pc["stage2_prob_correct"].tolist(), reverse=True)[:5]
            print(f"    CWNS_ID={cwns_id}: {len(pc)} raw candidate(s), "
                  f"top scores: {top}")
    else:
        print("\n  No competition losses detected. Candidate-count distribution "
              "for flagged plants (to confirm there was anything to compete over):")
        print(flagged["raw_n_candidates"].describe().to_string())
        multi = int((flagged["raw_n_candidates"] > 1).sum())
        print(f"    plants with >1 candidate: {multi} of {n_flagged}")
        if multi == 0:
            print("    >>> No flagged plant has more than one candidate. There "
                  "was no competition to lose, and this check is uninformative.")

    print("\n=== complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())