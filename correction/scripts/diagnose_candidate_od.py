"""
diagnose_candidate_od.py
========================
Read-only. Answers, before any modelling: does object detection actually
DISCRIMINATE among Stage 2a's top-K candidates for a plant?

WHY THIS COMES FIRST
    The hypothesis is that OD earns its keep among close competitors --
    the top ~20 candidates are all plausible municipal parcels, parcel
    features are near-tied, and only OD can look at the infrastructure.
    That hypothesis has a precondition: OD has to fire on SOME candidates
    and not others, within the same plant.

    Two failure modes make re-ranking pointless regardless of model quality:
      - fires almost never (say 5% of candidates): nearly every plant has
        20 candidates all reading "nothing here", so there is nothing to
        rank on
      - fires almost always (say 90%): same problem, inverted

    The useful regime is in between, and better still is when the TRUE
    parcel fires more often than its competitors. That last comparison is
    the actual point, and this script measures it directly against the
    holdout truth -- no model, no training, no distribution-shift caveat.

    01c found detections on 232 of 423 tiles at KNOWN-CORRECT locations
    (55%). That is the natural reference: candidates should fire well below
    it, and true parcels among the candidates should fire near it.

WHAT IT REPORTS
    1. overall od_ran / od_has_detection rates across all candidates
    2. distribution of "how many of this plant's candidates fired" -- the
       discriminative-power number
    3. true parcel vs competitors: detection rate, mean confidence, and
       whether the true parcel ranks top-1 by OD alone
    4. a naive OD-only recall@1, as a floor -- if OD alone already picks
       the right parcel often, Stage 2b re-ranking has real headroom

Usage (via sbatch diagnose_candidate_od.slurm):
    python diagnose_candidate_od.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from holdout import MANIFEST_PATH, TRUTH_PATH

CAND_ROOT = C.DATA_DIR / "od_features_candidates" / "candidates"


def load_candidate_od() -> pd.DataFrame:
    files = list(CAND_ROOT.rglob("*.parquet"))
    if not files:
        print(f"ERROR: no parquet under {CAND_ROOT}. Run 01e first.")
        sys.exit(2)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["CWNS_ID"] = df["CWNS_ID"].astype(str)
    df["ll_uuid"] = df["ll_uuid"].astype(str)
    return df.drop_duplicates(subset=["CWNS_ID", "ll_uuid"], keep="last")


def main():
    print("=== diagnose_candidate_od.py ===\n")

    od = load_candidate_od()
    print(f"Candidate OD rows: {len(od)} across {od['CWNS_ID'].nunique()} plant(s)")

    # ---- 1. overall rates -------------------------------------------------
    print("\n--- 1. overall ---")
    print(f"  od_ran           : {od['od_ran'].mean():.1%}  "
          f"({int(od['od_ran'].sum())}/{len(od)})")
    fired = od["od_has_detection"].fillna(False)
    print(f"  od_has_detection : {fired.mean():.1%}  ({int(fired.sum())}/{len(od)})")
    print(f"  reference: 01c saw 55% at KNOWN-CORRECT locations "
          f"(232 of 423 tiles)")

    # ---- 2. discriminative power -----------------------------------------
    print("\n--- 2. how many of each plant's candidates fired ---")
    per_plant = od.groupby("CWNS_ID")["od_has_detection"].agg(["sum", "count"])
    per_plant["sum"] = per_plant["sum"].astype(int)
    vc = per_plant["sum"].value_counts().sort_index()
    for k, n in vc.items():
        bar = "#" * min(40, n)
        print(f"    {k:3d} fired: {n:4d} plant(s)  {bar}")
    n_all = int((per_plant["sum"] == per_plant["count"]).sum())
    n_none = int((per_plant["sum"] == 0).sum())
    n_mixed = len(per_plant) - n_all - n_none
    print(f"\n    all candidates fired : {n_all} plant(s)  (OD cannot rank these)")
    print(f"    no candidate fired   : {n_none} plant(s)  (OD cannot rank these)")
    print(f"    MIXED                : {n_mixed} plant(s)  (OD has something to say)")
    if len(per_plant):
        print(f"    -> OD is potentially informative for "
              f"{n_mixed / len(per_plant):.0%} of plants")

    # ---- 3. true parcel vs competitors ------------------------------------
    if not TRUTH_PATH.exists():
        print(f"\n(no {TRUTH_PATH.name} -- skipping the truth comparison)")
        return
    truth = pd.read_parquet(TRUTH_PATH)
    truth["CWNS_ID"] = truth["CWNS_ID"].astype(str)
    truth = truth[truth["true_ll_uuid"].notna()].copy()
    truth["true_ll_uuid"] = truth["true_ll_uuid"].astype(str)

    od = od.merge(truth[["CWNS_ID", "true_ll_uuid"]], on="CWNS_ID", how="left")
    od["is_true"] = od["ll_uuid"] == od["true_ll_uuid"]

    scored = od[od["true_ll_uuid"].notna()]
    plants_with_truth = scored["CWNS_ID"].nunique()
    hits = scored.groupby("CWNS_ID")["is_true"].any()
    n_pool_hit = int(hits.sum())

    print("\n--- 3. true parcel vs competitors ---")
    print(f"  plants with a resolved true parcel : {plants_with_truth}")
    print(f"  true parcel present in the top-K   : {n_pool_hit} "
          f"({n_pool_hit / plants_with_truth:.0%} candidate recall@K)")
    if n_pool_hit == 0:
        print("\n  The true parcel is never in the candidate pool. Re-ranking "
              "cannot help -- the ceiling is 0. The problem is Stage 2a's "
              "candidate generation (K_RINGS reach or the reported-parcel "
              "exclusion), not the ranking model.")
        return

    only = scored[scored["CWNS_ID"].isin(hits[hits].index)]
    t = only[only["is_true"]]
    f = only[~only["is_true"]]
    print(f"\n  detection rate -- true parcels : {t['od_has_detection'].mean():.1%} "
          f"(n={len(t)})")
    print(f"  detection rate -- competitors  : {f['od_has_detection'].mean():.1%} "
          f"(n={len(f)})")
    lift = t["od_has_detection"].mean() - f["od_has_detection"].mean()
    print(f"  difference                     : {lift:+.1%}")

    for col in ["od_max_confidence", "od_n_objects"]:
        if col in only.columns:
            print(f"  {col:18s} true {t[col].mean():.3f}  "
                  f"vs competitors {f[col].mean():.3f}")

    # ---- 4. OD-only recall@1 ---------------------------------------------
    print("\n--- 4. OD-only recall@1 (no model, just rank by confidence) ---")
    ranked = only.sort_values(
        ["CWNS_ID", "od_has_detection", "od_max_confidence"],
        ascending=[True, False, False], na_position="last")
    top1 = ranked.groupby("CWNS_ID").head(1)
    od_r1 = top1["is_true"].mean()
    print(f"  OD alone picks the true parcel first: {od_r1:.1%} "
          f"({int(top1['is_true'].sum())}/{len(top1)})")
    print(f"  random baseline within a 20-candidate pool: 5.0%")

    print("\n=== reading this ===")
    if lift > 0.15:
        print("  True parcels fire clearly more than competitors. OD carries")
        print("  signal at the candidate level that Stage 2b's reported-vs-")
        print("  corrected training never had a chance to show. Building the")
        print("  re-rank -- and retraining Stage 2b on this distribution -- is")
        print("  justified.")
    elif lift > 0.05:
        print("  A modest edge. Real but small; with this holdout's ~7pp")
        print("  standard error, a re-rank built on it may not produce a")
        print("  measurable recall@1 gain. Worth retraining Stage 2b on")
        print("  candidate-distribution data before judging.")
    else:
        print("  True parcels fire no more than competitors. At current")
        print("  detector quality OD does not separate the true parcel from")
        print("  its close competitors, and re-ranking cannot help. Check")
        print("  section 2: if most plants are all-fired or none-fired, the")
        print("  detector is not the bottleneck -- candidate generation is.")


if __name__ == "__main__":
    main()
