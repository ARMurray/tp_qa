"""
12_score_holdout.py
===================
Scores the frozen evaluation holdout. This is the clean read on the
pipeline: every plant here was excluded from Stage 1, 2a, 2b and the
re-ranker by the anti-joins, so unlike 07b's internal test split these
numbers involve no data the models selected hyperparameters against.

THREE SEPARATE METRICS, NOT ONE
    Straight from 09_build_holdout.py's design. Averaging them hides both:

    corrections  recall@k. Of plants known to be misplaced, how often does
                 the pipeline land on the right parcel. The headline, and
                 what the re-ranker exists to move.
    correct      false-move rate. Of plants already correct, how often does
                 the pipeline wrongly flag them for relocation. A real harm,
                 invisible in the metric above, and the thing that degrades
                 if the model is tuned only on corrections. Note this is a
                 STAGE 1 property -- re-ranking cannot change it, since a
                 plant Stage 1 never flagged has no candidates to reorder.
    unlabeled    the deployment distribution. Reported only once review has
                 supplied truth; until then these plants score as unknown
                 rather than counting against anything.

TWO DENOMINATORS, BOTH REPORTED
    Restricting to plants whose pool contains the true parcel measures
    RANKING. Using every holdout plant measures the PIPELINE. The second is
    always lower and is the honest end-to-end figure -- a plant whose true
    parcel never entered the candidate pool is a failure regardless of how
    good the ranking is. Reporting only the first would flatter every arm
    equally and overstate what the system does.

MATCHING
    Primary test is ll_uuid equality against holdout_truth. Where truth has
    no resolved uuid, falls back to a geometric containment check: does the
    top-ranked candidate parcel contain the true point. 09's docstring
    prefers containment precisely because it survives Regrid vintage churn
    in a way that comparing uuid strings does not.

Usage:
    python 12_score_holdout.py
    python 12_score_holdout.py --no-od          # score the ablation table
    python 12_score_holdout.py --bins corrections,correct
"""
import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from holdout import MANIFEST_PATH, TRUTH_PATH

INFER_DIR = C.DATA_DIR / "inference"
KS = (1, 3, 5, 10)


def resolve_containment(con, pairs: pd.DataFrame) -> set:
    """(CWNS_ID, ll_uuid) pairs where the parcel contains the true point.

    Only called for rows whose truth carries no resolved uuid, so this is a
    handful of lookups, not a scan."""
    hits = set()
    for state, grp in pairs.groupby("STATE_CODE"):
        con.register("want", grp[["CWNS_ID", "ll_uuid", "true_lon", "true_lat"]])
        try:
            res = con.execute(f"""
                SELECT w.CWNS_ID, w.ll_uuid
                FROM read_parquet('{C.PARCEL_BASE.as_posix()}/state={state}/*.parquet') p
                JOIN want w ON w.ll_uuid = p.{C.PARCEL_ID_FIELD}
                WHERE ST_Intersects(ST_GeomFromWKB(p.{C.PARCEL_WKB_FIELD}),
                                    ST_Point(w.true_lon, w.true_lat))
            """).df()
            hits |= set(zip(res["CWNS_ID"].astype(str), res["ll_uuid"].astype(str)))
        except Exception as e:
            print(f"    [{state}] containment check failed: {str(e)[:150]}")
        finally:
            con.unregister("want")
    return hits


def recall_table(cand: pd.DataFrame, rank_col: str, n_all: int) -> dict:
    """recall@k under both denominators."""
    in_pool = cand.groupby("CWNS_ID")["is_true"].max()
    n_pool = int(in_pool.sum())
    out = {}
    for k in KS:
        topk = cand[cand[rank_col] <= k]
        hits = int(topk.groupby("CWNS_ID")["is_true"].max().sum())
        out[k] = (hits, n_pool, n_all)
    return out


def print_recall(label: str, tbl: dict):
    print(f"\n  {label}")
    for k, (hits, n_pool, n_all) in tbl.items():
        pool_pct = hits / n_pool if n_pool else float("nan")
        all_pct = hits / n_all if n_all else float("nan")
        print(f"    recall@{k:<2}  {hits:3d}/{n_pool:<3d} = {pool_pct:6.1%} "
              f"(pool)     {hits:3d}/{n_all:<3d} = {all_pct:6.1%} (all)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-od", action="store_true",
                    help="score stage2_candidates_reranked_noOD.parquet")
    ap.add_argument("--bins", default="corrections,correct",
                    help="holdout bins to score (default: corrections,correct)")
    ap.add_argument("--cohort", default=None, help="restrict to one cohort")
    args = ap.parse_args()

    tag = "_noOD" if args.no_od else ""
    bins = [b.strip() for b in args.bins.split(",")]

    print("=== 12_score_holdout.py ===")
    if args.no_od:
        print("*** scoring the --no-od ablation table ***")

    for p in (MANIFEST_PATH, TRUTH_PATH):
        if not p.exists():
            print(f"ERROR: {p} not found. Run 09_build_holdout.py first.")
            sys.exit(2)

    man = pd.read_parquet(MANIFEST_PATH)
    man["CWNS_ID"] = man["CWNS_ID"].astype(str)
    if args.cohort:
        man = man[man["cohort"] == args.cohort]
        print(f"Cohort {args.cohort} only")
    truth = pd.read_parquet(TRUTH_PATH)
    truth["CWNS_ID"] = truth["CWNS_ID"].astype(str)

    print(f"\nManifest: {len(man)} plant(s)")
    for b, n in man["bin"].value_counts().items():
        print(f"  {b:12s}: {n}")

    cand_path = INFER_DIR / f"stage2_candidates_reranked{tag}.parquet"
    if not cand_path.exists():
        print(f"\nERROR: {cand_path} not found. Run 05b_rerank_candidates.py"
              f"{' --no-od' if args.no_od else ''} first.")
        sys.exit(2)
    cand = pd.read_parquet(cand_path)
    cand["CWNS_ID"] = cand["CWNS_ID"].astype(str)
    cand["ll_uuid"] = cand["ll_uuid"].astype(str)

    summ_path = INFER_DIR / "plant_summary.parquet"
    summ = pd.read_parquet(summ_path) if summ_path.exists() else None
    if summ is not None:
        summ["CWNS_ID"] = summ["CWNS_ID"].astype(str)

    # =======================================================================
    # corrections -- the headline
    # =======================================================================
    if "corrections" in bins:
        print(f"\n{'=' * 66}\nCORRECTIONS BIN -- recall@k\n{'=' * 66}")
        held = man[man["bin"] == "corrections"]
        n_all = len(held)
        t = truth[truth["CWNS_ID"].isin(set(held["CWNS_ID"]))]
        print(f"  Holdout corrections plants : {n_all}")
        print(f"  With derivable truth        : {len(t)}")

        c = cand[cand["CWNS_ID"].isin(set(t["CWNS_ID"]))].merge(
            t[["CWNS_ID", "true_ll_uuid", "true_lon", "true_lat"]],
            on="CWNS_ID", how="left")
        n_with_cands = c["CWNS_ID"].nunique()
        print(f"  With Stage 2a candidates    : {n_with_cands}")
        print(f"  With NO candidates          : {n_all - n_with_cands}  "
              f"(unreachable by any ranking)")

        c["is_true"] = (c["ll_uuid"] == c["true_ll_uuid"].astype(str))

        # containment fallback where truth has no uuid
        need = c[c["true_ll_uuid"].isna() & c["true_lon"].notna()]
        if len(need):
            print(f"\n  {need['CWNS_ID'].nunique()} plant(s) have no resolved "
                  f"true_ll_uuid -- checking containment geometrically")
            con = duckdb.connect()
            con.execute("INSTALL spatial; LOAD spatial; "
                        "SET enable_geoparquet_conversion = false;")
            hits = resolve_containment(con, need)
            con.close()
            if hits:
                mask = [(r.CWNS_ID, r.ll_uuid) in hits for r in c.itertuples()]
                c["is_true"] = c["is_true"] | pd.Series(mask, index=c.index)
                print(f"    recovered {len(hits)} containment match(es)")

        in_pool = int(c.groupby("CWNS_ID")["is_true"].max().sum())
        print(f"\n  True parcel in the top-20   : {in_pool}/{n_all} "
              f"({in_pool / n_all:.0%} candidate recall)")
        print(f"    -> that is the CEILING for every arm below.")

        stage2a = recall_table(c, "stage2a_rank", n_all)
        rerank = recall_table(c, "rerank_rank", n_all)
        print_recall("Stage 2a ordering", stage2a)
        print_recall(f"Re-ranked ordering{tag}", rerank)

        print("\n  Delta (re-rank minus Stage 2a), pool denominator:")
        for k in KS:
            d = (rerank[k][0] - stage2a[k][0])
            n_pool = rerank[k][1]
            print(f"    recall@{k:<2}  {d:+3d} plant(s)  "
                  f"{d / n_pool if n_pool else float('nan'):+7.1%}")

        se = 100 * 0.5 / np.sqrt(max(in_pool, 1))
        print(f"\n  n={in_pool} scoreable -> standard error about "
              f"+/-{se:.0f}pp on any single figure.")
        print(f"  Differences smaller than that are not distinguishable "
              f"from noise.")

        if "rerank_fallback" in c.columns:
            fb = c[c["rerank_fallback"]]["CWNS_ID"].nunique()
            print(f"\n  Plants resolved by Stage 2a fallback (nothing fired): "
                  f"{fb}/{n_with_cands}")

    # =======================================================================
    # correct -- false-move rate
    # =======================================================================
    if "correct" in bins and summ is not None:
        print(f"\n{'=' * 66}\nCORRECT BIN -- false-move rate\n{'=' * 66}")
        held = man[man["bin"] == "correct"]
        s = summ[summ["CWNS_ID"].isin(set(held["CWNS_ID"]))]
        print(f"  Holdout correct plants   : {len(held)}")
        print(f"  Present in plant_summary : {len(s)}")
        if len(s):
            flagged = s[s["trigger_reason"] != "none"]
            print(f"\n  Flagged for relocation   : {len(flagged)} "
                  f"({len(flagged) / len(s):.1%})  <-- FALSE-MOVE RATE")
            for r, n in flagged["trigger_reason"].value_counts().items():
                print(f"    {r:16s}: {n}")
            print(f"\n  These plants are known CORRECT, so every flag is a "
                  f"false positive.")
            print(f"  This is a STAGE 1 property -- re-ranking cannot change "
                  f"it. A plant\n  Stage 1 never flagged has no candidates to "
                  f"reorder.")
            no_parcel = int((flagged["trigger_reason"] == "no_parcel").sum())
            if no_parcel:
                print(f"\n  NOTE: {no_parcel} were flagged as no_parcel -- the "
                      f"reported point hit\n  no parcel at all. That is a "
                      f"Regrid coverage gap, not a model error,\n  and is "
                      f"arguably not a 'false move' in the same sense.")

    # =======================================================================
    # unlabeled
    # =======================================================================
    if "unlabeled" in bins:
        print(f"\n{'=' * 66}\nUNLABELED BIN -- deployment distribution\n{'=' * 66}")
        held = man[man["bin"] == "unlabeled"]
        scored = truth[truth["CWNS_ID"].isin(set(held["CWNS_ID"]))]
        print(f"  Plants: {len(held)}  |  with truth: {len(scored)}")
        if len(scored) == 0:
            print("  None reviewed yet. This is the ONLY slice that estimates "
                  "the true\n  national error rate -- review it first in the "
                  "next round.")

    print(f"\n{'=' * 66}")
    print("Reading these: the 'pool' denominator measures RANKING, the 'all'")
    print("denominator measures the PIPELINE. If they diverge sharply, the")
    print("gap belongs to candidate generation (08_diagnose_candidate_")
    print("coverage.py), not to any ranking model.")


if __name__ == "__main__":
    main()
