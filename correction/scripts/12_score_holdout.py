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

INFER_DIR = C.INFERENCE_DIR
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


def multi_source_outcomes(held: pd.DataFrame, truth: pd.DataFrame, summ: pd.DataFrame, cand: pd.DataFrame,
                          site_parcels: dict) -> pd.DataFrame:
    """End-to-end outcome per holdout plant when Stage 1 can accept a SEED SITE (config.SEED_MODE == "multi").

    The recall@k tables above only see plants Stage 1 flagged. In multi mode a misplaced plant can be put right
    by Stage 1 itself, when a source other than the CWNS point is accepted, and such a plant has no candidates:
    it would read as a failure there. So each plant gets one outcome:
      accepted_right   Stage 1 accepted a site holding the truth parcel
      accepted_wrong   Stage 1 accepted a site that does not (a false accept: the plant stays wrong)
      moved_right      flagged, and the re-ranked #1 is the truth parcel
      flagged_wrong    flagged, and the #1 is not (or there is none)
      truth_unresolved no resolved truth parcel (not scored)
    plus the sources on the accepted site. site_parcels: site_id -> list of ll_uuid."""
    s = summ.set_index("CWNS_ID")
    t = truth.drop_duplicates("CWNS_ID").set_index("CWNS_ID")
    rank1 = cand[cand["rerank_rank"] == 1].drop_duplicates("CWNS_ID").set_index("CWNS_ID")["ll_uuid"]
    rows = []
    for cid, b in zip(held["CWNS_ID"], held["bin"]):
        if cid not in s.index:
            rows.append((cid, b, "not_scored", None))
            continue
        true_uuid = t["true_ll_uuid"].get(cid) if cid in t.index else None
        if not isinstance(true_uuid, str) or not true_uuid or true_uuid == "nan":
            rows.append((cid, b, "truth_unresolved", None))
            continue
        row = s.loc[cid]
        if row["trigger_reason"] == "none":
            parcels = set(site_parcels.get(row.get("accepted_site_id"), [])) or {str(row["reported_ll_uuid"])}
            rows.append((cid, b, "accepted_right" if true_uuid in parcels else "accepted_wrong",
                         row.get("accepted_sources")))
        else:
            rows.append((cid, b, "moved_right" if str(rank1.get(cid)) == true_uuid else "flagged_wrong", None))
    return pd.DataFrame(rows, columns=["CWNS_ID", "bin", "outcome", "accepted_sources"])


def print_multi_source(out: pd.DataFrame):
    print(f"\n{'=' * 66}\nMULTI-SOURCE END TO END -- Stage 1 accept, else re-ranked #1\n{'=' * 66}")
    for b, g in out.groupby("bin"):
        vc = g["outcome"].value_counts()
        scored = g[~g["outcome"].isin(["not_scored", "truth_unresolved"])]
        n = len(scored)
        right = int(scored["outcome"].isin(["accepted_right", "moved_right"]).sum())
        acc = scored[scored["outcome"].str.startswith("accepted")]
        print(f"\n  {b} bin: {len(g)} plants, {n} scored")
        for k, v in vc.items():
            print(f"    {k:<18} {v:>4}")
        print(f"    END-TO-END right   {right:>4} / {n} = {right / n if n else float('nan'):.1%}")
        if len(acc):
            r = int((acc["outcome"] == "accepted_right").sum())
            print(f"    Stage 1 accepts    {len(acc):>4}  precision {r / len(acc):.1%}  "
                  f"({len(acc) - r} false accept(s))")
        src = acc[acc["outcome"] == "accepted_right"]["accepted_sources"].fillna("?").value_counts().head(6)
        if len(src):
            print("    accepted-right by sources: " + ", ".join(f"{k} {v}" for k, v in src.items()))
        if b == "correct":
            print("    (a correct plant should be accepted_right; flagged_wrong / accepted_wrong are false moves)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-od", action="store_true",
                    help="score stage2_candidates_reranked_noOD.parquet")
    ap.add_argument("--bins", default="corrections,correct",
                    help="holdout bins to score (default: corrections,correct)")
    ap.add_argument("--cohort", default=None, help="restrict to one cohort")
    ap.add_argument("--min-pop", type=int, default=C.MIN_POP_SERVED,
                    help=f"score only holdout plants serving MORE than this many "
                         f"residents (default {C.MIN_POP_SERVED:,}; 0 disables). "
                         f"Filters at scoring time -- the manifest is untouched.")
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

    # Population floor, applied to the frozen manifest at SCORING time only.
    # Never resample the holdout to do this: a subset of a fixed sample stays
    # comparable round over round, a new draw does not. Compare against a
    # pre-filter score with --min-pop 0.
    man = C.apply_population_filter(man, "12 holdout", min_pop=args.min_pop)

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

    band_rows = {}   # bin -> (plant ids scored, per-plant outcome Series)

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
            # capped to the SLURM allocation -- a bare duckdb.connect() sizes itself
            # from the NODE'S physical RAM and gets OOM-killed instead of spilling.
            # See config.duckdb_connect (added after 08 died at 32G, 2026-09-24).
            con = C.duckdb_connect()
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

        # per-plant: is the re-ranked #1 the true parcel? (for the band split)
        hit1 = c[c["rerank_rank"] == 1].groupby("CWNS_ID")["is_true"].max()
        band_rows["corrections"] = (held["CWNS_ID"], hit1)

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
            band_rows["correct"] = (s["CWNS_ID"],
                                    s.set_index("CWNS_ID")["trigger_reason"] != "none")
            no_parcel = int((flagged["trigger_reason"] == "no_parcel").sum())
            if no_parcel:
                print(f"\n  NOTE: {no_parcel} were flagged as no_parcel -- the "
                      f"reported point hit\n  no parcel at all. That is a "
                      f"Regrid coverage gap, not a model error,\n  and is "
                      f"arguably not a 'false move' in the same sense.")

    # =======================================================================
    # multi-source: Stage 1 may accept a non-CWNS site, which the tables above cannot see
    # =======================================================================
    if C.MULTI and summ is not None and "accepted_site_id" in summ.columns:
        held_all = man[man["bin"].isin([b for b in bins if b in ("corrections", "correct")])]
        if len(held_all):
            import seed_sites as SS
            sites, _ = SS.load_sites(states=sorted(summ["STATE_CODE"].dropna().unique()),
                                     cwns_ids=held_all["CWNS_ID"])
            outcomes = multi_source_outcomes(held_all, truth, summ, cand, dict(zip(sites["site_id"], sites["ll_uuids"])))
            print_multi_source(outcomes)
            band_rows["corrections"] = (held_all.loc[held_all["bin"] == "corrections", "CWNS_ID"],
                                        outcomes[outcomes["bin"] == "corrections"].set_index("CWNS_ID")["outcome"]
                                        .isin(["accepted_right", "moved_right"]))

    # =======================================================================
    # by population band (since the floor dropped to 100, 2026-10-05)
    # =======================================================================
    # Plants serving 100-1,000 are newly admitted and expected to be harder.
    # Every headline above, split by band. Small n per band: read the counts.
    if band_rows:
        print(f"\n{'=' * 66}\nBY POPULATION BAND\n{'=' * 66}")
        pops = C.population_served()
        if "corrections" in band_rows:
            ids, hit1 = band_rows["corrections"]
            b = ids.map(lambda i: C.pop_band(pops.get(i)))
            print("  corrections bin -- re-ranked #1 is the true parcel (all plants):")
            for band, grp in ids.groupby(b):
                n = len(grp)
                k = int(hit1.reindex(grp).fillna(False).astype(bool).sum())
                print(f"    {band:<12} {k:3d}/{n:<3d} = {k / n if n else float('nan'):6.1%}")
        if "correct" in band_rows:
            ids, flag = band_rows["correct"]
            b = ids.map(lambda i: C.pop_band(pops.get(i)))
            print("  correct bin -- flagged by Stage 1 (every flag is a false one):")
            for band, grp in ids.groupby(b):
                n = len(grp)
                k = int(flag.reindex(grp).fillna(False).astype(bool).sum())
                print(f"    {band:<12} {k:3d}/{n:<3d} = {k / n if n else float('nan'):6.1%}")

    # =======================================================================
    # OSM-confirmed rule (config.STAGE1_OSM_PASS)
    # =======================================================================
    # The independent check on a rule the model does not judge: of holdout
    # plants passed because their reported parcel carries an OSM wastewater
    # tag, how many are actually correct -- and how many known-misplaced
    # plants the rule kept where they were.
    if summ is not None and "stage1_route" in summ.columns:
        print(f"\n{'=' * 66}\nOSM-CONFIRMED RULE -- plants passed without Stage 1\n{'=' * 66}")
        routed = summ.set_index("CWNS_ID")["stage1_route"]
        for b in ("correct", "corrections"):
            ids = man.loc[man["bin"] == b, "CWNS_ID"]
            r = routed.reindex(ids)
            n_osm = int((r == "osm_confirmed").sum())
            print(f"  {b:12s}: {n_osm} of {int(r.notna().sum())} scored plants passed by the OSM rule")
        n_bad = int((routed.reindex(man.loc[man["bin"] == "corrections", "CWNS_ID"])
                     == "osm_confirmed").sum())
        if n_bad:
            print(f"  -> {n_bad} known-MISPLACED plant(s) kept at their reported location "
                  f"by the rule. That is the rule's error rate; worth a look.")

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
