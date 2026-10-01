"""
stage1_keyword_diagnostic.py
============================
Read-only: does Stage 1 undervalue a utility-sounding owner on the REPORTED
parcel?

    sbatch stage1_keyword_diagnostic.slurm
    -> correction/diagnostics/stage1_keyword_<host>_<stamp>.txt

WHY (2026-10-01)
    Round 4 reviewer note: "I see sewer keywords in parcels where Stage 1 had
    a very low score. If a reported location is on a parcel with an owner
    name that has words like sewer, sewage, water ... I would expect a very
    high Stage 1 score." In the 2026-09-30 Stage 1 model has_ww_keyword is
    not in the top 20 permutation importances.

    Two explanations predict different numbers, and this separates them:
      (a) the LABELS say the keyword is weak evidence -- incorrect plants are
          often reported at a utility's OTHER property (an office, a pump or
          lift station, a water plant), so 'owned by the sewer district' does
          not mean 'this is the treatment plant'. Then Correct rates among
          keyword parcels are not much higher than baseline, and the model is
          right to discount it.
      (b) the MODEL undervalues it -- the labels say keyword parcels are
          mostly Correct, but scores do not reflect that. Then predicted
          probability for keyword parcels sits well below their observed
          Correct rate.

SECTIONS
    1. Training labels: Correct rate with/without each owner-text feature
       (has_ww_keyword, owner_is_utility, owner_water, osm_ww, name match).
    2. The deployed model: mean predicted P(correct) vs the observed Correct
       rate for the same groups, on the training table. (Training-set scores
       are optimistic overall; the question is the GAP between groups.)
    3. Which words: the owner text of the reported parcel, keyword parcels
       only, split by label -- what 'utility owner but wrong location' looks
       like.
    4. Round 4: reviewed plants Stage 1 flagged whose reported parcel has a
       keyword owner, with the reviewer's verdict.

Writes nothing but its report.
"""
import datetime as dt
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from collect_diagnostics import Report, sec_environment

OUT_DIR = C.ROOT / "diagnostics"
S1_PATH = C.FEATURES_OUTPUT_DIR / "14_stage1_training.parquet"
PARCELS_PATH = C.FEATURES_OUTPUT_DIR / "10_parcel_features.parquet"
SUMMARY_PATH = C.DATA_DIR / "inference" / "plant_summary_reranked.parquet"
REVIEW_LOG = C.ROOT.parent / "review_app" / "data" / "outgoing" / "review_log_round4.parquet"

FLAGS = ["has_ww_keyword", "owner_is_utility", "owner_water", "owner_is_govt",
         "is_municipal", "osm_ww"]


def as_bool(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(str).str.lower().isin(["true", "1", "1.0"])


def sec_labels(r, s1):
    y = s1["class"].eq("Correct")
    print(f"  training rows: {len(s1)}  |  Correct {int(y.sum())} ({y.mean():.1%})  "
          f"Incorrect {int((~y).sum())}\n")
    print(f"  {'feature':<22}{'n True':>8}{'Correct|True':>14}{'n False':>9}{'Correct|False':>15}")
    for f in FLAGS + ["name_match>=0.5"]:
        if f == "name_match>=0.5":
            if "name_match_score" not in s1.columns:
                continue
            m = s1["name_match_score"].fillna(0) >= 0.5
        elif f in s1.columns:
            m = as_bool(s1[f])
        else:
            continue
        a, b = y[m], y[~m]
        print(f"  {f:<22}{int(m.sum()):>8}{a.mean() if len(a) else float('nan'):>14.1%}"
              f"{int((~m).sum()):>9}{b.mean() if len(b) else float('nan'):>15.1%}")
    print("\n  Read: if Correct|True is far above the baseline, the label supports the "
          "reviewer's intuition.")


def sec_model(r, s1):
    path = C.MODELS_DIR / "stage1_rf_model.joblib"
    bundle = joblib.load(path)
    X = s1.reindex(columns=bundle["feature_columns"])
    if "place_match" in X.columns:
        X["place_match"] = X["place_match"].fillna(False)
    p = bundle["pipeline"].predict_proba(X)[:, 1]
    thr = float(joblib.load(C.MODELS_DIR / "stage1_optimal_threshold.joblib"))
    y = s1["class"].eq("Correct").to_numpy()
    print(f"  model: {path.name}, threshold {thr:.3f} (below it = flagged as wrong)\n")
    print(f"  {'group':<28}{'n':>6}{'observed Correct':>18}{'mean P(correct)':>17}{'flagged':>9}")
    groups = {"all": np.ones(len(s1), bool)}
    for f in ("has_ww_keyword", "owner_is_utility", "osm_ww"):
        if f in s1.columns:
            groups[f"{f}=True"] = as_bool(s1[f]).to_numpy()
    if "has_ww_keyword" in s1.columns and "osm_ww" in s1.columns:
        groups["keyword, NOT osm_ww"] = (as_bool(s1["has_ww_keyword"])
                                         & ~as_bool(s1["osm_ww"])).to_numpy()
    for name, m in groups.items():
        if m.sum() == 0:
            continue
        print(f"  {name:<28}{int(m.sum()):>6}{y[m].mean():>18.1%}{p[m].mean():>17.3f}"
              f"{(p[m] < thr).mean():>9.1%}")
    print("\n  Read: a group whose mean P(correct) sits well BELOW its observed Correct "
          "rate is being undervalued by the model (explanation b).")


def sec_words(r, s1):
    if "ll_uuid" not in s1.columns or "has_ww_keyword" not in s1.columns:
        print("  14_stage1_training has no ll_uuid / has_ww_keyword -- skipped")
        return
    kw = s1[as_bool(s1["has_ww_keyword"]) | as_bool(s1.get("owner_is_utility", pd.Series(False, index=s1.index)))]
    con = C.duckdb_connect(spatial=False)
    con.register("want", kw[["ll_uuid"]].drop_duplicates())
    own = con.execute(f"""
        SELECT p.ll_uuid, p.owner, p.lbcs_activity, p.ll_gisacre
        FROM read_parquet('{PARCELS_PATH.as_posix()}') p JOIN want w USING (ll_uuid)
    """).df()
    con.close()
    kw = kw.merge(own.drop_duplicates("ll_uuid"), on="ll_uuid", how="left",
                  suffixes=("", "_p"))
    for cls in ("Incorrect", "Correct"):
        sub = kw[kw["class"] == cls]
        print(f"\n  {cls}: {len(sub)} plants whose reported parcel has a keyword / utility owner")
        acres = pd.to_numeric(sub.get("ll_gisacre_p", sub.get("ll_gisacre")), errors="coerce")
        if len(sub):
            print(f"    median parcel acreage: {acres.median():.1f}")
        for _, x in sub.head(25).iterrows():
            print(f"    {x['CWNS_ID']}  {str(x.get('owner') or '')[:60]:<60} "
                  f"{str(x.get('lbcs_activity_p', x.get('lbcs_activity', '')))[:22]}")


def sec_round4(r):
    if not REVIEW_LOG.exists():
        print(f"  {REVIEW_LOG} not found (pull the repo) -- skipped")
        return
    rv = pd.read_parquet(REVIEW_LOG)
    rv["CWNS_ID"] = rv["cwns_id"].astype(str)
    summ = pd.read_parquet(SUMMARY_PATH, columns=["CWNS_ID", "reported_ll_uuid",
                                                   "stage1_prob_correct", "trigger_reason"])
    summ["CWNS_ID"] = summ["CWNS_ID"].astype(str)
    d = rv.merge(summ, on="CWNS_ID", how="left")
    d = d[d["reported_ll_uuid"].notna()]
    con = C.duckdb_connect(spatial=False)
    con.register("want", d[["reported_ll_uuid"]].rename(columns={"reported_ll_uuid": "ll_uuid"}))
    pf = con.execute(f"""
        SELECT p.ll_uuid, p.owner, p.has_ww_keyword, p.owner_is_utility, p.osm_ww
        FROM read_parquet('{PARCELS_PATH.as_posix()}') p JOIN want w USING (ll_uuid)
    """).df().drop_duplicates("ll_uuid")
    con.close()
    d = d.merge(pf, left_on="reported_ll_uuid", right_on="ll_uuid", how="left")
    kw = d[as_bool(d["has_ww_keyword"]) | as_bool(d["owner_is_utility"])]
    print(f"  round 4 reviewed plants with a reported parcel: {len(d)}; "
          f"of those with a keyword/utility owner: {len(kw)}")
    if len(kw):
        print(f"  verdicts: {kw['plant_verdict'].value_counts().to_dict()}\n")
        for _, x in kw.sort_values("stage1_prob_correct").iterrows():
            print(f"    {x['CWNS_ID']}  P(correct)={x['stage1_prob_correct']:.3f}  "
                  f"{x['plant_verdict']:<25} {str(x['owner'])[:55]}")
        wrong_flag = kw[(kw["plant_verdict"] == "reported_correct")
                        & (kw["trigger_reason"] != "none")]
        print(f"\n  flagged by Stage 1 but the reported location was RIGHT: {len(wrong_flag)}")


def main():
    r = Report()
    r.p("tp_qa Stage 1 owner-keyword diagnostic -- read-only")
    r.section("0. ENVIRONMENT", sec_environment)
    s1 = pd.read_parquet(S1_PATH)
    s1["CWNS_ID"] = s1["CWNS_ID"].astype(str)
    r.section("1. TRAINING LABELS: Correct rate by owner-text feature", lambda r: sec_labels(r, s1))
    r.section("2. MODEL vs LABELS: is the keyword group undervalued?", lambda r: sec_model(r, s1))
    r.section("3. WHO THE KEYWORD OWNERS ARE, by label", lambda r: sec_words(r, s1))
    r.section("4. ROUND 4: flagged plants on keyword-owned parcels", sec_round4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = OUT_DIR / f"stage1_keyword_{platform.node().split('.')[0]}_{stamp}.txt"
    out.write_text(r.buf.getvalue(), encoding="utf-8")
    print(r.buf.getvalue())
    print(f"\nWRITTEN: {out}\n\nThen: cd /work/GRDVULN/tp_qa && git add correction/diagnostics/ "
          f"&& git commit -m 'stage1 keyword diagnostic' && git push")


if __name__ == "__main__":
    main()
