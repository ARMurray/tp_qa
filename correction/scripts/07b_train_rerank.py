"""
07b_train_rerank.py
===================
Trains the Stage 2a RE-RANKER on 17_rerank_training.parquet: one row per
candidate parcel, 1 for the plant's true parcel and 0 for the ~19
competitors Stage 2a also surfaced.

HOW THIS DIFFERS FROM 07_train_stage2b.py
    Same algorithm, same preprocessor, same group-aware CV. What changed is
    the training distribution, and that is the whole point -- see
    06b_build_rerank_training.py's docstring for why 06's reported-vs-
    corrected pairs could never reveal OD's contribution.

    Three concrete differences:

    1. THE METRIC. ROC AUC over pooled rows answers "can this model tell a
       true parcel from a competitor anywhere in the dataset". The task is
       narrower: within ONE plant's 20 candidates, does the true parcel come
       first. A model can post 0.95 AUC and still rank the true parcel
       second in every pool. Recall@1 and recall@5 are computed per plant on
       the test split and are the numbers to read. AUC is kept for the
       hyperparameter search only, where it is a serviceable proxy and much
       cheaper to compute inside RandomizedSearchCV.

    2. LEAKING COLUMNS. true_ll_uuid is the label. pool_has_positive is
       subtler and just as fatal: it says whether ANY row in this plant's
       pool is positive, so a model can learn "pool_has_positive is False ->
       predict 0 for all twenty". That is unavailable at inference and would
       inflate every metric here. Both are dropped, loudly.

    3. CLASS BALANCE. 07 trains near 60/40. Here it is roughly 1:19 by
       construction, and worse once pools with no positive are counted.
       Weights are derived from the observed balance rather than assumed.

WHAT A GOOD RESULT LOOKS LIKE
    Stage 2a alone is the baseline: its recall@1 on the same test plants is
    printed alongside. Measured on the holdout (diagnose_candidate_od.py,
    2026-09-08), ranking by OD confidence with no model at all reached 35.7%
    against a 5% random baseline. A re-ranker that cannot beat Stage 2a's
    own ordering is not worth deploying, however good its AUC looks.

Artifacts: rerank_rf_model.joblib, rerank_optimal_threshold.joblib,
           rerank_cv_comparison.parquet, rerank_rf_importance.parquet
           (all with a _noOD suffix under --no-od)

Usage:
    python 07b_train_rerank.py
    python 07b_train_rerank.py --no-od          # ablation baseline
    python 07b_train_rerank.py --seed 7
    python 07b_train_rerank.py --drop-no-positive-pools
"""
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from holdout import exclude_holdout
from model_utils import build_preprocessor

# Identity/label columns plus the two that leak. Mirrors 07's DROP_COLS and
# adds the re-rank-specific ones.
DROP_COLS = [
    "ll_uuid", "h3_index_9", "h3_res9", "state", "geoid",
    "county_geoid", "LATITUDE", "LONGITUDE", "n_parcels",
    "pct_lbcs_activity_known", "pct_owner_known",
    "subdivision", "place", "county", "zoning_type",
    "FACILITY_ID", "DISCHARGE_TYPE", "PRESENT_DISCHARGE_PERCENTAGE",
    "PROJECTED_DISCHARGE_PERCENTAGE", "DISCHARGES_TO", "STATE_CODE",
    "centroid_lat", "centroid_lng",
]
LEAKING_COLS = ["true_ll_uuid", "pool_has_positive"]


def spatial_cluster_folds_grouped(coords, groups, n_splits=5, random_state=123):
    """Group-aware spatial folds -- identical to 07's helper. Clusters on
    per-plant mean coordinates so a plant's candidates never straddle a
    fold boundary."""
    from sklearn.cluster import KMeans
    per_group = coords.copy()
    per_group["_group"] = groups.to_numpy()
    group_coords = per_group.groupby("_group")[["x_5070", "y_5070"]].mean()
    km = KMeans(n_clusters=n_splits, random_state=random_state, n_init=10)
    group_cluster = pd.Series(km.fit_predict(group_coords.to_numpy()),
                              index=group_coords.index)
    row_cluster = groups.map(group_cluster).to_numpy()
    idx = np.arange(len(groups))
    return [(idx[row_cluster != c], idx[row_cluster == c]) for c in range(n_splits)]


def recall_at_k(df: pd.DataFrame, score_col: str, k: int = 1) -> tuple[float, int, int]:
    """Per-plant recall@k: of plants whose pool CONTAINS the true parcel, how
    often is it in the top k by score_col.

    Pools with no positive are excluded from the denominator -- no ranking
    can find a parcel that is not there, so including them would measure
    candidate generation rather than ranking. They are reported separately.
    """
    have = df.groupby("CWNS_ID")["label"].transform("max") == 1
    sub = df[have]
    if sub.empty:
        return float("nan"), 0, 0
    ranked = sub.sort_values(["CWNS_ID", score_col], ascending=[True, False])
    topk = ranked.groupby("CWNS_ID").head(k)
    hits = int(topk.groupby("CWNS_ID")["label"].max().sum())
    n = topk["CWNS_ID"].nunique()
    return hits / n, hits, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42,
                    help="a single seed's test metrics are noisy at this n -- "
                         "run 3-5 and report the spread before trusting one")
    ap.add_argument("--no-od", action="store_true",
                    help="train without any od_* feature. THE key ablation "
                         "here: if recall@1 is unchanged, OD is contributing "
                         "nothing at the candidate level and the whole 01e "
                         "fetch cost is unjustified. Writes _noOD artifacts.")
    ap.add_argument("--drop-no-positive-pools", action="store_true",
                    help="train only on pools containing the true parcel. "
                         "Off by default: near-miss pools are real at "
                         "inference and teach what a plausible-but-wrong "
                         "parcel looks like.")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--allow-no-holdout", action="store_true")
    args = ap.parse_args()

    tag = "_noOD" if args.no_od else ""
    C.ensure_dirs()
    print("=== 07b_train_rerank.py: Stage 2a Re-ranker ===\n")
    if args.no_od:
        print("*** --no-od: OD ABLATION. Artifacts get a _noOD suffix. ***\n")

    path = C.FEATURES_OUTPUT_DIR / "17_rerank_training.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found. Run 06b_build_rerank_training.py first.")
        sys.exit(2)

    print("Loading re-rank training data...")
    df = pd.read_parquet(path)
    df = exclude_holdout(df, "rerank", allow_missing=args.allow_no_holdout)
    df["CWNS_ID"] = df["CWNS_ID"].astype(str)

    n_plants = df["CWNS_ID"].nunique()
    pools_with_pos = int(df.groupby("CWNS_ID")["label"].max().sum())
    print(f"  Rows: {len(df)}  |  plants: {n_plants}")
    print(f"  Label=1 (true parcel): {int((df['label'] == 1).sum())}")
    print(f"  Label=0 (competitor) : {int((df['label'] == 0).sum())}")
    print(f"  Pools containing the true parcel: {pools_with_pos}/{n_plants} "
          f"({pools_with_pos / n_plants:.0%})")
    if pools_with_pos == 0:
        raise ValueError("No positive rows at all -- 06b's true-parcel join failed.")

    if args.drop_no_positive_pools:
        keep = df.groupby("CWNS_ID")["label"].transform("max") == 1
        print(f"  --drop-no-positive-pools: {len(df)} -> {int(keep.sum())} rows")
        df = df[keep].reset_index(drop=True)

    # ---- group-aware plant-level split -----------------------------------
    # Split by PLANT, never by row: a pool straddling train/test lets the
    # model see 19 of a plant's 20 candidates during training and then rank
    # the twentieth, which is not the inference task.
    rng = np.random.default_rng(args.seed)
    plants = df["CWNS_ID"].unique()
    rng.shuffle(plants)
    n_test = max(1, int(len(plants) * args.test_frac))
    test_plants = set(plants[:n_test])
    is_test = df["CWNS_ID"].isin(test_plants)
    train_df, test_df = df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
    print(f"\n  Split by plant: {len(plants) - n_test} train / {n_test} test plant(s)")

    # ---- features --------------------------------------------------------
    leaking = [c for c in LEAKING_COLS if c in df.columns]
    if leaking:
        print(f"  Dropping leaking columns: {leaking}")

    def make_X(frame):
        X = frame.drop(columns=[c for c in DROP_COLS + LEAKING_COLS + ["label", "CWNS_ID"]
                                if c in frame.columns], errors="ignore")
        X = X.drop(columns=["x_5070", "y_5070"], errors="ignore")
        suffix_artifacts = [c for c in X.columns if c.endswith(("_x", "_y"))]
        if suffix_artifacts:
            X = X.drop(columns=suffix_artifacts)
        if "place_match" in X.columns:
            X["place_match"] = X["place_match"].fillna(False)
        return X

    X_train, X_test = make_X(train_df), make_X(test_df)
    all_nan = [c for c in X_train.columns if X_train[c].isna().all()]
    if all_nan:
        print(f"  Dropping {len(all_nan)} all-NaN columns: {sorted(all_nan)}")
        X_train, X_test = X_train.drop(columns=all_nan), X_test.drop(columns=all_nan)

    if args.no_od:
        od_cols = [c for c in X_train.columns if c.startswith("od_")]
        X_train, X_test = X_train.drop(columns=od_cols), X_test.drop(columns=od_cols)
        print(f"  --no-od: dropped {len(od_cols)} OD features")

    X_test = X_test.reindex(columns=X_train.columns)
    y_train = train_df["label"].astype(int)
    y_test = test_df["label"].astype(int)
    groups = train_df["CWNS_ID"]
    coords = train_df[["x_5070", "y_5070"]].reset_index(drop=True)
    print(f"  Model columns: {X_train.shape[1]}")

    n_pos, n_neg = int((y_train == 1).sum()), int((y_train == 0).sum())
    class_weight = {1: n_neg / len(y_train), 0: n_pos / len(y_train)}
    print(f"  Class balance 1:{n_neg / max(n_pos, 1):.0f}  |  weights "
          f"true {class_weight[1]:.3f} / competitor {class_weight[0]:.3f}")

    # ---- CV --------------------------------------------------------------
    print("\nSetting up cross-validation folds...")
    n_splits = min(5, groups.nunique())
    spatial_folds = spatial_cluster_folds_grouped(coords, groups, n_splits=n_splits)
    standard_folds = list(GroupKFold(n_splits=n_splits).split(X_train, y_train, groups=groups))
    print(f"  Spatial folds: {len(spatial_folds)}  |  Standard (grouped): {len(standard_folds)}")

    param_dist = {
        "model__max_features": np.linspace(0.1, 1.0, 20),
        "model__min_samples_leaf": np.arange(1, 21),
    }

    def make_pipeline():
        return Pipeline([
            ("prep", build_preprocessor(X_train)),
            ("model", RandomForestClassifier(
                n_estimators=1000, class_weight=class_weight,
                random_state=args.seed, n_jobs=args.n_jobs)),
        ])

    def run_cv(folds, label):
        print(f"\n{'=' * 40}\nTuning with {label} cross-validation...\n{'=' * 40}")
        t0 = time.time()
        search = RandomizedSearchCV(
            make_pipeline(), param_distributions=param_dist, n_iter=args.n_iter,
            scoring="roc_auc", cv=folds, random_state=args.seed,
            n_jobs=args.n_jobs, refit=False)
        search.fit(X_train, y_train)
        print(f"  {label} CV complete in {(time.time() - t0) / 60:.1f} minutes")
        cv = pd.DataFrame(search.cv_results_)
        print(f"  Best {label} ROC AUC: {search.best_score_:.4f}")
        print(f"    across {len(cv)} configs: mean {cv['mean_test_score'].mean():.4f}, "
              f"sd {cv['mean_test_score'].std():.4f}")
        return search

    spatial_search = run_cv(spatial_folds, "SPATIAL")
    standard_search = run_cv(standard_folds, "STANDARD (grouped)")
    compare_cv = pd.DataFrame([
        dict(cv="Spatial", roc_auc=spatial_search.best_score_),
        dict(cv="Standard", roc_auc=standard_search.best_score_),
    ]).sort_values("roc_auc", ascending=False)
    print(f"\nCV Comparison:\n{compare_cv}")
    compare_cv.to_parquet(C.MODELS_DIR / f"rerank_cv_comparison{tag}.parquet", index=False)

    # ---- final fit -------------------------------------------------------
    print(f"\n{'=' * 40}\nFitting final model...\n{'=' * 40}")
    pipe = make_pipeline()
    pipe.set_params(**standard_search.best_params_)
    pipe.fit(X_train, y_train)

    test_df = test_df.copy()
    test_df["rerank_score"] = pipe.predict_proba(X_test)[:, 1]

    # ---- THE metric ------------------------------------------------------
    print(f"\n{'=' * 40}\nRanking performance on held-out plants\n{'=' * 40}")
    for k in (1, 3, 5):
        r_new, h_new, n_new = recall_at_k(test_df, "rerank_score", k)
        r_old, h_old, _ = recall_at_k(test_df, "stage2_prob_correct", k)
        delta = r_new - r_old
        print(f"  recall@{k}:  Stage 2a {r_old:.1%} ({h_old}/{n_new})   "
              f"re-rank {r_new:.1%} ({h_new}/{n_new})   {delta:+.1%}")

    n_test_plants = test_df["CWNS_ID"].nunique()
    n_scoreable = int((test_df.groupby("CWNS_ID")["label"].max() == 1).sum())
    print(f"\n  Denominator is the {n_scoreable} test plant(s) whose pool contains")
    print(f"  the true parcel, of {n_test_plants} total. The other "
          f"{n_test_plants - n_scoreable} cannot be")
    print(f"  found by ANY ranking -- that gap belongs to candidate generation")
    print(f"  (see 08_diagnose_candidate_coverage.py), not to this model.")
    if n_scoreable < 30:
        print(f"\n  CAUTION: {n_scoreable} plants gives a standard error near "
              f"+/-{100 * 0.5 / np.sqrt(max(n_scoreable, 1)):.0f}pp. Differences")
        print(f"  smaller than that are noise. Re-run with several --seed values.")

    # ---- importance + artifacts -----------------------------------------
    print("\nComputing feature importance...")
    perm = permutation_importance(pipe, X_test, y_test, n_repeats=5,
                                  random_state=args.seed, n_jobs=args.n_jobs,
                                  scoring="roc_auc")
    imp = pd.DataFrame({
        "Variable": X_train.columns,
        "Importance": perm.importances_mean,
        "Importance_std": perm.importances_std,
    }).sort_values("Importance", ascending=False)
    print(imp.head(20).to_string(index=False))
    imp.to_parquet(C.MODELS_DIR / f"rerank_rf_importance{tag}.parquet", index=False)

    joblib.dump({"pipeline": pipe, "feature_columns": list(X_train.columns)},
                C.MODELS_DIR / f"rerank_rf_model{tag}.joblib")
    print(f"\nSaved: rerank_rf_model{tag}.joblib")
    print("\nNext: 05b_rerank_candidates.py, then score the holdout.")


if __name__ == "__main__":
    main()
