"""
07_train_stage2b.py
=====================
Trains Stage 2b: the OD-aware final-candidate-selection model (design
settled 2026-08-21). Binary classifier reused as a ranker -- same pattern
as Stage 1/2a: predict P(this is the right parcel), rank candidates by that
score at inference time. Mirrors 03_train_stage1.py's structure closely.

GROUP-AWARE CV (the one real structural difference from 03/04): Stage 2b's
training table pairs each corrections-bin plant's WRONG (reported) and
RIGHT (corrected) location as two separate rows when both are available.
A plain StratifiedKFold/spatial_cluster_folds split could put one of a
plant's two rows in train and the other in test -- the model would then be
evaluated on a plant it already partly saw. Fixed here with:
  - Standard CV: GroupKFold on CWNS_ID (sklearn), not StratifiedKFold --
    guarantees both of a plant's rows land in the same fold.
  - Spatial CV: a local per-plant variant of model_utils.spatial_cluster_folds
    -- clusters ONCE per plant (using the mean of that plant's row
    coordinates, so a plant with both a wrong and right location still gets
    exactly one cluster assignment), then assigns every row belonging to
    that plant to whichever fold its cluster lands in.

Changes 2026-08-24:
  - Fixed a TypeError that killed every run at the final-fit split: groups is
    a pyarrow-backed column, so .unique() returns an ArrowExtensionArray that
    train_test_split cannot index with a numpy integer array.
  - x_5070/y_5070 are no longer model features. They are fold-construction
    inputs; left in X they let the model memorize geography, which inflates
    standard (grouped) CV relative to spatial CV.
  - All-NaN columns are dropped before fitting. Three OD classes (chlorine
    contact, drying bed, oxidation pond) arrive empty by design -- the
    detector does not cover them reliably -- and were producing an imputer
    warning on every fold plus three columns of pure noise.
  - --no-od ablation, --seed, and tuning-spread reporting. See the notes on
    each below; all three exist because n is in the hundreds and a single
    best_score_ from a 20-config search invites over-reading.

Usage:
    python 07_train_stage2b.py [--n-iter 20] [--n-jobs -1] [--seed 42]
    python 07_train_stage2b.py --no-od      # non-OD baseline, _noOD artifacts
"""
import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.model_selection import train_test_split, GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from holdout import exclude_holdout
from model_utils import build_preprocessor, youden_threshold, compute_specificity

# Same drop list as Stage 1/2a, minus the corrections-specific parcel-uuid
# columns that never made it this far (already consumed during assembly in
# 06_build_stage2b_training.py).
#
# x_5070/y_5070 are NOT in this list because coords are read off s2b before
# X is built -- they are dropped from X explicitly further down instead.
#
# "owner" added 2026-08-25 from inspect_model_features.py. Stage 1 and 2a never
# needed it listed because add_name_matching() consumes the raw owner text into
# is_municipal/owner_water/sd_match/place_match/county_match and drops the
# column itself. 06_build_stage2b_training.py doesn't call add_name_matching at
# all, so on this path the raw text survived and was one-hot encoded: 226 levels
# out of the deployed model's 306 total, i.e. three quarters of Stage 2b's
# categorical feature space is owner-name strings, almost all of them novel at
# inference. Same trap the geography dummies were (master reference S3), on a
# column with far higher cardinality.
#
# Dropping it leaves Stage 2b with NEITHER the raw text nor the name-match
# booleans, since 06 never derives them -- that gap is real and worth closing,
# but wiring add_name_matching into 06 changes Stage 2b's feature set on
# purpose rather than removing something that was never meant to be there, so
# it is left as a separate decision.
DROP_COLS = [
    "ll_uuid", "h3_index_9", "h3_res9", "state", "geoid",
    "county_geoid", "LATITUDE", "LONGITUDE", "n_parcels",
    "pct_lbcs_activity_known", "pct_owner_known",
    "subdivision", "place", "county", "zoning_type",
    "FACILITY_ID", "DISCHARGE_TYPE", "PRESENT_DISCHARGE_PERCENTAGE",
    "PROJECTED_DISCHARGE_PERCENTAGE", "DISCHARGES_TO_CWNSID", "STATE_CODE",
    "owner",
]
# DISCHARGES_TO renamed to DISCHARGES_TO_CWNSID 2026-08-26, matching
# DISCHARGES.csv's repaired header (CWNSDatabaseDictionaryJanuary2025.xlsx's
# real column name -- see 02_feature_engineering.py's build_discharge_features).
# These raw discharge/STATE_CODE columns have never actually reached this
# script -- build_discharge_features' join matched zero rows until the
# 2026-08-26 fix, and even now only the 7 aggregate columns (surface_water_
# discharge etc., not in this list, and not currently merged into Stage 2b's
# frame at all -- see master reference S4 item 5) would flow through 06. Kept
# here defensively in case that changes.


def spatial_cluster_folds_grouped(coords: pd.DataFrame, groups: pd.Series,
                                   n_splits: int = 5, random_state: int = 123):
    """Group-aware variant of model_utils.spatial_cluster_folds -- clusters
    once per GROUP (plant), not once per row, so a plant contributing two
    rows (its wrong and right locations) always gets both assigned to the
    same fold. See module docstring."""
    per_group = coords.copy()
    per_group["_group"] = groups.to_numpy()
    group_coords = per_group.groupby("_group")[["x_5070", "y_5070"]].mean()

    km = KMeans(n_clusters=n_splits, random_state=random_state, n_init=10)
    group_cluster = pd.Series(
        km.fit_predict(group_coords.to_numpy()), index=group_coords.index)

    row_cluster = groups.map(group_cluster).to_numpy()
    idx = np.arange(len(coords))
    folds = []
    for c in range(n_splits):
        test_idx = idx[row_cluster == c]
        train_idx = idx[row_cluster != c]
        folds.append((train_idx, test_idx))
    return folds


def report_search_spread(search, label: str) -> None:
    """best_score_ is the MAXIMUM over n_iter configurations scored on the same
    folds. At n in the hundreds that maximum is optimistically biased, and one
    number invites over-reading. Print the spread across configs and the best
    config's own fold-to-fold sd so the noise floor is visible next to it."""
    cv = pd.DataFrame(search.cv_results_)
    print(f"  Best {label} ROC AUC: {search.best_score_:.4f}")
    print(f"    across {len(cv)} configs: mean {cv['mean_test_score'].mean():.4f}, "
          f"sd {cv['mean_test_score'].std():.4f}, "
          f"range {cv['mean_test_score'].min():.4f}-{cv['mean_test_score'].max():.4f}")
    print(f"    best config's own fold sd: "
          f"{cv.loc[search.best_index_, 'std_test_score']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42,
                    help="random_state for the RF and the train/test split. At "
                         "n in the hundreds a single seed's test metrics are "
                         "noisy -- run 3-5 seeds and report the spread before "
                         "trusting any specific figure.")
    ap.add_argument("--no-od", action="store_true",
                    help="train without any od_* feature. Establishes the non-OD "
                         "baseline so OD's contribution can be measured as the "
                         "detector improves -- NOT a test of whether Stage 2b "
                         "should exist. Writes _noOD artifacts so it never "
                         "overwrites the real model.")
    ap.add_argument("--allow-no-holdout", action="store_true",
                    help="proceed even if the holdout manifest is missing. Only "
                         "for deliberate pre-holdout runs -- normally a missing "
                         "manifest should stop the job.")
    args = ap.parse_args()

    tag = "_noOD" if args.no_od else ""

    C.ensure_dirs()
    print("=== 07_train_stage2b.py: Stage 2b Model Training (Random Forest) ===\n")
    if args.no_od:
        print("*** --no-od: OD ABLATION RUN. Artifacts get a _noOD suffix. ***\n")

    print("Loading Stage 2b training data...")
    s2b = pd.read_parquet(C.FEATURES_OUTPUT_DIR / "16_stage2b_training.parquet")
    s2b = exclude_holdout(s2b, "stage2b", allow_missing=args.allow_no_holdout)
    print(f"  Rows: {len(s2b)}")
    print(f"  Label=1 (right): {(s2b['label'] == 1).sum()}")
    print(f"  Label=0 (wrong): {(s2b['label'] == 0).sum()}")
    n_distinct_plants = s2b["CWNS_ID"].nunique()
    n_paired = s2b.groupby("CWNS_ID").size().eq(2).sum()
    print(f"  Distinct plants: {n_distinct_plants}  ({n_paired} contribute both a "
          f"wrong and right row, {n_distinct_plants - n_paired} contribute only one)")
    if len(s2b) == 0:
        raise ValueError("No labeled rows found -- check 06_build_stage2b_training.py output")
    print("  NOTE: this is a MUCH smaller training set than Stage 1/2a (hundreds of rows,\n"
          "  not thousands) -- expect higher variance in CV scores and test metrics than\n"
          "  those models. Treat single-run numbers with real caution; consider repeating\n"
          "  with different --seed values before trusting a specific figure.")
    if n_paired < n_distinct_plants / 2:
        print(f"  CAUTION: only {n_paired} of {n_distinct_plants} plants contribute a "
              f"within-plant contrast.\n"
              f"  Unpaired rows carry no wrong-vs-right comparison for their own plant, so\n"
              f"  a good AUC here can reflect plant-vs-plant separability rather than the\n"
              f"  candidate-ranking task this model actually performs at inference.")

    groups = s2b["CWNS_ID"]
    coords = s2b[["x_5070", "y_5070"]].reset_index(drop=True)

    # ---- Feature selection ----
    suffix_artifacts = [c for c in s2b.columns if c.endswith(("_x", "_y"))
                        and c not in ("x_5070", "y_5070")]
    if suffix_artifacts:
        print(f"  WARNING: dropping apparent merge-suffix artifact columns: {suffix_artifacts}")
    X = s2b.drop(columns=[c for c in DROP_COLS if c in s2b.columns] +
                 suffix_artifacts + ["label", "CWNS_ID"], errors="ignore")

    # Projected coordinates are fold-construction inputs, not features. Left in
    # X they let the model memorize geography: spatial CV holds out whole
    # regions so it mostly resists this, but standard (grouped) CV does not,
    # which widens the gap between the two for reasons that have nothing to do
    # with spatial generalization.
    X = X.drop(columns=["x_5070", "y_5070"], errors="ignore")

    # OD classes the detector does not cover reliably (chlorine contact, drying
    # bed, oxidation pond) arrive all-NaN. They carry no information and make
    # the median imputer warn on every fold. Dropped by emptiness rather than by
    # name so this stays correct as the detector's class inventory changes.
    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        print(f"  Dropping {len(all_nan)} all-NaN columns: {sorted(all_nan)}")
        X = X.drop(columns=all_nan)

    if args.no_od:
        od_cols = [c for c in X.columns if c.startswith("od_")]
        X = X.drop(columns=od_cols)
        print(f"  --no-od: dropped {len(od_cols)} OD features")

    if "place_match" in X.columns:
        X["place_match"] = X["place_match"].fillna(False)
    X = X.reset_index(drop=True)
    y = s2b["label"].astype(int).reset_index(drop=True)
    groups = groups.reset_index(drop=True)

    print(f"  Model columns: {X.shape[1]}")

    # ---- Class weights ----
    n_right = int((y == 1).sum())
    n_wrong = int((y == 0).sum())
    n_total = len(y)
    class_weight = {1: n_wrong / n_total, 0: n_right / n_total}
    print(f"  Class weights -- Right: {class_weight[1]:.3f} | Wrong: {class_weight[0]:.3f}")

    # ---- CV fold construction (group-aware, see module docstring) ----
    print("\nSetting up cross-validation folds...")
    n_splits = min(5, groups.nunique())
    if n_splits < 5:
        print(f"  WARNING: only {groups.nunique()} distinct plants -- reducing "
              f"n_splits to {n_splits}")

    spatial_folds = spatial_cluster_folds_grouped(coords, groups, n_splits=n_splits,
                                                  random_state=123)
    for i, (tr, te) in enumerate(spatial_folds, 1):
        print(f"  Spatial fold {i} train -- Right: {(y.iloc[tr] == 1).sum()}, "
              f"Wrong: {(y.iloc[tr] == 0).sum()}")

    gkf = GroupKFold(n_splits=n_splits)
    standard_folds = list(gkf.split(X, y, groups=groups))
    print(f"  Spatial folds: {len(spatial_folds)}  |  Standard (grouped) folds: {len(standard_folds)}")

    # ---- Pipeline + search space ----
    param_dist = {
        "model__max_features": np.linspace(0.1, 1.0, 20),
        "model__min_samples_leaf": np.arange(1, 21),
    }

    def make_pipeline():
        return Pipeline([
            ("prep", build_preprocessor(X)),
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
            n_jobs=args.n_jobs, refit=False,
        )
        search.fit(X, y)
        elapsed = (time.time() - t0) / 60
        print(f"  {label} CV complete in {elapsed:.1f} minutes")
        report_search_spread(search, label)
        return search

    spatial_search = run_cv(spatial_folds, "SPATIAL")
    standard_search = run_cv(standard_folds, "STANDARD (grouped)")

    compare_cv = pd.DataFrame([
        dict(cv="Spatial", roc_auc=spatial_search.best_score_),
        dict(cv="Standard", roc_auc=standard_search.best_score_),
    ]).sort_values("roc_auc", ascending=False)
    print(f"\nCV Comparison:\n{compare_cv}")
    compare_cv.to_parquet(C.MODELS_DIR / f"stage2b_cv_comparison{tag}.parquet", index=False)

    # ---- Final fit on held-out test split ----
    # Group-aware split here too: a plant's rows must not straddle
    # train/test, same reasoning as the CV folds above. sklearn's
    # train_test_split has no native groups= support, so split at the
    # PLANT level directly, then take all rows for the chosen plants.
    print(f"\n{'=' * 40}\nFitting final model on full training data...\n{'=' * 40}")
    # np.asarray(... .astype("object")) is load-bearing: CWNS_ID is a
    # pyarrow-backed string column, so groups.unique() returns an
    # ArrowExtensionArray, and train_test_split's internal integer-array
    # indexing raises "only integer scalar arrays can be converted to a
    # scalar index" against it.
    unique_plants = np.asarray(groups.astype("object").unique())
    plant_labels = (s2b.groupby("CWNS_ID")["label"].max()
                       .reindex(unique_plants).to_numpy())  # for stratification
    train_plants, test_plants = train_test_split(
        unique_plants, test_size=0.25, stratify=plant_labels, random_state=args.seed)
    train_mask = groups.isin(train_plants).to_numpy()
    test_mask = groups.isin(test_plants).to_numpy()
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    best_params = spatial_search.best_params_
    final_pipeline = make_pipeline()
    final_pipeline.set_params(**best_params)
    final_pipeline.fit(X_train, y_train)

    y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]
    y_pred = final_pipeline.predict(X_test)

    print("Test set metrics:")
    print(f"  roc_auc     : {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"  accuracy    : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  sensitivity : {(y_pred[y_test == 1] == 1).mean():.4f}")
    print(f"  specificity : {compute_specificity(y_test.to_numpy(), y_pred, pos_label=1):.4f}")
    print(f"  brier_class : {brier_score_loss(y_test, y_pred_proba):.4f}")
    print(f"  (test set: {len(y_test)} rows from {len(test_plants)} plants -- small, "
          f"expect noisy metrics)")

    print("\nConfusion matrix (rows=truth, cols=predicted; 1=Right, 0=Wrong):")
    print(confusion_matrix(y_test, y_pred))

    # ---- Feature importance ----
    # Worth reading closely on this model specifically: every label=0 row's OD
    # features come from 01b and every label=1 row's from 01c. If a run-level
    # artifact (od_ran, od_has_detection coverage) sits at the top of this list
    # rather than a substantive parcel or detection feature, the model may be
    # reading provenance rather than evidence.
    print("\nComputing feature importance...")
    perm = permutation_importance(final_pipeline, X_test, y_test, n_repeats=10,
                                   random_state=args.seed, n_jobs=args.n_jobs,
                                   scoring="roc_auc")
    rf_importance = pd.DataFrame({
        "Variable": X_test.columns, "Importance": perm.importances_mean,
        "Importance_std": perm.importances_std,
    }).sort_values("Importance", ascending=False)
    print("Top 20 features:")
    print(rf_importance.head(20).to_string(index=False))
    rf_importance.to_parquet(C.MODELS_DIR / f"stage2b_rf_importance{tag}.parquet", index=False)

    # ---- Threshold analysis ----
    print("\nRunning threshold analysis...")
    optimal_threshold, roc_df = youden_threshold(y_test.to_numpy(), y_pred_proba)
    print(f"  Optimal Stage 2b threshold (Youden J): {optimal_threshold:.3f}")
    roc_df.to_parquet(C.MODELS_DIR / f"stage2b_threshold_analysis{tag}.parquet", index=False)

    # ---- Save model ----
    print("\nSaving model...")
    model_path = C.MODELS_DIR / f"stage2b_rf_model{tag}.joblib"
    joblib.dump(dict(pipeline=final_pipeline, feature_columns=list(X.columns),
                     class_labels={1: "Right", 0: "Wrong"},
                     seed=args.seed, no_od=args.no_od), model_path)
    joblib.dump(optimal_threshold, C.MODELS_DIR / f"stage2b_optimal_threshold{tag}.joblib")
    print(f"  {model_path.name} saved: {model_path.stat().st_size / 1e6:.1f} MB")

    print("\n=== Stage 2b training complete ===")
    print(f"\nFINAL RESULTS SUMMARY")
    print(f"  Seed: {args.seed}  |  OD features: {'EXCLUDED' if args.no_od else 'included'}")
    print(f"  Test roc_auc: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"  Optimal threshold: {optimal_threshold:.3f}")
    print(f"  CV comparison:\n{compare_cv}")


if __name__ == "__main__":
    main()