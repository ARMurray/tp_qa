"""
03_train_stage1.py
===================
Port of train_stage1.R. Trains the Stage 1 classifier: is the REPORTED
location correct or incorrect? See model_utils.py's docstring for the
porting decisions shared with 04_train_stage2.py (preprocessing, spatial
CV approximation, hyperparameter search method).

Model: sklearn RandomForestClassifier with an explicit class-weight dict
(closest direct port of ranger's class.weights= argument).

Usage:
    python 03_train_stage1.py [--n-iter 20] [--n-jobs -1]
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
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from holdout import exclude_holdout
from model_utils import build_preprocessor, spatial_cluster_folds, youden_threshold, compute_specificity

# Same drop list as the R version's drop_cols -- IDs/keys/low-value geography
# text that shouldn't be fed to the model directly (geography's SIGNAL is
# already captured via is_municipal/any_geo_match from add_name_matching).
#
# The second group is SCAFFOLDING: columns some upstream step added for its own
# internal use that were never meant to be features. build_preprocessor() bins
# columns by dtype, not intent, so nothing downstream can tell them apart from
# real signal -- they have to be named here. Added 2026-08-25 after
# inspect_model_features.py showed all three in the deployed model's fitted
# feature set:
#   geom_wkb       point_in_parcel_lookup's input geometry, returned by its old
#                  `SELECT pts.*`. One-hot encoded into 1629 levels -- a per-row
#                  unique blob, i.e. a row ID, and 93% of Stage 1's entire
#                  categorical feature space. Also fixed at source in 02 now;
#                  kept here so a stale 14_stage1_training.parquet can't
#                  reintroduce it.
#   x_5070/y_5070  Albers coords computed below purely to build the KMeans
#                  spatial-CV folds. As live features they invite geographic
#                  memorisation instead of transferable land-use signal --
#                  precisely what spatial CV exists to detect, so leaving them
#                  in also corrupts the measurement meant to catch it. Dropped
#                  from X only; `coords` is read off s1 directly, so fold
#                  construction is unaffected.
DROP_COLS = [
    "ll_uuid", "h3_index_9", "h3_res9", "state", "geoid",
    "county_geoid", "LATITUDE", "LONGITUDE", "n_parcels",
    "pct_lbcs_activity_known", "pct_owner_known",
    "subdivision", "place", "county", "zoning_type",
    "geom_wkb", "x_5070", "y_5070",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=20, help="random search iterations (R used grid size=20)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--allow-no-holdout", action="store_true",
                    help="proceed even if the holdout manifest is missing. Only "
                         "for deliberate pre-holdout runs -- normally a missing "
                         "manifest should stop the job.")
    args = ap.parse_args()

    C.ensure_dirs()
    print("=== 03_train_stage1.py: Stage 1 Model Training (Random Forest) ===\n")

    print("Loading Stage 1 training data...")
    s1 = pd.read_parquet(C.FEATURES_OUTPUT_DIR / "14_stage1_training.parquet")
    s1 = exclude_holdout(s1, "stage1", allow_missing=args.allow_no_holdout)
    print(f"  Rows: {len(s1)}")
    print(f"  Correct: {(s1['class'] == 'Correct').sum()}")
    print(f"  Incorrect: {(s1['class'] == 'Incorrect').sum()}")
    if s1["class"].isna().any() or len(s1) == 0:
        raise ValueError("No labeled rows found -- check 02_feature_engineering.py output")

    # ---- Coordinates for spatial CV (before dropping CWNS_ID) ----
    plant_coords = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt",
                                dtype={"CWNS_ID": str}, encoding="latin1")[
        ["CWNS_ID", "LATITUDE", "LONGITUDE"]].dropna()
    plant_coords["LATITUDE"] = pd.to_numeric(plant_coords["LATITUDE"], errors="coerce")
    plant_coords["LONGITUDE"] = pd.to_numeric(plant_coords["LONGITUDE"], errors="coerce")
    plant_coords = plant_coords.dropna()

    s1 = s1.merge(plant_coords, on="CWNS_ID", how="left")
    n_before = len(s1)
    s1 = s1.dropna(subset=["LATITUDE", "LONGITUDE"])
    if len(s1) < n_before:
        print(f"  Dropped {n_before - len(s1)} rows with no plant coordinates for spatial CV")

    import geopandas as gpd
    coords_5070 = gpd.GeoSeries(
        gpd.points_from_xy(s1["LONGITUDE"], s1["LATITUDE"]), crs=4326
    ).to_crs(5070)
    s1["x_5070"] = coords_5070.x.to_numpy()
    s1["y_5070"] = coords_5070.y.to_numpy()

    # ---- Feature selection ----
    y_raw = s1["class"]
    X = s1.drop(columns=[c for c in DROP_COLS if c in s1.columns] +
                ["class", "CWNS_ID", "LATITUDE", "LONGITUDE"], errors="ignore")
    if "place_match" in X.columns:
        X["place_match"] = X["place_match"].fillna(False)
    coords = s1[["x_5070", "y_5070"]].reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = (y_raw == "Correct").astype(int).reset_index(drop=True)   # 1=Correct, 0=Incorrect

    print(f"  Model columns: {X.shape[1]}")

    # ---- Class weights (identical formula to the R version) ----
    n_correct = int((y == 1).sum())
    n_incorrect = int((y == 0).sum())
    n_total = len(y)
    class_weight = {1: n_incorrect / n_total, 0: n_correct / n_total}
    print(f"  Class weights -- Correct: {class_weight[1]:.3f} | Incorrect: {class_weight[0]:.3f}")

    # ---- CV fold construction ----
    print("\nSetting up cross-validation folds...")
    spatial_folds = spatial_cluster_folds(coords, n_splits=5, random_state=123)
    for i, (tr, te) in enumerate(spatial_folds, 1):
        print(f"  Spatial fold {i} train -- Correct: {(y.iloc[tr] == 1).sum()}, "
              f"Incorrect: {(y.iloc[tr] == 0).sum()}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    standard_folds = list(skf.split(X, y))
    print(f"  Spatial folds: {len(spatial_folds)}  |  Standard folds: {len(standard_folds)}")

    # ---- Pipeline + search space ----
    n_features = X.shape[1]
    param_dist = {
        "model__max_features": np.linspace(0.1, 1.0, 20),   # ~ mtry(range=c(1,n_features))
        "model__min_samples_leaf": np.arange(1, 21),          # ~ min_n()
    }

    def make_pipeline():
        return Pipeline([
            ("prep", build_preprocessor(X)),
            ("model", RandomForestClassifier(
                n_estimators=1000, class_weight=class_weight,
                random_state=123, n_jobs=args.n_jobs)),
        ])

    def run_cv(folds, label):
        print(f"\n{'=' * 40}\nTuning with {label} cross-validation...\n{'=' * 40}")
        t0 = time.time()
        search = RandomizedSearchCV(
            make_pipeline(), param_distributions=param_dist, n_iter=args.n_iter,
            scoring="roc_auc", cv=folds, random_state=123, n_jobs=args.n_jobs, refit=False,
        )
        search.fit(X, y)
        elapsed = (time.time() - t0) / 60
        print(f"  {label} CV complete in {elapsed:.1f} minutes")
        print(f"  Best {label} ROC AUC: {search.best_score_:.4f}")
        return search

    spatial_search = run_cv(spatial_folds, "SPATIAL")
    standard_search = run_cv(standard_folds, "STANDARD")

    compare_cv = pd.DataFrame([
        dict(cv="Spatial", roc_auc=spatial_search.best_score_),
        dict(cv="Standard", roc_auc=standard_search.best_score_),
    ]).sort_values("roc_auc", ascending=False)
    print(f"\nCV Comparison:\n{compare_cv}")
    compare_cv.to_parquet(C.MODELS_DIR / "stage1_cv_comparison.parquet", index=False)

    # ---- Final fit on held-out test split, using SPATIAL CV's best params
    #      (matches the R version's choice: select_best(rf_spatial_results)) ----
    print(f"\n{'=' * 40}\nFitting final model on full training data...\n{'=' * 40}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=456)

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

    print("\nConfusion matrix (rows=truth, cols=predicted; 1=Correct, 0=Incorrect):")
    print(confusion_matrix(y_test, y_pred))

    # ---- Feature importance (permutation, on held-out test set) ----
    print("\nComputing feature importance...")
    perm = permutation_importance(final_pipeline, X_test, y_test, n_repeats=10,
                                   random_state=123, n_jobs=args.n_jobs, scoring="roc_auc")
    rf_importance = pd.DataFrame({
        "Variable": X_test.columns, "Importance": perm.importances_mean,
        "Importance_std": perm.importances_std,
    }).sort_values("Importance", ascending=False)
    print("Top 20 features:")
    print(rf_importance.head(20).to_string(index=False))
    rf_importance.to_parquet(C.MODELS_DIR / "stage1_rf_importance.parquet", index=False)

    # ---- Threshold analysis ----
    print("\nRunning threshold analysis...")
    optimal_threshold, roc_df = youden_threshold(y_test.to_numpy(), y_pred_proba)
    print(f"  Optimal Stage 1 threshold (Youden J): {optimal_threshold:.3f}")
    roc_df.to_parquet(C.MODELS_DIR / "stage1_threshold_analysis.parquet", index=False)

    # ---- Save model ----
    print("\nSaving model...")
    model_path = C.MODELS_DIR / "stage1_rf_model.joblib"
    joblib.dump(dict(pipeline=final_pipeline, feature_columns=list(X.columns),
                     class_labels={1: "Correct", 0: "Incorrect"}), model_path)
    joblib.dump(optimal_threshold, C.MODELS_DIR / "stage1_optimal_threshold.joblib")
    print(f"  {model_path.name} saved: {model_path.stat().st_size / 1e6:.1f} MB")

    print("\n=== Stage 1 training complete ===")
    print(f"\nFINAL RESULTS SUMMARY")
    print(f"  Test roc_auc: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"  Optimal threshold: {optimal_threshold:.3f}")
    print(f"  CV comparison:\n{compare_cv}")


if __name__ == "__main__":
    main()