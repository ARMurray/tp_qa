"""
04_train_stage2.py
===================
Port of train_stage2.R. Trains the Stage 2 classifier: GIVEN that Stage 1
flagged a reported location as incorrect, which candidate parcel (from the
k=18 ring) is the actual correct one?

Model: imblearn.ensemble.BalancedRandomForestClassifier -- the direct
equivalent of ranger's sample.fraction=c(0.5,0.5) (per-tree balanced
bootstrap), which MODEL_NOTES.md documents as the fix for a real bug where
plain class-weighting alone caused ranger to collapse to majority-class-only
predictions. class_weight alone (what Stage 1 uses) is NOT the same fix;
Stage 2 needs the balanced-bootstrap mechanism specifically.
Install: pip install imbalanced-learn

Usage:
    python 04_train_stage2.py [--neg-ratio 50] [--n-iter 20] [--n-jobs -1]
"""
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
except ImportError:
    raise ImportError(
        "imbalanced-learn is required for Stage 2 (pip install imbalanced-learn). "
        "This is NOT optional -- see module docstring for why plain class_weight "
        "isn't an equivalent substitute here."
    )

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from holdout import exclude_holdout
from model_utils import build_preprocessor, spatial_cluster_folds, youden_threshold, compute_specificity

# Scaffolding group -- see 03_train_stage1.py's DROP_COLS comment for the
# general failure mode. Added 2026-08-25 from inspect_model_features.py:
#   x_5070/y_5070            Albers coords for the KMeans spatial-CV folds.
#                            Dropped from X only; `coords` is read off
#                            s2_balanced directly, so folds are unaffected.
#   centroid_lat/centroid_lng  the candidate parcel's H3 cell centre, added in
#                            02's build_stage2_training solely to compute
#                            distance_m. The distance features it feeds
#                            (distance_m, log_distance, within_1km, within_5km)
#                            are the transferable signal; the raw centroid is
#                            just an absolute position, and Stage 2a's whole job
#                            is ranking candidates for out-of-training-state
#                            plants where absolute position generalises to
#                            nothing.
#   geom_wkb                 not currently in 15_stage2_training.parquet, but
#                            build_stage2_training calls the same
#                            point_in_parcel_lookup Stage 1 leaked it through.
#                            Listed so a future change there can't silently
#                            repeat the Stage 1 contamination here.
DROP_COLS = [
    "ll_uuid", "h3_index_9", "h3_res9", "state", "geoid",
    "county_geoid", "LATITUDE", "LONGITUDE", "n_parcels",
    "pct_lbcs_activity_known", "pct_owner_known",
    "is_reported", "distance_ring", "subdivision", "place", "county", "zoning_type",
    "geom_wkb", "x_5070", "y_5070", "centroid_lat", "centroid_lng",
]


def sample_negatives(s2: pd.DataFrame, neg_ratio: int, random_state: int = 123) -> pd.DataFrame:
    """Hard-negative mining: negatives that LOOK plausible (utility zoning,
    government ownership, OSM tag, water present, etc.) are oversampled
    relative to easy/obvious negatives, so the model actually has to learn
    to discriminate rather than just filtering out obviously-wrong parcels."""
    positives = s2[s2["label"] == 1]

    hard_mask = (
        s2["lbcs_activity"].isin(["Water Utility", "Utility Other"]) |
        (s2["lbcs_ownership"] == "Government") |
        (s2["osm_ww"] == True) |
        (s2["has_water"] == True) |
        (s2["owner_is_utility"] == True) |
        (s2["owner_is_govt"] == True) |
        (s2["has_ww_keyword"] == True)
    )
    hard_negatives = s2[(s2["label"] == 0) & hard_mask]
    easy_negatives = s2[(s2["label"] == 0) & ~s2["ll_uuid"].isin(hard_negatives["ll_uuid"])]

    n_positives = len(positives)
    n_target_neg = n_positives * neg_ratio
    n_hard = min(len(hard_negatives), round(n_target_neg * 0.7))
    n_easy = min(len(easy_negatives), n_target_neg - n_hard)

    print(f"  Positives: {n_positives}")
    print(f"  Hard negatives sampled: {n_hard}")
    print(f"  Easy negatives sampled: {n_easy}")
    print(f"  Total training rows: {n_positives + n_hard + n_easy}")

    rng = np.random.RandomState(random_state)
    hard_sample = hard_negatives.sample(n=n_hard, random_state=rng) if n_hard > 0 else hard_negatives.iloc[:0]
    easy_sample = easy_negatives.sample(n=n_easy, random_state=rng) if n_easy > 0 else easy_negatives.iloc[:0]

    return pd.concat([positives, hard_sample, easy_sample], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-ratio", type=int, default=50)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--allow-no-holdout", action="store_true",
                    help="proceed even if the holdout manifest is missing. Only "
                         "for deliberate pre-holdout runs -- normally a missing "
                         "manifest should stop the job.")
    args = ap.parse_args()

    C.ensure_dirs()
    print("=== 04_train_stage2.py: Stage 2 Model Training (Balanced Random Forest) ===")
    print(f"NEG_RATIO: {args.neg_ratio}\n")

    print("Loading Stage 2 training data...")
    s2 = pd.read_parquet(C.FEATURES_OUTPUT_DIR / "15_stage2_training.parquet")
    s2 = exclude_holdout(s2, "stage2a", allow_missing=args.allow_no_holdout)
    print(f"  Total rows: {len(s2)}")
    print(f"  Positive labels: {(s2['label'] == 1).sum()}")
    print(f"  Negative labels: {(s2['label'] == 0).sum()}")
    print(f"  Plants: {s2['CWNS_ID'].nunique()}")
    if len(s2) == 0:
        raise ValueError("No Stage 2 training rows -- check 02_feature_engineering.py output")

    print("\nSampling negatives with hard negative mining...")
    s2_balanced = sample_negatives(s2, args.neg_ratio)
    print("\n  Balanced class distribution:")
    print(s2_balanced["label"].value_counts())

    # ---- Coordinates for spatial CV ----
    plant_coords = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt",
                                dtype={"CWNS_ID": str}, encoding="latin1")[
        ["CWNS_ID", "LATITUDE", "LONGITUDE"]].dropna()
    plant_coords["LATITUDE"] = pd.to_numeric(plant_coords["LATITUDE"], errors="coerce")
    plant_coords["LONGITUDE"] = pd.to_numeric(plant_coords["LONGITUDE"], errors="coerce")
    plant_coords = plant_coords.dropna()

    s2_balanced = s2_balanced.merge(plant_coords, on="CWNS_ID", how="left")
    n_before = len(s2_balanced)
    s2_balanced = s2_balanced.dropna(subset=["LATITUDE", "LONGITUDE"])
    if len(s2_balanced) < n_before:
        print(f"  Dropped {n_before - len(s2_balanced)} rows with no plant coordinates for spatial CV")

    import geopandas as gpd
    coords_5070 = gpd.GeoSeries(
        gpd.points_from_xy(s2_balanced["LONGITUDE"], s2_balanced["LATITUDE"]), crs=4326
    ).to_crs(5070)
    s2_balanced["x_5070"] = coords_5070.x.to_numpy()
    s2_balanced["y_5070"] = coords_5070.y.to_numpy()

    # ---- Feature selection ----
    y = s2_balanced["label"].astype(int).reset_index(drop=True)   # already 1/0
    X = s2_balanced.drop(columns=[c for c in DROP_COLS if c in s2_balanced.columns] +
                          ["label", "CWNS_ID", "LATITUDE", "LONGITUDE"], errors="ignore")
    if "place_match" in X.columns:
        X["place_match"] = X["place_match"].fillna(False)
    coords = s2_balanced[["x_5070", "y_5070"]].reset_index(drop=True)
    X = X.reset_index(drop=True)

    print(f"  Model columns: {X.shape[1]}")

    # ---- CV folds ----
    print("\nSetting up cross-validation folds...")
    spatial_folds = spatial_cluster_folds(coords, n_splits=5, random_state=123)
    for i, (tr, te) in enumerate(spatial_folds, 1):
        print(f"  Spatial fold {i} train -- Correct: {(y.iloc[tr] == 1).sum()}, "
              f"Incorrect: {(y.iloc[tr] == 0).sum()}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    standard_folds = list(skf.split(X, y))
    print(f"  Spatial folds: {len(spatial_folds)}  |  Standard folds: {len(standard_folds)}")

    # ---- Pipeline + search space ----
    param_dist = {
        "model__max_features": np.linspace(0.1, 1.0, 20),
        "model__min_samples_leaf": np.arange(1, 21),
    }

    def make_pipeline():
        return Pipeline([
            ("prep", build_preprocessor(X)),
            ("model", BalancedRandomForestClassifier(
                n_estimators=1000, random_state=123, n_jobs=args.n_jobs,
                sampling_strategy="all", replacement=True)),
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
    compare_cv.to_parquet(C.MODELS_DIR / "stage2_cv_comparison.parquet", index=False)

    # ---- Final fit ----
    print(f"\n{'=' * 40}\nFitting final model on full training data...\n{'=' * 40}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=456)

    best_params = spatial_search.best_params_
    final_pipeline = make_pipeline()
    final_pipeline.set_params(**best_params)
    final_pipeline.fit(X_train, y_train)

    y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]
    y_pred = final_pipeline.predict(X_test)

    print("\nFINAL RESULTS SUMMARY")
    print("Test set metrics:")
    print(f"  roc_auc     : {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"  accuracy    : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  sensitivity : {(y_pred[y_test == 1] == 1).mean():.4f}")
    print(f"  specificity : {compute_specificity(y_test.to_numpy(), y_pred, pos_label=1):.4f}")
    print(f"  brier_class : {brier_score_loss(y_test, y_pred_proba):.4f}")

    print("\nConfusion matrix (rows=truth, cols=predicted; 1=Correct, 0=Incorrect):")
    print(confusion_matrix(y_test, y_pred))

    # ---- Feature importance ----
    print("\nComputing feature importance...")
    perm = permutation_importance(final_pipeline, X_test, y_test, n_repeats=10,
                                   random_state=123, n_jobs=args.n_jobs, scoring="roc_auc")
    rf_importance = pd.DataFrame({
        "Variable": X_test.columns, "Importance": perm.importances_mean,
        "Importance_std": perm.importances_std,
    }).sort_values("Importance", ascending=False)
    print("Top 20 features:")
    print(rf_importance.head(20).to_string(index=False))
    rf_importance.to_parquet(C.MODELS_DIR / "stage2_rf_importance.parquet", index=False)

    # ---- Threshold analysis ----
    print("\nRunning threshold analysis...")
    optimal_threshold, roc_df = youden_threshold(y_test.to_numpy(), y_pred_proba)
    print(f"  Optimal Stage 2 threshold (Youden J): {optimal_threshold:.3f}")
    roc_df.to_parquet(C.MODELS_DIR / "stage2_threshold_analysis.parquet", index=False)

    print(f"\nCV Comparison:\n{compare_cv}")
    print(f"Optimal threshold: {optimal_threshold:.3f}")

    # ---- Save model ----
    print("\nSaving model...")
    model_path = C.MODELS_DIR / "stage2_rf_model.joblib"
    joblib.dump(dict(pipeline=final_pipeline, feature_columns=list(X.columns),
                     class_labels={1: "Correct", 0: "Incorrect"},
                     neg_ratio=args.neg_ratio), model_path)
    joblib.dump(optimal_threshold, C.MODELS_DIR / "stage2_optimal_threshold.joblib")
    print(f"  {model_path.name} saved: {model_path.stat().st_size / 1e6:.1f} MB")

    print("\n=== Stage 2 training complete ===")


if __name__ == "__main__":
    main()