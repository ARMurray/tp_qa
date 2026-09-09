"""
model_utils.py
===============
Shared logic for 03_train_stage1.py and 04_train_stage2.py -- preprocessing
pipeline, spatial CV fold construction, and Youden-J threshold selection.
Factored out because both training scripts need IDENTICAL versions of this
logic; duplicating it would just be two places for the same bug to hide.

Porting notes vs. the R (tidymodels/ranger) originals:
  - step_unknown/step_novel -> OneHotEncoder(handle_unknown="ignore") after
    filling NaN with the literal string "unknown". handle_unknown="ignore"
    is what gives novel-category robustness at inference time (a category
    never seen in training just gets an all-zero one-hot row instead of
    erroring), matching step_novel's purpose.
  - step_impute_median -> SimpleImputer(strategy="median").
  - step_dummy -> OneHotEncoder(drop=None) -- NOT drop="first". ranger (like
    most tree libraries) doesn't need reference-level dummy encoding the way
    a linear model does; keeping all levels lets the RF split on any of them
    directly rather than only relative to a dropped reference.
  - step_zv -> handled implicitly: a truly zero-variance column can't produce
    a useful split anywhere in a tree, so it's harmless to leave in rather
    than requiring an explicit VarianceThreshold step.
  - step_normalize -> DELIBERATELY OMITTED. Tree splits are invariant to any
    monotonic transform of a numeric feature, so normalizing does nothing
    for a random forest (unlike a linear/distance-based model). Omitting it
    changes no result; it's not a shortcut that costs accuracy.
  - spatial_clustering_cv -> approximated with KMeans on Albers (EPSG:5070)
    projected coordinates, then leave-one-cluster-out folds. Same spirit
    (holdout fold is geographically contiguous, not a random sample) but a
    different underlying clustering algorithm than the R package uses.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Categorical -> fillna('unknown') + one-hot (handle_unknown='ignore').
    Numeric -> median impute. Returns an UNFITTED ColumnTransformer; fit it
    only on the training fold, never on the full dataset, or CV metrics will
    be optimistic (median/categories leaking test-fold information)."""
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    cat_pipe = Pipeline([
        ("fillna", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])
    return ColumnTransformer([
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols),
    ])


def spatial_cluster_folds(coords: pd.DataFrame, n_splits: int = 5, random_state: int = 123):
    """coords: DataFrame with 'x_5070','y_5070' columns (Albers-projected).
    Returns list of (train_idx, test_idx) tuples, one per cluster, mirroring
    spatial_clustering_cv's leave-one-geographic-cluster-out design -- each
    fold's test set is spatially contiguous, not a random sample, so a model
    that's just memorizing "which county" can't cheat its way to a good score."""
    km = KMeans(n_clusters=n_splits, random_state=random_state, n_init=10)
    cluster_id = km.fit_predict(coords[["x_5070", "y_5070"]].to_numpy())
    idx = np.arange(len(coords))
    folds = []
    for c in range(n_splits):
        test_idx = idx[cluster_id == c]
        train_idx = idx[cluster_id != c]
        folds.append((train_idx, test_idx))
    return folds


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Youden's J = sensitivity + specificity - 1 = TPR - FPR, maximized.
    Returns (optimal_threshold, full roc_curve DataFrame) -- the DataFrame
    is saved alongside the model, matching the R version's
    stageN_threshold_analysis.parquet output."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    roc_df = pd.DataFrame({
        "threshold": thresholds, "sensitivity": tpr,
        "specificity": 1 - fpr, "j_index": j,
    })
    return float(thresholds[best_idx]), roc_df


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray, pos_label) -> float:
    """sklearn has no built-in 'specificity' scorer -- it's recall of the
    NEGATIVE class, computed directly from the confusion matrix."""
    from sklearn.metrics import confusion_matrix
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    neg_idx = [i for i, l in enumerate(labels) if l != pos_label]
    if len(neg_idx) != 1:
        return np.nan
    i = neg_idx[0]
    tn = cm[i, i]
    fp = cm[i, :].sum() - tn
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan