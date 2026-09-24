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


def spatial_cluster_folds(coords: pd.DataFrame, n_splits: int = 5,
                          random_state: int = 123, blocks_per_fold: int = 6,
                          n_blocks: int | None = None):
    """Spatially blocked CV folds of roughly equal size.

    coords: DataFrame with 'x_5070','y_5070' (Albers-projected). Returns a list
    of (train_idx, test_idx), one per fold. Each fold's test set is a union of
    spatially contiguous blocks, so a model memorizing "which county" still
    cannot cheat -- every point in a block is held out together with its
    neighbours.

    WHY THIS IS NOT KMeans(n_clusters=n_splits) ANY MORE (2026-09-23)
        It used to be exactly that: cluster into 5, hold out one cluster per
        fold. Plants are not uniformly distributed, so KMeans with k=5 on US
        plant locations reliably produces one enormous cluster and one tiny
        one. Measured on Stage 2, three separate runs a month apart, cluster
        sizes were ~59 / 14 / 12 / 10 / 4 % of rows every time -- the same
        shape, merely permuted in index. Concretely:

            fold 1 trained on 40% of the data and tested on 60%
            fold 3 trained on 96% and computed its ROC AUC from 10 positives

        RandomizedSearchCV averages the folds with EQUAL weight, so a fold
        scored on 10 positives -- where AUC is mostly noise -- got the same
        vote as one scored on 208. Hyperparameters were being chosen partly by
        cluster geometry.

        Now: cluster into many more, smaller blocks (n_splits * blocks_per_fold,
        30 by default), then pack whole blocks into n_splits folds, greedily
        assigning each block to whichever fold is currently smallest. Blocks
        stay contiguous and a block is never split across folds, so the
        anti-memorization property holds; fold SIZES come out even.

        The honest trade-off: with 30 small blocks instead of 5 big regions, a
        held-out block's nearest neighbours may sit in an adjacent block that
        is in the training set. That is weaker spatial separation than
        leave-one-big-region-out gave. It is the standard spatial-block-CV
        compromise, and it buys folds whose scores mean something. Raise
        blocks_per_fold for more even folds, lower it for stronger separation.

    CLUSTERING IS DONE ON UNIQUE LOCATIONS
        Stage 2 passes ~51 rows per plant (1 positive + NEG_RATIO sampled
        negatives), all carrying that plant's coordinates. Clustering the rows
        directly lets row counts drag the centroids toward whichever plants
        happen to have the most candidates. Clustering the distinct locations
        and mapping back also guarantees every row of a plant lands in the
        same fold, which is the grouping we want anyway.
    """
    pts = coords[["x_5070", "y_5070"]].to_numpy()
    uniq, inverse = np.unique(pts, axis=0, return_inverse=True)

    if len(uniq) < n_splits:
        raise ValueError(
            f"{len(uniq)} distinct location(s) cannot be split into {n_splits} "
            f"spatial folds. Pass fewer splits, or use StratifiedKFold.")

    k = n_blocks if n_blocks is not None else n_splits * blocks_per_fold
    k = int(min(max(k, n_splits), len(uniq)))

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    block_of_uniq = km.fit_predict(uniq)
    block_id = block_of_uniq[inverse]

    # Largest block first, each to the smallest fold so far. Greedy, but for
    # this shape it lands within a few percent of even -- and unlike the old
    # behaviour it cannot produce a fold holding 60% of the data.
    sizes = sorted(((int((block_id == b).sum()), b) for b in range(k)),
                   reverse=True)
    fold_of_block, fold_rows = {}, [0] * n_splits
    for n_rows, b in sizes:
        f = int(np.argmin(fold_rows))
        fold_of_block[b] = f
        fold_rows[f] += n_rows

    fold_id = np.array([fold_of_block[b] for b in block_id])
    empty = [f for f in range(n_splits) if fold_rows[f] == 0]
    if empty:
        raise ValueError(
            f"fold(s) {empty} came out empty from {k} block(s) over "
            f"{len(uniq)} distinct location(s) -- lower n_splits.")

    idx = np.arange(len(coords))
    return [(idx[fold_id != f], idx[fold_id == f]) for f in range(n_splits)]


def describe_folds(folds, y, label="fold"):
    """Print train/test size and POSITIVE COUNT per fold.

    The positive count per TEST fold is the number that matters and the one
    that used to be absent: a fold scored on 10 positives contributes noise at
    full weight to the mean CV score, and nothing in the old output made that
    visible -- it printed training counts only. Printing this is how the
    imbalance above was finally caught.
    """
    total = len(y)
    for i, (tr, te) in enumerate(folds, 1):
        n_pos_te = int((y.iloc[te] == 1).sum()) if hasattr(y, "iloc") else int((y[te] == 1).sum())
        n_pos_tr = int((y.iloc[tr] == 1).sum()) if hasattr(y, "iloc") else int((y[tr] == 1).sum())
        print(f"  {label} {i}: train {len(tr):>7,} ({len(tr) / total:>5.1%}, "
              f"{n_pos_tr:>5} pos)  |  test {len(te):>7,} "
              f"({len(te) / total:>5.1%}, {n_pos_te:>5} pos)")
        if n_pos_te < 15:
            print(f"      WARNING: {n_pos_te} positive(s) in this test fold -- "
                  f"its ROC AUC is mostly noise, and RandomizedSearchCV still "
                  f"weights it equally.")


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