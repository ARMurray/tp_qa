"""
inspect_model_features.py
==========================
Audits what each DEPLOYED model was actually fitted on, against the training
table that produced it. Answers the standing question from
TPQA_MASTER_REFERENCE.md S10: "are there other scaffolding columns in the
trained feature sets?"

WHY THIS EXISTS
---------------
`geom_wkb` reached Stage 1's fitted pipeline as a one-hot encoded categorical
-- a per-row-unique binary blob, i.e. a row ID -- and nothing caught it until
inference crashed (2026-08-25). The failure mode is structural, not specific
to that column:

  1. A helper adds a column for its own internal use (a geometry blob for a
     spatial join, projected coords for KMeans fold construction, H3 cell
     centroids for a distance calculation).
  2. Nothing drops it before `X = df.drop(columns=DROP_COLS)`.
  3. `build_preprocessor()` bins columns by DTYPE, not by intent -- any
     object/bool column becomes a one-hot categorical, any numeric column
     becomes a median-imputed feature. Neither branch can tell scaffolding
     from signal.
  4. Training succeeds. The column is silently part of the model.

So the check that matters is not "does this column look wrong" but "how many
one-hot levels would this column produce relative to the row count". A
categorical whose cardinality tracks the row count is an identifier, whatever
it is named. This script measures that directly rather than matching names
against a blocklist, so it stays correct as new columns appear.

WHAT IT REPORTS, per model
--------------------------
  - trained feature count and how each column would be TREATED by
    build_preprocessor() (one-hot categorical vs numeric)
  - for categoricals: distinct levels, and levels-per-row -- the identifier test
  - columns flagged as scaffolding, by category, with the reason
  - which of those the training script's own DROP_COLS already covers, so the
    gap between "should be dropped" and "is dropped" is explicit
  - merge-suffix artifacts (_x/_y), which mean an upstream join collided and
    the model is fitted on one arbitrary side of it

Reads only committed artifacts -- no refit, no writes. Safe to run any time.

Usage:
    python inspect_model_features.py
    python inspect_model_features.py --stages stage1,stage2
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

SCRIPTS_DIR = Path(__file__).resolve().parent

# Each deployed model, the 02 output it was trained from, and the training
# script whose DROP_COLS governs it. Stage 2b's table is built by 06, not 02.
STAGES = {
    "stage1":  dict(table="14_stage1_training.parquet",  script="03_train_stage1.py"),
    "stage2":  dict(table="15_stage2_training.parquet",  script="04_train_stage2.py"),
    "stage2b": dict(table="16_stage2b_training.parquet", script="07_train_stage2b.py"),
}

# Columns that exist ONLY to make some upstream step work, grouped by what
# added them. Used to explain a flag, never as the flag's sole basis -- the
# cardinality test below is what catches the ones nobody has named yet.
SCAFFOLDING_REASONS = {
    "geom_wkb":     "WKB blob returned by point_in_parcel_lookup's `SELECT pts.*`",
    "x_5070":       "Albers coords added for spatial_cluster_folds() KMeans",
    "y_5070":       "Albers coords added for spatial_cluster_folds() KMeans",
    "centroid_lat": "H3 cell centroid, added to compute distance_m",
    "centroid_lng": "H3 cell centroid, added to compute distance_m",
    "owner":        "raw owner text; add_name_matching() consumes it into "
                    "is_municipal/owner_water/*_match and drops it",
    "ll_uuid":      "parcel identifier",
    "CWNS_ID":      "plant identifier",
    "h3_index_9":   "H3 cell identifier",
    "h3_res9":      "H3 cell identifier",
    "geoid":        "county identifier",
    "county_geoid": "county identifier",
}

# Cardinality at or above this fraction of rows means the column is behaving
# as an identifier rather than a category, regardless of its name or dtype.
# 0.5 is deliberately loose: a genuine categorical feature in this pipeline
# (lbcs_activity, dominant_class_group) has a fixed vocabulary in the tens,
# so real features sit orders of magnitude below the line and there is no
# need to tune this precisely.
ID_LIKE_LEVEL_RATIO = 0.5

# ...but the ratio alone is meaningless on a short table: against 3 rows a
# plain boolean scores 0.67 and every real feature "looks like" an ID. Require
# a large ABSOLUTE level count too, and skip the test entirely below
# ID_LIKE_MIN_ROWS -- which is the normal case for the tpqa_test fixture, where
# the training tables are single-digit rows by construction. On a short table
# this script can still report treatment, merge suffixes, and named
# scaffolding; the cardinality test needs a real table to say anything.
ID_LIKE_MIN_LEVELS = 50
ID_LIKE_MIN_ROWS = 200


def load_drop_cols(script_name: str) -> list[str]:
    """Pull DROP_COLS out of a training script. Filenames start with a digit,
    so a plain `import` isn't available -- same importlib pattern 05 uses to
    reuse 02's join logic."""
    path = SCRIPTS_DIR / script_name
    spec = importlib.util.spec_from_file_location(f"_ds_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(getattr(mod, "DROP_COLS", []))


def classify_for_preprocessor(series: pd.Series) -> str:
    """Mirror build_preprocessor()'s select_dtypes split exactly, so this
    reports how the column WAS treated, not how it ought to have been."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return "numeric"
    if (pd.api.types.is_object_dtype(series) or pd.api.types.is_bool_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(series)):
        return "onehot"
    return "other"


def audit_stage(stage: str, spec: dict) -> None:
    model_path = C.MODELS_DIR / f"{stage}_rf_model.joblib"
    table_path = C.FEATURES_OUTPUT_DIR / spec["table"]

    print(f"\n{'=' * 74}\n{stage.upper()}\n{'=' * 74}")
    if not model_path.exists():
        print(f"  no model at {model_path} -- skipping")
        return

    bundle = joblib.load(model_path)
    features = list(bundle["feature_columns"])
    print(f"  model : {model_path.name}  ({len(features)} trained features)")

    if not table_path.exists():
        print(f"  table : {table_path} NOT FOUND -- listing features only\n")
        for f in features:
            print(f"    {f}")
        return

    df = pd.read_parquet(table_path)
    print(f"  table : {spec['table']}  ({len(df)} rows, {len(df.columns)} cols)")
    print(f"  drops : {spec['script']} DROP_COLS")

    drop_cols = set(load_drop_cols(spec["script"]))
    n_rows = max(len(df), 1)

    rows = []
    for f in features:
        if f not in df.columns:
            rows.append(dict(feature=f, treated="MISSING", levels=np.nan,
                             ratio=np.nan, flag="not in training table"))
            continue
        s = df[f]
        treated = classify_for_preprocessor(s)
        levels = int(s.nunique(dropna=False)) if treated == "onehot" else np.nan
        ratio = levels / n_rows if treated == "onehot" else np.nan

        flag = ""
        if (treated == "onehot" and n_rows >= ID_LIKE_MIN_ROWS
                and levels >= ID_LIKE_MIN_LEVELS and ratio >= ID_LIKE_LEVEL_RATIO):
            flag = f"ID-LIKE ({levels} levels / {n_rows} rows)"
        elif f in SCAFFOLDING_REASONS:
            flag = "SCAFFOLDING"
        elif f.endswith(("_x", "_y")) and f not in ("x_5070", "y_5070"):
            flag = "MERGE-SUFFIX"
        rows.append(dict(feature=f, treated=treated, levels=levels,
                         ratio=ratio, flag=flag))

    audit = pd.DataFrame(rows)
    flagged = audit[audit["flag"] != ""]

    print(f"\n  Preprocessor treatment: "
          f"{(audit['treated'] == 'onehot').sum()} one-hot categorical, "
          f"{(audit['treated'] == 'numeric').sum()} numeric, "
          f"{(audit['treated'] == 'MISSING').sum()} absent from table")
    if n_rows < ID_LIKE_MIN_ROWS:
        print(f"  NOTE: {n_rows} rows is below {ID_LIKE_MIN_ROWS} -- the "
              f"cardinality test is disabled (it cannot separate an identifier "
              f"from a boolean on a table this short). Named-scaffolding and "
              f"merge-suffix checks still apply.")

    if flagged.empty:
        print("\n  No scaffolding or identifier-like columns in the trained "
              "feature set.")
    else:
        print(f"\n  {len(flagged)} FLAGGED FEATURE(S):\n")
        for _, r in flagged.iterrows():
            covered = "already in DROP_COLS" if r["feature"] in drop_cols \
                else "NOT DROPPED -- reaches the model"
            print(f"    {r['feature']:<26} {r['flag']}")
            reason = SCAFFOLDING_REASONS.get(r["feature"])
            if reason:
                print(f"    {'':<26} why present: {reason}")
            print(f"    {'':<26} {covered}")
            print()

    # A high-cardinality categorical is the specific thing that makes a model
    # unservable (novel level at inference -> all-zero one-hot row) as well as
    # untrustworthy, so surface the worst offenders even when unflagged.
    cats = audit[audit["treated"] == "onehot"].dropna(subset=["levels"])
    if len(cats):
        top = cats.sort_values("levels", ascending=False).head(5)
        print("  Highest-cardinality one-hot features (levels / rows):")
        for _, r in top.iterrows():
            print(f"    {r['feature']:<26} {int(r['levels']):>7} / {n_rows}"
                  f"   ({r['ratio']:.3f})")

    # DROP_COLS entries that never appear in the table are dead weight -- not a
    # bug, but they hide which protections are actually load-bearing.
    inert = sorted(c for c in drop_cols if c not in df.columns)
    if inert:
        print(f"\n  DROP_COLS entries absent from this table (inert): {inert}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=str, default="stage1,stage2,stage2b",
                    help="comma-separated subset of: stage1, stage2, stage2b")
    args = ap.parse_args()

    print("=== inspect_model_features.py ===")
    print(f"Models   : {C.MODELS_DIR}")
    print(f"Features : {C.FEATURES_OUTPUT_DIR}")

    for stage in [s.strip() for s in args.stages.split(",")]:
        if stage not in STAGES:
            print(f"\nUnknown stage {stage!r} -- expected one of {list(STAGES)}")
            continue
        audit_stage(stage, STAGES[stage])

    print(f"\n{'=' * 74}")
    print("Reminder: a flagged column means the DEPLOYED model was fitted with "
          "it.\nAdding it to DROP_COLS changes nothing until that model is "
          "retrained.")


if __name__ == "__main__":
    main()
