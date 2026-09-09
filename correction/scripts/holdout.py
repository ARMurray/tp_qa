"""
holdout.py
============
Shared helpers for the frozen evaluation holdout. Imported by 03, 04, 06, 07
(to EXCLUDE holdout plants from training) and by 12 (to score against it).

Kept as its own module rather than added to model_utils.py so that the
exclusion logic lives in exactly one place and every training script provably
calls the same function. Four bespoke anti-joins would drift.

THE RULE THIS ENFORCES
----------------------
The holdout is sampled ONCE, by 09_build_holdout.py, and frozen. Plants in it
never enter training, in any round, in any form. If the sample changed between
rounds, plants would rotate between train and eval and every round-over-round
comparison would be meaningless -- and the corruption would be silent, showing
up only as unexplainably good numbers.

exclude_holdout() is therefore FAIL-CLOSED: if the manifest is missing it
raises rather than returning the frame untouched. A training run that silently
trains on the holdout is worse than a training run that stops. Pass
allow_missing=True only for a deliberate pre-holdout run.
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

HOLDOUT_DIR = C.DATA_DIR / "holdout"
MANIFEST_PATH = HOLDOUT_DIR / "holdout_manifest.parquet"
TRUTH_PATH = HOLDOUT_DIR / "holdout_truth.parquet"
SCORES_PATH = HOLDOUT_DIR / "holdout_scores.parquet"


def manifest_checksum(path: Path = MANIFEST_PATH) -> str:
    """SHA-256 of the manifest file. Written into round_manifest.json by the
    HPC side and verified by ingest_review_log.R locally, so a stale or edited
    manifest on the review machine is caught rather than silently used."""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_holdout_ids(allow_missing: bool = False) -> set[str]:
    """CWNS_IDs reserved for evaluation. Returns a set of str."""
    if not MANIFEST_PATH.exists():
        if allow_missing:
            print(f"  [holdout] no manifest at {MANIFEST_PATH} -- proceeding "
                  f"WITHOUT exclusion (allow_missing=True)")
            return set()
        raise FileNotFoundError(
            f"Holdout manifest not found: {MANIFEST_PATH}\n"
            f"Run 09_build_holdout.py first. Training without it would put "
            f"evaluation plants into the training set and quietly invalidate "
            f"every holdout score from here on.\n"
            f"If this is a deliberate pre-holdout run, pass --allow-no-holdout."
        )
    ids = set(pd.read_parquet(MANIFEST_PATH, columns=["CWNS_ID"])["CWNS_ID"].astype(str))
    return ids


def exclude_holdout(df: pd.DataFrame, stage: str,
                    id_col: str = "CWNS_ID",
                    allow_missing: bool = False) -> pd.DataFrame:
    """Drop holdout plants from a training frame. Call immediately after
    loading, before any feature selection or splitting.

    Prints what it removed -- a silent anti-join is hard to notice when it
    misfires, and the row count is the only visible evidence it ran."""
    if id_col not in df.columns:
        raise KeyError(
            f"[holdout] {stage}: '{id_col}' not in the frame, cannot exclude "
            f"holdout plants. Columns: {list(df.columns)[:20]}")

    ids = load_holdout_ids(allow_missing=allow_missing)
    if not ids:
        return df

    mask = df[id_col].astype(str).isin(ids)
    n_rows, n_plants = int(mask.sum()), int(df.loc[mask, id_col].nunique())
    out = df.loc[~mask].reset_index(drop=True)
    print(f"  [holdout] {stage}: removed {n_rows} rows / {n_plants} plants "
          f"({len(df)} -> {len(out)})")
    if len(out) == 0:
        raise ValueError(
            f"[holdout] {stage}: exclusion removed every row. The manifest is "
            f"probably not the right one for this table -- check it before "
            f"proceeding.")
    return out


def load_holdout_truth() -> pd.DataFrame:
    """Reviewed ground truth for holdout plants. Empty frame with the right
    schema if round 1 review hasn't happened yet."""
    if not TRUTH_PATH.exists():
        return pd.DataFrame(columns=[
            "CWNS_ID", "true_ll_uuid", "true_lon", "true_lat",
            "truth_source", "reviewed_at", "reviewer"])
    df = pd.read_parquet(TRUTH_PATH)
    df["CWNS_ID"] = df["CWNS_ID"].astype(str)
    return df
