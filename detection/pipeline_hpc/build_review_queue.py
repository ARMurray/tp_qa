"""
build_review_queue.py
=====================
Collects samples whose detection result needs human eyes and writes a single
mappable parquet.

Motivated by the national run: the detector fires on only 53.7% of KNOWN-GOOD
locations, 45.8% of matched pairs are zero-on-both, but among pairs it decides
it is right 84.1% of the time (132 vs 25). High precision, low recall. The
open question is WHAT the silent 46% are -- the strong prior is lagoon plants,
since the deployed model is aeration_basin/clarifier/digester with no
oxidation_pond, and lagoon systems skew small and rural. If that's confirmed,
oxidation_pond is the highest-value annotation class to add next.

Output columns are chosen so the parquet drops straight into sf/QGIS: rep_lon,
rep_lat for the reported point and det_lon, det_lat for the detection centroid,
plus a `review_reason` and a `priority` for sorting.

REVIEW CATEGORIES
  zero_on_correct     Correct label, imagery fine, nothing found. The recall
                      failures. Biggest group, highest diagnostic value.
  both_zero_pair      Matched pair with nothing on either side -- neither
                      confirms nor refutes; pure detector silence.
  detected_on_wrong   Incorrect label but strong detection. Either a genuine
                      plant at the original point (label may be wrong) or a
                      neighbouring facility. Small group, worth reading.
  reversed_pair       Matched pair where the INCORRECT parcel has more objects
                      than the Correct one. 25 nationally. If these are real,
                      they're either mislabels or detector errors -- the single
                      most informative set to inspect.
  far_detection       Detection present but far from the reported point.
                      Possible relocation candidates.
  oversized_parcel    Regrid coverage gap; parcel_* suppressed.
  parcel_clipped      Hard extent clip bit into the parcel; parcel_* undercounts.
  bad_imagery         Fetch failures or nodata tiles.

Usage:
    python build_review_queue.py
    python build_review_queue.py --state 39
    python build_review_queue.py --out /path/to/review_queue.parquet
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/work/GRDVULN/infrastructure")
SAMPLES = REPO / "data" / "inference_train" / "samples"
OBJECTS = REPO / "data" / "inference_train" / "objects"
DEFAULT_OUT = REPO / "data" / "samples" / "review_queue.parquet"

FAR_DETECTION_M = 250.0

PRIORITY = {
    "reversed_pair": 1,
    "detected_on_wrong": 2,
    "zero_on_correct": 3,
    "both_zero_pair": 4,
    "far_detection": 5,
    "parcel_clipped": 6,
    "oversized_parcel": 7,
    "bad_imagery": 8,
}


def hdr(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def load(root: Path, state=None):
    pat = str(root / (f"state={state}" if state else "state=*") / "*.parquet")
    files = sorted(glob.glob(pat))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "sample_id" in df and "processed_at" in df and root.name == "samples":
        df = df.sort_values("processed_at").drop_duplicates("sample_id", keep="last")
    return df


def flag(df):
    """Assign review reasons. A sample can hit several; we keep them all as a
    comma-joined string and sort on the most urgent."""
    reasons = {r: pd.Series(False, index=df.index) for r in PRIORITY}

    ok = df["imagery_ok"].fillna(False).astype(bool)
    n = df["win_n_objects"].fillna(0)

    reasons["zero_on_correct"] = (df["label_class"] == "Correct") & ok & (n == 0)
    reasons["detected_on_wrong"] = (df["label_class"] == "Incorrect") & (n >= 2)
    reasons["far_detection"] = (
        df["win_dist_nearest_object_m"].notna()
        & (df["win_dist_nearest_object_m"] > FAR_DETECTION_M))
    reasons["bad_imagery"] = (~ok) | (df["n_tiles_fetch_failed"].fillna(0) > 0)
    if "parcel_oversized" in df:
        reasons["oversized_parcel"] = df["parcel_oversized"].fillna(False).astype(bool)
    if "parcel_clipped" in df:
        reasons["parcel_clipped"] = df["parcel_clipped"].fillna(False).astype(bool)

    # Pair-level reasons need both halves of a CWNS_ID.
    piv = df.pivot_table(index="CWNS_ID", columns="label_class",
                         values="win_n_objects", aggfunc="first")
    if {"Correct", "Incorrect"}.issubset(piv.columns):
        piv = piv.dropna(subset=["Correct", "Incorrect"])
        both_zero = set(piv[(piv["Correct"] == 0) & (piv["Incorrect"] == 0)].index)
        reversed_ = set(piv[piv["Incorrect"] > piv["Correct"]].index)
        reasons["both_zero_pair"] = df["CWNS_ID"].isin(both_zero)
        reasons["reversed_pair"] = df["CWNS_ID"].isin(reversed_)
        df["pair_delta"] = df["CWNS_ID"].map(piv["Correct"] - piv["Incorrect"])
    else:
        df["pair_delta"] = np.nan

    mat = pd.DataFrame(reasons)
    df["review_reason"] = mat.apply(
        lambda r: ",".join(sorted([c for c in mat.columns if r[c]],
                                  key=lambda k: PRIORITY[k])), axis=1)
    df["priority"] = mat.apply(
        lambda r: min([PRIORITY[c] for c in mat.columns if r[c]], default=99), axis=1)
    return df, mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--far-m", type=float, default=FAR_DETECTION_M)
    args = ap.parse_args()

    global FAR_DETECTION_M
    FAR_DETECTION_M = args.far_m

    df = load(SAMPLES, args.state)
    if df is None:
        raise SystemExit(f"No samples found under {SAMPLES}")
    print(f"Loaded {len(df)} samples across {df['state_fips'].nunique()} state(s)")

    df, mat = flag(df)

    hdr("REVIEW CATEGORIES")
    for r in sorted(PRIORITY, key=PRIORITY.get):
        k = int(mat[r].sum())
        print(f"  {r:20s} {k:5d}  ({k/len(df):5.1%})")

    flagged = df[df["priority"] < 99].copy()
    hdr("WHAT ARE THE SILENT ONES? (zero_on_correct vs the rest)")
    z = mat["zero_on_correct"]
    if z.any() and "parcel_area_m2" in df:
        comp = df.assign(zero_on_correct=z).groupby("zero_on_correct").agg(
            n=("sample_id", "size"),
            median_parcel_m2=("parcel_area_m2", "median"),
            median_tiles=("n_tiles_attempted", "median"))
        print(comp.to_string())
        print("\n  A much SMALLER median parcel among the silent ones is consistent")
        print("  with the lagoon hypothesis -- small rural plants the current")
        print("  3-class model has no oxidation_pond category for.")
        print("\n  worst states for silence (Correct samples only):")
        cor = df[df["label_class"] == "Correct"]
        by = (cor.assign(silent=cor["win_n_objects"] == 0)
                 .groupby("st")["silent"].agg(["size", "sum", "mean"])
                 .sort_values("mean", ascending=False))
        print(by[by["size"] >= 20].head(12).to_string())

    keep = [c for c in [
        "sample_id", "CWNS_ID", "label_class", "st", "geoid", "state_fips",
        "review_reason", "priority", "pair_delta",
        "rep_lon", "rep_lat", "win_det_lon", "win_det_lat",
        "win_n_objects", "win_n_distinct_classes", "win_max_confidence",
        "win_dist_nearest_object_m", "win_detection_state", "win_dominant_class",
        "parcel_n_objects", "parcel_detection_state",
        "ll_uuid", "parcel_area_m2", "parcel_oversized", "parcel_clipped",
        "n_tiles_attempted", "n_tiles_kept", "n_tiles_nodata",
        "n_tiles_fetch_failed", "imagery_ok", "model_weights", "model_classes",
    ] if c in flagged.columns]

    out = flagged[keep].sort_values(["priority", "st", "sample_id"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, engine="pyarrow", index=False)

    hdr("WROTE REVIEW QUEUE")
    print(f"  {out_path}")
    print(f"  {len(out)} rows / {len(df)} samples flagged")
    print("\n  Map it with:")
    print("    library(sf); library(arrow)")
    print(f"    q <- read_parquet('{out_path}')")
    print("    pts <- st_as_sf(q, coords = c('rep_lon','rep_lat'), crs = 4326)")
    print("    mapview::mapview(pts, zcol = 'review_reason')")
    print("\n  Start with priority 1 (reversed_pair) -- smallest group, most")
    print("  informative. Then priority 3 (zero_on_correct) to settle whether")
    print("  the silence is lagoons.")


if __name__ == "__main__":
    main()
