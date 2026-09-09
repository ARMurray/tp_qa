"""
inspect_samples.py
==================
QA + first-look at the detection features produced by 09.

Three jobs:
  1. Schema integrity -- catches the v1/v2 mixed-schema situation caused by
     09's append-only writer (a schema change requires DELETING the state
     partition, not just --no-resume, or old and new rows stack).
  2. Tile cost -- confirms the window + parcel cap actually bounded the run.
  3. Signal check -- Correct vs Incorrect on the win_* block, plus the
     within-plant paired comparison, which is the powerful one: it holds
     state, plant size, NAIP vintage and parcel-data quality constant.

Usage:
    python inspect_samples.py                 # all states present
    python inspect_samples.py --state 39
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/work/GRDVULN/infrastructure/data/inference_train/samples")
MANIFEST = Path("/work/GRDVULN/infrastructure/data/samples/training_manifest.parquet")


def hdr(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def load(state=None):
    pat = str(ROOT / (f"state={state}" if state else "state=*") / "*.parquet")
    files = sorted(glob.glob(pat))
    if not files:
        raise SystemExit(f"No samples parquet found under {pat}")

    frames, schemas = [], {}
    for f in files:
        d = pd.read_parquet(f)
        schemas[f] = frozenset(d.columns)
        frames.append(d)

    distinct = set(schemas.values())
    if len(distinct) > 1:
        hdr("SCHEMA MISMATCH -- STOP")
        print(f"{len(distinct)} different column sets across {len(files)} part files.")
        ref = max(distinct, key=len)
        for f, s in schemas.items():
            if s != ref:
                print(f"\n  {Path(f).parent.name}/{Path(f).name}")
                print(f"    missing: {sorted(ref - s)[:8]}")
        print("\nThis is the append-only writer stacking old rows under new ones.")
        print("Delete the affected state partitions in ALL FOUR tables and re-run:")
        print("  rm -rf .../inference_train/{tiles,detections,objects,samples}/state=XX")
        raise SystemExit(1)

    df = pd.concat(frames, ignore_index=True)
    dup = df["sample_id"].duplicated().sum()
    if dup:
        print(f"WARNING: {dup} duplicate sample_id rows -- stale parts not cleared. "
              f"Keeping the most recent by processed_at.")
        df = df.sort_values("processed_at").drop_duplicates("sample_id", keep="last")
    return df


def coverage(df):
    hdr("COVERAGE vs MANIFEST")
    if not MANIFEST.exists():
        print("  (manifest not found, skipping)")
        return
    m = pd.read_parquet(MANIFEST)
    states = sorted(df["state_fips"].unique())
    m = m[m["state_fips"].isin(states)]
    print(f"  manifest rows : {len(m)}")
    print(f"  samples rows  : {df['sample_id'].nunique()}")
    missing = set(m["sample_id"]) - set(df["sample_id"])
    if missing:
        print(f"  MISSING       : {len(missing)}  e.g. {sorted(missing)[:5]}")
    else:
        print("  MISSING       : 0")


def cost(df):
    hdr("TILE COST")
    print(df["n_tiles_attempted"].describe(percentiles=[.5, .9, .99]).to_string())
    if "tiling_mode" in df:
        print("\nby tiling_mode:")
        print(df.groupby("tiling_mode")["n_tiles_attempted"]
                .agg(["count", "median", "max"]).to_string())
    if "parcel_oversized" in df:
        n = int(df["parcel_oversized"].sum())
        print(f"\noversized parcels (window-only): {n} / {len(df)}")
    worst = df.nlargest(5, "n_tiles_attempted")[
        ["sample_id", "tiling_mode", "parcel_area_m2", "n_tiles_attempted"]]
    print("\nmost expensive samples:")
    print(worst.to_string(index=False))
    print("\n  Anything in the hundreds means the union-bounds cap has a hole.")


def signal(df):
    hdr("SIGNAL: Correct vs Incorrect (pooled)")
    cols = [c for c in ["win_n_objects", "win_n_distinct_classes", "win_max_confidence",
                        "win_dist_nearest_object_m", "win_objects_per_ha"] if c in df]
    g = df.groupby("label_class")
    out = g[cols].mean()
    out["n"] = g.size()
    out["any_object"] = g["win_n_objects"].apply(lambda x: (x > 0).mean())
    if "win_has_clarifier" in df:
        out["has_clarifier"] = g["win_has_clarifier"].mean()
    print(out.to_string())
    print("\n  Pooled contrast is confounded by geography, plant size and NAIP")
    print("  vintage. Treat the paired result below as the real answer.")

    hdr("SIGNAL: within-plant paired comparison")
    paired = df[df.duplicated("CWNS_ID", keep=False)]
    piv = paired.pivot_table(index="CWNS_ID", columns="label_class",
                             values="win_n_objects", aggfunc="first").dropna()
    if len(piv) == 0:
        print("  No matched pairs in this subset (expected for a single small state).")
        print("  Re-run across all states -- there are 301 pairs nationally.")
        return
    piv["delta"] = piv["Correct"] - piv["Incorrect"]
    wins = int((piv["delta"] > 0).sum())
    ties = int((piv["delta"] == 0).sum())
    loss = int((piv["delta"] < 0).sum())
    print(f"  pairs: {len(piv)}")
    print(f"  correct parcel has MORE objects : {wins}  ({wins/len(piv):.1%})")
    print(f"  tie                              : {ties}  ({ties/len(piv):.1%})")
    print(f"  incorrect has more               : {loss}  ({loss/len(piv):.1%})")
    print(f"  mean delta: {piv['delta'].mean():+.2f}   median: {piv['delta'].median():+.1f}")
    try:
        from scipy.stats import wilcoxon
        nz = piv[piv["delta"] != 0]
        if len(nz) >= 6:
            s, p = wilcoxon(nz["Correct"], nz["Incorrect"])
            print(f"  Wilcoxon signed-rank (n={len(nz)} non-tied): p = {p:.4g}")
    except ImportError:
        pass
    print("\n  Ties matter: a tie at 0-0 means neither parcel showed anything,")
    print("  which is a model-recall problem, not a discrimination problem.")
    if "Correct" in piv:
        both_zero = int(((piv["Correct"] == 0) & (piv["Incorrect"] == 0)).sum())
        print(f"  pairs where BOTH are zero: {both_zero} ({both_zero/len(piv):.1%})")


def sentinels(df):
    hdr("SENTINEL CHECK (-1 = region not evaluated, 0 = looked, found nothing)")
    for c in [c for c in df.columns if c.startswith("parcel_n_obj")][:4]:
        print(f"  {c:34s} -1: {int((df[c] == -1).sum()):4d}   0: {int((df[c] == 0).sum()):4d}")
    if "model_classes" in df:
        print(f"\n  model_classes present: {df['model_classes'].unique()}")
        print("  Classes absent from that list have structurally-zero columns --")
        print("  Stage 1 must not read those as evidence of absence.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=None)
    args = ap.parse_args()
    df = load(args.state)
    print(f"Loaded {len(df)} samples across {df['state_fips'].nunique()} state(s)")
    coverage(df)
    cost(df)
    sentinels(df)
    signal(df)


if __name__ == "__main__":
    main()
