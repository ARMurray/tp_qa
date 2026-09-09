"""
diagnose_parcel_size.py
=======================
Diagnoses the oversized-parcel problem before patching 08/09.

Two questions:

  1. Are the town-sized parcels an ARTIFACT of 08's tie-break, or real?
     08 resolves multi-match points by DESCENDING area, on the theory that
     you want the containing parcel rather than a sliver. For Regrid that
     reasoning may be backwards -- overlapping records there are often a
     specific parcel stacked under a large tract (subdivision, easement,
     municipal/utility landholding), so largest-first grabs the tract every
     time. If oversized parcels are mostly multi-match, this is self-inflicted
     and the fix is one line. If they're mostly single-match, Regrid genuinely
     has one record covering a whole town and only a window fallback helps.

  2. Where should --max-parcel-km2 be set? Sweeps candidate caps against
     national coverage, weighted by what it costs in scarce Incorrect samples
     (only 398 exist nationally -- that's the binding constraint, not tiles).

Reads the manifest (national, all 2,591 resolved parcels) and, if present,
the Ohio samples output (tile counts, which the manifest doesn't have).

Usage:
    python diagnose_parcel_size.py
    python diagnose_parcel_size.py --state 39
"""
import argparse
import glob
from pathlib import Path

import pandas as pd

REPO = Path("/work/GRDVULN/infrastructure")
MANIFEST = REPO / "data" / "samples" / "training_manifest.parquet"
SAMPLES_GLOB = str(REPO / "data" / "inference_train" / "samples" / "state={st}" / "*.parquet")

CAPS = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0)


def hdr(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def national(man_path: Path):
    m = pd.read_parquet(man_path)
    m = m[m["parcel_found"].astype(bool)].copy()
    m["area_km2"] = m["parcel_area_m2"] / 1e6

    hdr("NATIONAL PARCEL AREA (km2)")
    print(f"resolved parcels: {len(m)}")
    print(m["area_km2"].describe(percentiles=[.5, .75, .9, .95, .99]).to_string())

    hdr("QUESTION 1: are oversized parcels an 08 tie-break artifact?")
    for cap in (1.0, 2.0, 5.0):
        big = m[m["area_km2"] > cap]
        if len(big) == 0:
            print(f"  > {cap} km2: none")
            continue
        multi = int((big["n_parcel_matches"] > 1).sum())
        print(f"  > {cap:>4} km2: {len(big):4d} samples, "
              f"{multi:4d} multi-match ({multi/len(big):5.1%})")
    print("\n  >50% multi-match -> 08's descending-area tie-break is a major cause;")
    print("     switch to smallest-containing and much of this disappears.")
    print("  <20% multi-match -> genuine Regrid town-sized records; the window")
    print("     fallback in 09 is the only remedy.")

    hdr("QUESTION 2: where to set --max-parcel-km2")
    n_inc = int((m["label_class"] == "Incorrect").sum())
    print(f"total Incorrect samples nationally: {n_inc}  (the scarce class)")
    print(f"\n{'cap':>7} {'over':>6} {'pct':>7} {'Incorrect':>10} {'pct_inc':>8}")
    for cap in CAPS:
        over = m["area_km2"] > cap
        inc = int((over & (m["label_class"] == "Incorrect")).sum())
        print(f"{cap:>7.2f} {int(over.sum()):>6d} {over.mean():>6.1%} "
              f"{inc:>10d} {inc/max(n_inc,1):>7.1%}")
    print("\n  These are samples that would switch to WINDOW tiling, not samples")
    print("  dropped -- but the count is the size of the affected subpopulation.")

    hdr("GEOGRAPHY OF THE PROBLEM (states with worst parcel quality)")
    m["over2"] = m["area_km2"] > 2.0
    by_st = (m.groupby("st")
               .agg(n=("sample_id", "size"), over2=("over2", "sum"),
                    median_km2=("area_km2", "median"), max_km2=("area_km2", "max"))
               .sort_values("over2", ascending=False))
    print(by_st[by_st["over2"] > 0].to_string())

    hdr("WORST OFFENDERS NATIONALLY")
    cols = ["sample_id", "label_class", "st", "geoid", "area_km2", "n_parcel_matches"]
    print(m.nlargest(15, "area_km2")[cols].to_string(index=False))

    return m


def state_tiles(st: str, m: pd.DataFrame):
    files = glob.glob(SAMPLES_GLOB.format(st=st))
    if not files:
        print(f"\n(no samples output found for state {st} -- skipping tile analysis)")
        return
    s = pd.concat([pd.read_parquet(f) for f in files])
    s["area_km2"] = s["parcel_area_m2"] / 1e6

    hdr(f"STATE {st}: TILE COST vs PARCEL AREA")
    cols = ["sample_id", "label_class", "geoid", "area_km2", "n_parcel_matches",
            "n_tiles_attempted", "n_objects", "n_distinct_classes"]
    print(s.nlargest(10, "area_km2")[cols].to_string(index=False))

    tot = int(s["n_tiles_attempted"].sum())
    print(f"\ntotal tiles: {tot}   mean: {tot/len(s):.1f}/sample")
    print(f"\n{'cap':>7} {'over':>5} {'tiles_over':>11} {'pct_tiles':>10} {'tiles_after':>12} {'mean_after':>11}")
    for cap in CAPS:
        over = s["area_km2"] > cap
        t_over = int(s.loc[over, "n_tiles_attempted"].sum())
        # A capped sample still costs ~9 tiles for a 600m window.
        after = tot - t_over + int(over.sum()) * 9
        print(f"{cap:>7.2f} {int(over.sum()):>5d} {t_over:>11d} "
              f"{t_over/max(tot,1):>9.1%} {after:>12d} {after/len(s):>10.1f}")

    hdr(f"STATE {st}: DOES A HUGE PARCEL CORRUPT THE FEATURE?")
    s["big"] = s["area_km2"] > 2.0
    if s["big"].any():
        print(s.groupby("big").agg(
            n=("sample_id", "size"),
            median_tiles=("n_tiles_attempted", "median"),
            median_objects=("n_objects", "median"),
            max_objects=("n_objects", "max"),
            mean_classes=("n_distinct_classes", "mean")).to_string())
        print("\n  Inflated n_objects on the big rows is the corruption: detections")
        print("  from an entire town attributed to one plant.")
    else:
        print("  no samples over 2 km2 in this state")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--state", default="39", help="state whose samples output to analyze")
    args = ap.parse_args()

    print("=== diagnose_parcel_size.py ===")
    print(f"manifest: {args.manifest}")

    m = national(Path(args.manifest))
    state_tiles(str(args.state).zfill(2), m)

    hdr("SUMMARY")
    big = m[m["area_km2"] > 2.0]
    if len(big):
        rate = (big["n_parcel_matches"] > 1).mean()
        print(f"  {len(big)} samples over 2 km2 ({len(big)/len(m):.1%})")
        print(f"  {rate:.1%} of them are multi-match")
        if rate > 0.5:
            print("  -> 08's tie-break is a major contributor. Fix it first, re-run 08,")
            print("     then re-measure before choosing a cap.")
        elif rate < 0.2:
            print("  -> genuine Regrid coverage gaps. Window fallback in 09 is the fix;")
            print("     the tie-break change is worth making anyway but won't move much.")
        else:
            print("  -> mixed cause. Do both: fix the tie-break AND add the window fallback.")
    else:
        print("  no oversized parcels nationally -- Ohio may have been unlucky")


if __name__ == "__main__":
    main()
