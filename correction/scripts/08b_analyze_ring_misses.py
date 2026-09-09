"""
08b_analyze_ring_misses.py
============================
Reads 08's correction_coverage.parquet and answers one question: would raising
K_RINGS actually recover the corrections currently outside the search window,
and at what cost?

Run this whenever K_RINGS is up for reconsideration. It needs no compute --
just the diagnostic parquet 08 already wrote.

WHY THIS IS A SEPARATE SCRIPT: 08's inline suggestion ("k=N would capture 95%
of them") is a bad heuristic and should be ignored. Taking a high percentile of
a distribution that runs from ring 20 to ring 948 produces a meaningless number
-- a 948-ring miss is ~295 km, which is not a search-radius failure at all but
a plant recorded in the wrong county or state. No radius fixes those. This
script separates the two populations instead of averaging over them.

The three categories it splits on:

  NEAR MISSES (just outside the ring)
      Genuinely a radius problem. Recoverable by raising K_RINGS, and the
      recovery curve below prices that.

  FAR MISSES (well outside, but plausibly the same metro area)
      Recoverable in principle, but at a pool cost that likely hurts more than
      it helps -- see the ranking-dilution note below.

  RECORDS ERRORS (tens to hundreds of km)
      Not a geometry problem. These need a second candidate source -- ECHO/ICIS
      facility coordinates, NPDES outfall locations, or geocoding the plant's
      address -- not a wider ring.

THE COST THAT IS NOT COMPUTE: every extra candidate is another distractor
Stage 2a must rank below the truth. Candidate count scales ~k^2, so doubling
the radius quadruples the pool. Candidate recall rises while recall@1 can
FALL. That tradeoff, not runtime, is the reason to keep k tight.

Decision as of 2026-08-24: K_RINGS stays at 18 through pilot/testing. Revisit
once the full workflow runs end to end and the expensive stages (01a/01b) are
already banked -- at that point a wider ring costs an incremental run rather
than a full recompute.

Usage:
    python 08b_analyze_ring_misses.py
    python 08b_analyze_ring_misses.py --records-error-km 50 --candidate-ks 20,25,30,40
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage-file", type=str, default=None,
                    help="default: data/diagnostics/correction_coverage.parquet")
    ap.add_argument("--records-error-km", type=float, default=50.0,
                    help="above this distance a miss is treated as a records error, "
                         "not a radius problem (default 50)")
    ap.add_argument("--candidate-ks", type=str, default="20,22,25,30,35,40,50",
                    help="comma-separated k values to price in the recovery curve")
    ap.add_argument("--current-k", type=int, default=C.K_RINGS)
    args = ap.parse_args()

    path = Path(args.coverage_file) if args.coverage_file else \
        C.DATA_DIR / "diagnostics" / "correction_coverage.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found. Run 08_diagnose_candidate_coverage.py first.")
        return

    d = pd.read_parquet(path)
    print("=== 08b_analyze_ring_misses.py ===")
    print(f"Source: {path.name}   |   current K_RINGS = {args.current_k}\n")

    total = len(d)
    usable = int((d["status"] == "ok_usable").sum())
    out = d[d["status"] == "a_outside_search_window"].copy()
    out["distance_km"] = out["distance_m"] / 1000.0
    out = out.sort_values("ring_distance")

    print(f"Corrections total            : {total}")
    print(f"Currently usable (ceiling)   : {usable}  ({100 * usable / total:.1f}%)")
    print(f"Outside the search window    : {len(out)}  ({100 * len(out) / total:.1f}%)\n")
    if not len(out):
        print("Nothing outside the window -- K_RINGS is not a constraint.")
        return

    # ---- Ring-distance distribution ----
    print("--- Ring distance of the misses (current ring ends at "
          f"k={args.current_k}) ---")
    bins = [args.current_k, args.current_k + 7, args.current_k + 17,
            args.current_k + 32, 100, 250, 10_000]
    labels = [f"{args.current_k + 1}-{args.current_k + 7} (near miss)",
              f"{args.current_k + 8}-{args.current_k + 17}",
              f"{args.current_k + 18}-{args.current_k + 32}",
              f"{args.current_k + 33}-100", "101-250", "250+"]
    binned = pd.cut(out["ring_distance"], bins=bins, labels=labels, right=True)
    counts = binned.value_counts().sort_index()
    for lab, n in counts.items():
        bar = "#" * int(n)
        print(f"  ring {lab:>28s} : {n:3d}  {bar}")

    print(f"\n  ring distance  min={out['ring_distance'].min():.0f}  "
          f"median={out['ring_distance'].median():.0f}  "
          f"max={out['ring_distance'].max():.0f}")
    print(f"  distance (km)  min={out['distance_km'].min():.1f}  "
          f"median={out['distance_km'].median():.1f}  "
          f"max={out['distance_km'].max():.1f}")

    # ---- Records errors vs radius problems ----
    records = out[out["distance_km"] > args.records_error_km]
    radius = out[out["distance_km"] <= args.records_error_km]
    print(f"\n--- Split at {args.records_error_km:.0f} km ---")
    print(f"  radius problems (<= {args.records_error_km:.0f} km) : {len(radius)}  "
          f"-- a wider ring could reach these")
    print(f"  records errors  (>  {args.records_error_km:.0f} km) : {len(records)}  "
          f"-- no radius reaches these; needs a second candidate source")
    print(f"\n  Even an unbounded ring leaves the ceiling at "
          f"{100 * (usable + len(radius)) / total:.1f}% "
          f"({usable + len(radius)}/{total}) unless the records errors are "
          f"handled separately.")

    if len(records):
        print(f"\n  Records-error plants (candidates for ECHO/ICIS cross-reference "
              f"or address geocoding):")
        cols = [c for c in ["CWNS_ID", "STATE_CODE", "distance_km", "ring_distance"]
                if c in records.columns]
        show = records[cols].sort_values("distance_km", ascending=False)
        print(show.head(25).to_string(index=False,
                                      float_format=lambda v: f"{v:.1f}"))
        if len(show) > 25:
            print(f"    ... and {len(show) - 25} more")

    # ---- Recovery curve ----
    print("\n--- Recovery curve: what each candidate k buys ---")
    print("  'pool' is the approximate multiple of today's candidate count "
          "(scales ~k^2).")
    print("  'per_recovered' is extra pool multiple per correction recovered -- "
          "lower is better.\n")
    ks = [int(k) for k in args.candidate_ks.split(",")]
    rows = []
    for k in ks:
        rec = int((out["ring_distance"] <= k).sum())
        rec_radius = int((radius["ring_distance"] <= k).sum())
        pool = (k / args.current_k) ** 2
        new_ceiling = 100 * (usable + rec) / total
        rows.append(dict(
            k=k, recovered=rec, of_which_radius=rec_radius,
            new_ceiling_pct=round(new_ceiling, 1),
            pool_multiple=round(pool, 2),
            per_recovered=round((pool - 1) / rec, 3) if rec else np.nan,
        ))
    curve = pd.DataFrame(rows)
    print(curve.to_string(index=False))

    best = curve[curve["recovered"] > 0]
    if len(best):
        knee = best.loc[best["per_recovered"].idxmin()]
        print(f"\n  Best cost-per-recovery in this range: k={int(knee['k'])} "
              f"({int(knee['recovered'])} recovered, "
              f"{knee['pool_multiple']:.2f}x pool, ceiling "
              f"{knee['new_ceiling_pct']:.1f}%)")
        print("  Read that as an upper bound on the benefit, not a recommendation: "
              "it prices\n  candidate RECALL only. It does not price the recall@1 "
              "lost to ranking\n  dilution, which is the cost that actually matters "
              "and is not knowable\n  until the holdout exists.")

    # ---- State concentration ----
    if "STATE_CODE" in out.columns:
        by_state = out["STATE_CODE"].value_counts()
        if len(by_state) and by_state.iloc[0] >= 3:
            print("\n--- States contributing the most misses ---")
            print("  (a concentrated state may indicate a parcel-coverage or "
                  "coordinate-convention\n   problem there, which is cheaper to "
                  "fix than a global radius change)")
            for st, n in by_state.head(8).items():
                print(f"    {st}: {n}")

    outdir = C.DATA_DIR / "diagnostics"
    curve.to_csv(outdir / "ring_recovery_curve.csv", index=False)
    keep = [c for c in ["CWNS_ID", "STATE_CODE", "distance_km", "ring_distance",
                        "status", "corrected_ll_uuid"] if c in out.columns]
    out[keep].to_csv(outdir / "outside_window_corrections.csv", index=False)
    print(f"\nWritten: {outdir / 'ring_recovery_curve.csv'}")
    print(f"Written: {outdir / 'outside_window_corrections.csv'}")
    print("\n=== complete ===")


if __name__ == "__main__":
    main()
