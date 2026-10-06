"""
label_priorities.py
===================
Which plants to label next for the detector, ranked -- from the national
corrected output rather than one review round.

extract_review_tiles.py already tiles each closed round's verified true
locations and its false positives. That covers ~150 plants a round. This
looks across ALL verified plants for the cases the CURRENT detector gets
wrong, and writes them in the format `extract_review_tiles.py --sites-csv`
reads, so the tiles land in the labeller's inventory.

PRIORITY (lower number first; decided 2026-10-05, see
docs/PLAN_20261005_detector_population_sites.md)
    1  small_plant_miss   verified plant serving 100-1,000 people where the
                          detector found nothing on its parcel. Small plants
                          (lagoons, package plants) come into scope when the
                          population floor drops to 100; the detector has
                          barely seen them.
    2  miss               verified plant serving > 1,000 where the detector
                          found nothing on its parcel.
    3  small_plant        verified small plant where it did fire -- still
                          worth labelling: the class mix differs from large
                          plants.

"Found nothing" = no object in 13's detections layer inside the plant's
site (correction/diagnostics/output/cwns_detections.parquet). Verified-
CORRECTED plants are left out: their parcel's detections come from 01c,
which has no current-detector output yet, so a miss cannot be told apart
from "never run".

Plants that already have a labelled tile are skipped (label files are named
<CWNS_ID>_<ll_uuid>_..., so any label file for the plant counts).

ALSO PRINTS the label inventory per class. Drying beds are the weakest class
by far (10 instances on 2026-10-05) and cannot be targeted from detections,
because the detector does not find them -- label them wherever they appear.

Usage (from review_app/):
    python -m analysis.label_priorities                 # top 300
    python -m analysis.label_priorities --top 500
then
    python -m analysis.extract_review_tiles --sites-csv ../detection/annotation/label_priorities.csv
(with detection/.venv's Python, like close_round's tile step).
"""
import argparse
import collections
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO / "correction" / "diagnostics" / "output"
POP_CSV = REPO / "correction" / "data" / "cwns" / "POPULATION_WASTEWATER_CONFIRMED_updated06242024.csv"
LABEL_DIR = REPO / "detection" / "annotation" / "ls_export" / "labels"
CLASSES = REPO / "detection" / "annotation" / "ls_export" / "classes.txt"
OUT_CSV = REPO / "detection" / "annotation" / "label_priorities.csv"
SMALL_MIN, SMALL_MAX = 100, 1000


def label_inventory() -> tuple[set[str], collections.Counter, int, int]:
    names = CLASSES.read_text().split() if CLASSES.exists() else []
    labelled, counts, n_files, n_empty = set(), collections.Counter(), 0, 0
    for f in LABEL_DIR.glob("*.txt"):
        n_files += 1
        labelled.add(f.name.split("_", 1)[0])
        rows = [ln.split() for ln in f.read_text().splitlines() if ln.strip()]
        if not rows:
            n_empty += 1
        for r in rows:
            i = int(r[0])
            counts[names[i] if i < len(names) else f"class_{i}"] += 1
    return labelled, counts, n_files, n_empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=300, help="rows to write")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    labelled, counts, n_files, n_empty = label_inventory()
    print("=== label inventory ===")
    print(f"  {n_files:,} label files ({n_empty:,} empty), "
          f"{len(labelled):,} plants with a labelled tile")
    for cls, n in sorted(counts.items(), key=lambda kv: kv[1]):
        print(f"  {cls:<18} {n:>6,}")

    plants = pd.read_parquet(OUTPUT_DIR / "cwns_corrected_locations.parquet")
    dets = pd.read_parquet(OUTPUT_DIR / "cwns_detections.parquet")
    v = plants[(plants["status"] == "verified_correct") & (plants["site_parcels"].fillna("") != "")].copy()
    if "pop_served" in v.columns:
        v["pop"] = v["pop_served"]
    else:
        # 13 output from before 2026-10-05 has no pop_served. The local CWNS
        # file is only a 1,421-row subset, so most plants come out unknown --
        # treated as > 1,000 below rather than dropped.
        print("\nNOTE: 13 output has no pop_served column; re-run 13 for population bands.")
        pop = pd.read_csv(POP_CSV, dtype={"CWNS_ID": str},
                          usecols=["CWNS_ID", "TOTAL_RES_POPULATION_2022"]).drop_duplicates(subset="CWNS_ID")
        pop["pop"] = pd.to_numeric(pop["TOTAL_RES_POPULATION_2022"], errors="coerce")
        v = v.merge(pop[["CWNS_ID", "pop"]], on="CWNS_ID", how="left")
    fired = set(dets["CWNS_ID"].astype(str))
    v["fired"] = v["CWNS_ID"].isin(fired)
    v["small"] = v["pop"].between(SMALL_MIN, SMALL_MAX, inclusive="right")
    v = v[~(v["pop"] <= SMALL_MIN)]        # at/below the new floor: out of scope (unknown kept)
    v = v[~v["CWNS_ID"].isin(labelled)]

    v["priority"] = 0
    v.loc[v["small"] & ~v["fired"], "priority"] = 1
    v.loc[~v["small"] & ~v["fired"], "priority"] = 2
    v.loc[v["small"] & v["fired"], "priority"] = 3
    v = v[v["priority"] > 0]
    v["reason"] = v["priority"].map({1: "small_plant_miss", 2: "miss", 3: "small_plant"})

    print("\n=== candidates (verified correct, unlabelled, population > 100 or unknown) ===")
    for reason, n in v["reason"].value_counts().items():
        print(f"  {reason:<18} {n:>6,}")

    # Spread across states within each priority so a short list is not one state.
    v = v.sample(frac=1.0, random_state=0)
    v["_k"] = v.groupby(["priority", "STATE_CODE"]).cumcount()
    v = v.sort_values(["priority", "_k"]).head(args.top)

    out = pd.DataFrame({
        "cwns_id": v["CWNS_ID"],
        "st": v["STATE_CODE"],
        "ll_uuid": v["site_parcels"].str.split(";").str[0],
        "role": "label_priority_" + v["reason"],
        "lat": None, "lon": None,
        "priority": v["priority"], "reason": v["reason"], "pop": v["pop"],
        "name": v["FACILITY_NAME"],
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {len(out):,} site(s) to {args.out}")
    print("  " + ", ".join(f"{k} {n}" for k, n in out["reason"].value_counts().items()))
    print("\nNext (detection/.venv Python, from review_app/):")
    print(f"  python -m analysis.extract_review_tiles --sites-csv {args.out}")


if __name__ == "__main__":
    main()
