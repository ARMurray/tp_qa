"""
Candidate-recall ceiling with every plant location as a search seed (docs/PLAN_20261103_multi_source_seeds.md).

Question: for plants a reviewer CORRECTED, does the candidate window contain the true location?
01a searches a k=K_RINGS H3 (res 9) disk around the CWNS point, and its window is the union of those
disks over every plant in the state. Grid distance is symmetric, so "truth is inside a seed's disk"
is the same as "that seed's cell is inside the disk around the truth cell" -- no parcel data needed.
This is a cell-level check: it does not know whether a parcel exists at the truth point (08's causes b-d).

Windows compared (corrected plants only):
  own CWNS point        disk around this plant's CWNS point
  own, all seeds        disks around this plant's seeds
  state, CWNS points    union over every CWNS point in the state          (today's 01a window)
  state, all seeds      union over every seed in the state                (the multi-seed window)
Also reports the pool-size cost: cells in a plant's own multi-seed window vs a single disk
(sample of plants; cells are a proxy for parcels, which are denser in towns).

Run:  python -I check_seed_recall.py        (build_plant_seeds.py first)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from h3.api import numpy_int as hn

REPO = Path(__file__).resolve().parents[2]
IN = REPO / "correction/data/candidates"
K = 18
RES = 9


def cells(lat, lon):
    return np.array([hn.latlng_to_cell(float(a), float(b), RES) for a, b in zip(lat, lon)], dtype=np.int64)


def main():
    W = pd.read_csv(IN / "plant_agreement.csv", dtype={"cwns_id": str}, low_memory=False)
    S = pd.read_csv(IN / "plant_seeds.csv", dtype={"cwns_id": str, "permit_number": str}, low_memory=False)
    S["cell"] = cells(S.lat, S.lon)
    S["state"] = S.cwns_id.str[:2]
    T = W[W.truth_kind == "corrected"].copy()
    T["cell"] = cells(T.truth_lat, T.truth_lon)
    T["state"] = T.cwns_id.str[:2]
    print(len(T), "corrected plants\n")

    own = {k: g for k, g in S.groupby("cwns_id")}
    st_cwns = {k: set(g.cell) for k, g in S[S.source == "cwns_point"].groupby("state")}
    st_all = {k: set(g.cell) for k, g in S.groupby("state")}

    rows = []
    for r in T.itertuples():
        disk = set(hn.grid_disk(r.cell, K).tolist())
        g = own.get(r.cwns_id)
        hit_src = sorted(set(g.source[g.cell.isin(disk)])) if g is not None else []
        rows.append({
            "cwns_id": r.cwns_id, "state": r.state, "cwns_dist": r.cwns_point_to_truth_m,
            "own_cwns": "cwns_point" in hit_src,
            "own_all": bool(hit_src),
            "state_cwns": not disk.isdisjoint(st_cwns.get(r.state, ())),
            "state_all": not disk.isdisjoint(st_all.get(r.state, ())),
            "hit_src": "+".join(hit_src),
        })
    R = pd.DataFrame(rows)
    n = len(R)
    print("truth inside the candidate window")
    for col, label in [("own_cwns", "own CWNS point"), ("own_all", "own, all seeds"),
                       ("state_cwns", "state, CWNS points (today)"), ("state_all", "state, all seeds")]:
        print(f"  {label:28s} {R[col].sum():4d} / {n}  {R[col].mean():.1%}")

    miss_today = R[~R.state_cwns]
    print(f"\nmissed today: {len(miss_today)}; recovered by own seeds: {int(miss_today.own_all.sum())}, "
          f"by any seed in the state: {int(miss_today.state_all.sum())}")
    print("sources inside the truth's disk, for plants missed today and recovered by own seeds:")
    print(miss_today[miss_today.own_all].hit_src.value_counts().to_string())
    still = R[~R.state_all]
    print(f"\nstill outside every window: {len(still)}; median CWNS-point error {still.cwns_dist.median():,.0f} m, "
          f"min {still.cwns_dist.min():,.0f} m")

    # cost: cells in the plant's own multi-seed window vs one disk
    rng = np.random.default_rng(0)
    sample = rng.choice(list(own), size=min(3000, len(own)), replace=False)
    one = len(hn.grid_disk(0x8928308280fffff, K))
    sizes, n_seeds = [], []
    for cid in sample:
        c = np.unique(own[cid].cell.to_numpy())
        sizes.append(len(np.unique(np.concatenate([hn.grid_disk(int(x), K) for x in c]))))
        n_seeds.append(len(c))
    sizes = np.array(sizes)
    print(f"\npool size (cells) for {len(sample)} sampled plants, single disk = {one}")
    print(f"  own multi-seed window: median {np.median(sizes) / one:.2f}x, mean {sizes.mean() / one:.2f}x, "
          f"p90 {np.percentile(sizes, 90) / one:.2f}x, max {sizes.max() / one:.2f}x")
    print(f"  plants whose window is > 1.5x a single disk: {np.mean(sizes > 1.5 * one):.1%}")


if __name__ == "__main__":
    sys.exit(main())
