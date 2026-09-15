"""
check_duplicate_detections.py
==============================
For plants with multiple same-class detections, checks whether they look
like overlap artifacts (close together, low n_merged, different
contributing tiles) or real distinct objects (spread out, or already
n_merged > 1 meaning the dedup already folded multiple raw hits together).

Usage:
    python check_duplicate_detections.py
    python check_duplicate_detections.py --cwns 17000643001
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

CLOSE_THRESHOLD_M = 25  # flag pairs closer than this as suspicious


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cwns", type=str, default=None, help="inspect one plant in detail")
    args = ap.parse_args()

    df = pd.read_csv(C.INFERENCE_DIR / "detections.csv", dtype={"CWNS_ID": str})

    if args.cwns:
        sub = df[df["CWNS_ID"] == args.cwns].sort_values("class")
        print(sub[["class", "lon", "lat", "confidence", "n_merged", "tile_ids"]].to_string(index=False))
        return

    print("Flagging plant/class groups with a same-class pair closer than "
          f"{CLOSE_THRESHOLD_M}m apart that did NOT get merged (n_merged==1 on both):\n")

    flagged = []
    for (cwns, cls), g in df.groupby(["CWNS_ID", "class"]):
        if len(g) < 2:
            continue
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                d = haversine_m(g.loc[i, "lat"], g.loc[i, "lon"], g.loc[j, "lat"], g.loc[j, "lon"])
                if d < CLOSE_THRESHOLD_M:
                    same_tiles = set(str(g.loc[i, "tile_ids"]).split("|")) & \
                                 set(str(g.loc[j, "tile_ids"]).split("|"))
                    flagged.append({
                        "CWNS_ID": cwns, "class": cls, "dist_m": round(d, 1),
                        "n_merged_i": g.loc[i, "n_merged"], "n_merged_j": g.loc[j, "n_merged"],
                        "shared_tiles": len(same_tiles),
                        "tiles_i": g.loc[i, "tile_ids"], "tiles_j": g.loc[j, "tile_ids"],
                    })

    if not flagged:
        print("None found -- no suspiciously close unmerged pairs. Repeated "
              "classes per plant are likely real multi-tank facilities.")
        return

    fdf = pd.DataFrame(flagged).sort_values("dist_m")
    print(fdf.to_string(index=False))
    print(f"\n{len(fdf)} suspicious pair(s) across {fdf['CWNS_ID'].nunique()} plant(s).")
    print("\nIf shared_tiles == 0 (different tiles) and dist_m is small, this is "
          "very likely the same real object seen from two overlapping tiles, "
          "surviving as two objects because their converted centroids landed "
          f">{12}m apart (dedup radius) despite being the same physical thing.")


if __name__ == "__main__":
    main()
