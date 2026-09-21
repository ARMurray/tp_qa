"""
dedupe_existing_tiles.py
=========================
Lives in review_app/analysis/, next to extract_review_tiles.py. One-time
(or periodic) cleanup pass over the REAL detection/data/tiles/ inventory:
finds tiles from DIFFERENT ll_uuid parcels (same cwns_id) whose bounding
boxes substantially overlap -- duplicates of the kind
dedupe_overlapping_sites() in extract_review_tiles.py now prevents going
forward, for whatever already got downloaded before that fix existed.

WHY THIS EXISTS
    Confirmed 2026-09-16: extract_review_tiles.py tiled two Regrid parcel
    records that physically overlap for the same plant (adjoining
    tracts/split lots), producing near-identical tiles at slightly
    different tile offsets (e.g. ..._r02_c01 vs ..._r03_c02, same two
    lagoons). Left in place these waste labeling effort and, worse, risk
    the same real-world scene landing in both a train and a val split as
    if independent -- this cleans up what's already on disk; it doesn't
    touch how future runs behave (that's extract_review_tiles.py's own
    dedupe_overlapping_sites(), which runs before any fetch, not after).

HOW DUPLICATES ARE FOUND
    Within each CWNS_ID, tiles are pairwise-compared by their WGS84 bbox
    (already in tile_metadata.csv -- no imagery needs to be opened). Two
    tiles with IoU above --iou-threshold (default 0.6) are one duplicate
    cluster; clusters are built with a simple union-find so a chain of
    overlaps (A~B, B~C) collapses to one group even if A and C don't
    directly overlap enough on their own.

    Scoped to source in {review_reported, review_candidate} by default --
    the two sources extract_review_tiles.py writes, and where the
    overlapping-parcel duplication actually comes from. Pass
    --all-sources to widen the check to every tile in the metadata (a
    plant/review_fp tile from a normal 02_extract_tiles.py run could in
    principle also duplicate a review tile of the same plant).

WHAT GETS KEPT
    - A tile that already has a label file (best-effort substring match
      against annotation/ls_export/labels/, same approach
      extract_review_tiles.py's --check-labels uses) is NEVER deleted,
      full stop. If a cluster contains more than one labeled tile, that's
      flagged as a WARNING to resolve by hand -- real duplicate labels are
      a training-set correctness problem, not something to silently
      pick a side on.
    - Otherwise, one representative per cluster is kept: source
      'review_reported' beats 'review_candidate' (the reported location is
      the more grounded signal), then the lexicographically smallest
      tile_id, for a deterministic tie-break.

WHAT HAPPENS TO THE REST
    Their rgb_path/ndwi_path files are deleted from disk and their rows
    are dropped from tile_metadata.csv. tile_metadata.csv is copied to
    tile_metadata.csv.bak-<timestamp> first -- always, on every --apply
    run -- since this rewrites a hand-curated file.

DEFAULT IS DRY-RUN. Nothing is deleted or rewritten without --apply.

Usage:
    python dedupe_existing_tiles.py                 # report only
    python dedupe_existing_tiles.py --apply         # actually delete + rewrite
    python dedupe_existing_tiles.py --iou-threshold 0.5
    python dedupe_existing_tiles.py --all-sources
    python dedupe_existing_tiles.py --detection-root "D:\\somewhere\\else\\detection"
"""
import argparse
import importlib.util
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

# review_app/config.py -- one level up from analysis/. Only needed here for
# TP_QA_ROOT (to default --detection-root); this script otherwise never
# touches review_app's own data.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

TP_QA_ROOT = C.APP_ROOT.parent
DETECTION_PIPELINE_DIR_DEFAULT = TP_QA_ROOT / "detection" / "pipeline"

REVIEW_SOURCES = {"review_reported", "review_candidate"}


def load_detection_config(pipeline_dir: Path):
    """Loads just detection/pipeline/config.py -- evicting the cached
    review_app 'config' module first, same collision as
    extract_review_tiles.py's load_detection_modules() and for the same
    reason (both repos have a bare config.py under the same module name)."""
    if not pipeline_dir.exists():
        raise SystemExit(f"detection pipeline not found at {pipeline_dir} -- "
                          f"pass --detection-root if it lives somewhere else.")
    sys.modules.pop("config", None)
    spec = importlib.util.spec_from_file_location("config", pipeline_dir / "config.py")
    det_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(det_config)
    return det_config


# ===========================================================================
# Union-find, for collapsing chains of overlap into one cluster
# ===========================================================================
class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def bbox_iou(b1, b2) -> float:
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# ===========================================================================
# Label check -- same best-effort substring match as
# extract_review_tiles.py's --check-labels
# ===========================================================================
def load_labeled_stems(labels_dir: Path) -> set:
    if not labels_dir.exists():
        print(f"(labels dir not found at {labels_dir} -- treating nothing as labeled)")
        return set()
    return {p.stem for p in labels_dir.glob("*.txt")}


def is_labeled(tile_id: str, labeled_stems: set) -> bool:
    return any(tile_id in stem for stem in labeled_stems)


# ===========================================================================
# Find duplicate clusters
# ===========================================================================
def find_clusters(df: pd.DataFrame, iou_threshold: float) -> list[list[int]]:
    clusters = []
    for cwns_id, grp in df.groupby("CWNS_ID"):
        idxs = grp.index.tolist()
        if len(idxs) < 2:
            continue
        uf = UnionFind(idxs)
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                # Only tiles from DIFFERENT parcels can be this kind of
                # duplicate -- two tiles from the SAME ll_uuid at
                # different row/col are legitimately different ground.
                if df.loc[i, "ll_uuid_primary"] == df.loc[j, "ll_uuid_primary"]:
                    continue
                b1 = (df.loc[i, "bbox_xmin"], df.loc[i, "bbox_ymin"],
                      df.loc[i, "bbox_xmax"], df.loc[i, "bbox_ymax"])
                b2 = (df.loc[j, "bbox_xmin"], df.loc[j, "bbox_ymin"],
                      df.loc[j, "bbox_xmax"], df.loc[j, "bbox_ymax"])
                if bbox_iou(b1, b2) > iou_threshold:
                    uf.union(i, j)

        groups = defaultdict(list)
        for i in idxs:
            groups[uf.find(i)].append(i)
        clusters.extend(g for g in groups.values() if len(g) > 1)
    return clusters


def pick_keepers(df: pd.DataFrame, cluster: list[int], labeled_stems: set) -> tuple[list[int], list[int], list[int]]:
    """Returns (keep, drop, label_conflict) index lists for one cluster."""
    labeled = [i for i in cluster if is_labeled(df.loc[i, "tile_id"], labeled_stems)]
    if len(labeled) > 1:
        return cluster, [], labeled          # flagged, nothing auto-dropped
    if len(labeled) == 1:
        keep = labeled
        drop = [i for i in cluster if i not in keep]
        return keep, drop, []

    # No labels anywhere in the cluster -- pick deterministically.
    def rank(i):
        source = df.loc[i, "source"]
        return (0 if source == "review_reported" else 1, df.loc[i, "tile_id"])

    ordered = sorted(cluster, key=rank)
    return [ordered[0]], ordered[1:], []


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detection-root", type=Path,
                    default=DETECTION_PIPELINE_DIR_DEFAULT,
                    help=f"path to detection/pipeline/ "
                         f"(default: {DETECTION_PIPELINE_DIR_DEFAULT})")
    ap.add_argument("--iou-threshold", type=float, default=0.6,
                    help="bbox IoU above which two different-parcel tiles "
                         "for the same plant count as duplicates (default: 0.6)")
    ap.add_argument("--all-sources", action="store_true",
                    help="check every tile, not just review_reported/"
                         "review_candidate")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete files and rewrite tile_metadata.csv "
                         "(default is a dry run / report only)")
    args = ap.parse_args()

    print("=== dedupe_existing_tiles.py ===")
    DC = load_detection_config(args.detection_root)

    df = pd.read_csv(DC.METADATA_CSV, dtype={
        "tile_id": str, "CWNS_ID": str, "TRI_FACILITY_ID": str,
        "ll_uuid_primary": str, "ll_uuid_alternates": str, "st": str,
        "geoid": str, "source": str,
    })
    print(f"{len(df)} tile(s) in {DC.METADATA_CSV}")

    scope = df if args.all_sources else df[df["source"].isin(REVIEW_SOURCES)]
    print(f"Checking {len(scope)} tile(s) "
          f"({'all sources' if args.all_sources else 'review_reported + review_candidate'})")
    if scope.empty:
        print("Nothing to check.")
        return

    labeled_stems = load_labeled_stems(DC.ANNOTATION_DIR / "labels")
    print(f"{len(labeled_stems)} labeled tile(s) on disk")

    clusters = find_clusters(scope, args.iou_threshold)
    if not clusters:
        print(f"\nNo duplicate clusters found at IoU > {args.iou_threshold}.")
        return

    print(f"\n{len(clusters)} duplicate cluster(s) found:\n")
    to_drop, conflicts = [], []
    for cluster in clusters:
        cwns_id = df.loc[cluster[0], "CWNS_ID"]
        print(f"  CWNS_ID {cwns_id}: {len(cluster)} overlapping tile(s)")
        for i in cluster:
            row = df.loc[i]
            flag = " [LABELED]" if is_labeled(row["tile_id"], labeled_stems) else ""
            print(f"    - {row['tile_id']} ({row['source']}){flag}")

        keep, drop, label_conflict = pick_keepers(df, cluster, labeled_stems)
        if label_conflict:
            print(f"    WARNING: {len(label_conflict)} of these are ALREADY "
                  f"LABELED -- not auto-resolving, review by hand")
            conflicts.append(cluster)
        else:
            print(f"    -> keeping {df.loc[keep[0], 'tile_id']}, "
                  f"dropping {len(drop)}")
            to_drop.extend(drop)
        print()

    print(f"--- summary ---")
    print(f"  {len(clusters)} cluster(s), {len(conflicts)} need manual review "
          f"(multiple labeled tiles)")
    print(f"  {len(to_drop)} tile(s) would be deleted")

    if not args.apply:
        print("\nDry run -- nothing deleted. Re-run with --apply to actually "
              "remove files and rewrite tile_metadata.csv.")
        return

    if not to_drop:
        print("\nNothing to apply (all clusters need manual review).")
        return

    backup = DC.METADATA_CSV.with_suffix(
        f".csv.bak-{datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy2(DC.METADATA_CSV, backup)
    print(f"\nBacked up metadata to {backup}")

    n_files_deleted = 0
    for i in to_drop:
        row = df.loc[i]
        for col in ("rgb_path", "ndwi_path"):
            p = Path(row[col]) if pd.notna(row[col]) and row[col] else None
            if p and p.exists():
                p.unlink()
                n_files_deleted += 1

    full_df = pd.read_csv(DC.METADATA_CSV, dtype={"tile_id": str, "CWNS_ID": str})
    dropped_ids = set(df.loc[to_drop, "tile_id"])
    kept_df = full_df[~full_df["tile_id"].isin(dropped_ids)]
    kept_df.to_csv(DC.METADATA_CSV, index=False)

    print(f"Deleted {n_files_deleted} file(s), removed {len(dropped_ids)} row(s) "
          f"from tile_metadata.csv")
    print(f"tile_metadata.csv now has {len(kept_df)} row(s) "
          f"(was {len(full_df)})")


if __name__ == "__main__":
    main()
