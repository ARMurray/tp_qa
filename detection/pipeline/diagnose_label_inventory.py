"""
diagnose_label_inventory.py
===========================
Partitions annotation/ls_export/labels/ into categories and reports which
labels are genuinely unaccounted for -- i.e. which ones represent real
annotation work that has NOT made it into a 500m tile and is NOT still
recoverable.

Motivating question: labels/ holds ~330 .txt files but rgb_500/ holds only
65 tiles. convert_tiles_to_500m.py collapses many old 200m tiles into one
500m tile, so a large drop is expected -- but "expected" and "confirmed"
are different things, and some of the 330 may be strays that no conversion
ever saw.

Categories reported:
  A. clean_500m      {tile_id}_500m_rgb.txt  -- already a converted 500m tile
  B. clean_200m      {tile_id}_rgb.txt       -- clean name, current round
  C. legacy_mangled  Label Studio export naming (see 03's parse_tile_stem)
  D. unparseable     filename matches no known convention

For B and C, cross-checks against tile_metadata.csv and against the set of
(CWNS_ID, ll_uuid) groups that DID produce a 500m tile. A B/C label whose
group produced a 500m tile is superseded (safe). One whose group did not is
UNCONVERTED -- that is the number that actually matters.

Run from detection/pipeline/:
    python diagnose_label_inventory.py
    python diagnose_label_inventory.py --csv unconverted.csv
"""
import argparse
import importlib.util
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

# Reuse 03's parser rather than reimplementing it -- same importlib pattern
# reconstruct_tile_metadata.py and correction/'s 01c/05 already use, so the
# two copies can never drift.
_spec = importlib.util.spec_from_file_location(
    "prep", Path(__file__).resolve().parent / "03_prepare_dataset.py")
_prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prep)
parse_tile_stem = _prep.parse_tile_stem

LABELS_DIR = C.ANNOTATION_DIR / "labels"

# tile_stem = {cwns_id}_{ll_uuid}_r{row}_c{col}
STEM_RE = re.compile(
    r'^(\d+)_([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})_r(\d+)_c(\d+)$')


def box_count(path: Path) -> int:
    """Non-empty lines. A 0 here is a legitimate confirmed-negative, not a bug."""
    try:
        return sum(1 for ln in path.read_text().splitlines() if ln.strip())
    except Exception:
        return -1


def classify(path: Path):
    """Return (category, cwns_id, ll_uuid) -- ids are None when unknown."""
    name = path.name

    if name.endswith("_500m_rgb.txt"):
        stem = name[: -len("_rgb.txt")]
        base = stem[: -len("_500m")]
        m = re.match(r'^(\d+)_([0-9a-f-]{36})$', base)
        if m:
            return "clean_500m", m.group(1), m.group(2)
        return "clean_500m", None, None

    parsed = parse_tile_stem(name)
    if parsed is None:
        return "unparseable", None, None

    cwns_id, tile_stem = parsed
    m = STEM_RE.match(tile_stem)
    ll_uuid = m.group(2) if m else None

    # Distinguish clean from mangled: a clean name is exactly {tile_stem}_rgb.txt
    category = "clean_200m" if name == f"{tile_stem}_rgb.txt" else "legacy_mangled"
    return category, cwns_id, ll_uuid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="write the unconverted-label detail to this path")
    args = ap.parse_args()

    print("=== diagnose_label_inventory.py ===")
    print(f"labels dir : {LABELS_DIR}")
    if not LABELS_DIR.exists():
        print("ERROR: labels dir does not exist. Check config.py's REPO_ROOT "
              "resolves to detection/, not detection/pipeline/.")
        sys.exit(1)

    label_files = sorted(LABELS_DIR.glob("*.txt"))
    print(f"total .txt : {len(label_files)}\n")

    records = []
    for p in label_files:
        cat, cwns, uuid = classify(p)
        records.append({
            "filename": p.name, "category": cat,
            "CWNS_ID": cwns, "ll_uuid": uuid, "n_boxes": box_count(p),
        })
    df = pd.DataFrame(records)

    print("--- category breakdown ---")
    for cat, n in df["category"].value_counts().items():
        boxes = df.loc[df["category"] == cat, "n_boxes"].clip(lower=0).sum()
        print(f"  {cat:16s} {n:5d} file(s)   {boxes:6d} box(es)")

    empties = int((df["n_boxes"] == 0).sum())
    unread = int((df["n_boxes"] < 0).sum())
    print(f"\n  files with 0 boxes (confirmed-negatives): {empties}")
    if unread:
        print(f"  files that could not be read           : {unread}  <-- investigate")

    # --- which groups actually produced a 500m tile -------------------------
    converted_groups = set(
        df.loc[(df["category"] == "clean_500m") & df["CWNS_ID"].notna(),
               ["CWNS_ID", "ll_uuid"]].itertuples(index=False, name=None))
    print(f"\n--- 500m coverage ---")
    print(f"  converted (CWNS_ID, ll_uuid) group(s): {len(converted_groups)}")

    old = df[df["category"].isin(["clean_200m", "legacy_mangled"])].copy()
    old_groups = set(
        old.loc[old["CWNS_ID"].notna(), ["CWNS_ID", "ll_uuid"]]
           .itertuples(index=False, name=None))
    print(f"  group(s) represented by old 200m labels: {len(old_groups)}")

    unconverted_groups = old_groups - converted_groups
    print(f"  old group(s) with NO 500m counterpart  : {len(unconverted_groups)}")

    old["converted"] = [
        (r.CWNS_ID, r.ll_uuid) in converted_groups for r in old.itertuples()]
    superseded = old[old["converted"]]
    unconverted = old[~old["converted"]]

    print(f"\n  old label files SUPERSEDED by a 500m tile : {len(superseded):5d} "
          f"({superseded['n_boxes'].clip(lower=0).sum()} boxes)")
    print(f"  old label files NOT converted             : {len(unconverted):5d} "
          f"({unconverted['n_boxes'].clip(lower=0).sum()} boxes)  <-- the real number")

    # --- do the unconverted ones have what conversion needs? ----------------
    if len(unconverted):
        if C.METADATA_CSV.exists():
            meta = pd.read_csv(C.METADATA_CSV, dtype={"CWNS_ID": str})
            meta_groups = set(
                meta[["CWNS_ID", "ll_uuid_primary"]].dropna()
                    .itertuples(index=False, name=None))
        else:
            meta_groups = set()
            print("\n  WARNING: tile_metadata.csv not found")

        in_meta = sum(1 for g in unconverted_groups if g in meta_groups)
        print(f"\n  of {len(unconverted_groups)} unconverted group(s):")
        print(f"    {in_meta} have a tile_metadata.csv row "
              f"(convertible now -- rerun convert_tiles_to_500m.py)")
        print(f"    {len(unconverted_groups) - in_meta} do not "
              f"(need reconstruct_tile_metadata.py first, per reference §6)")

        by_cat = Counter(unconverted["category"])
        print(f"\n    by filename convention: {dict(by_cat)}")
        if by_cat.get("legacy_mangled"):
            print("    NOTE: convert_tiles_to_500m.py matches labels by clean "
                  "filename only (line ~230) and will not see mangled names "
                  "even when the metadata row exists. Rename via "
                  "reconstruct_tile_metadata.py before rerunning.")

    if df["category"].eq("unparseable").any():
        print("\n--- unparseable filenames (first 10) ---")
        for n in df.loc[df["category"] == "unparseable", "filename"].head(10):
            print(f"    {n}")

    if args.csv:
        out = unconverted.drop(columns=["converted"]) if len(unconverted) else df.head(0)
        out.to_csv(args.csv, index=False)
        print(f"\nWrote {len(out)} unconverted-label row(s) -> {args.csv}")

    print("\n=== complete ===")


if __name__ == "__main__":
    main()
