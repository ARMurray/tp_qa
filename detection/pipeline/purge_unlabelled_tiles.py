"""
purge_unlabelled_tiles.py
=========================
Deletes every tile in the labelling inventory that has no label file, so the
labeller only holds tiles that already carry labels plus whatever the
targeted selection pulls next.

WHY (2026-10-08)
    The inventory had accumulated tiles from every past selection rule:
    the 190-plant sample, --all-candidates review rounds (every parcel the
    reviewer saw), ring tiles, and the label_priorities list. Most of it is
    undeveloped land, and there are already ~900 confirmed-empty labels, so
    more empty tiles teach the detector nothing. What is worth labelling is
    only (see extract_review_tiles.load_targeted_sites):
      - tiles on a verified true location's parcel(s), and
      - tiles on a parcel where the detector fired but the reviewer said it
        is not the plant (false positives).
    Purge, then re-pull those with
        python -m analysis.extract_review_tiles          (from review_app/)

WHAT COUNTS AS LABELLED
    A label file in any annotation/ls_export/labels*/ folder, matched with
    03_prepare_dataset.parse_tile_stem() -- the same matcher training uses,
    so old Label Studio names ({hash}-{tile}_rgb.txt, URL-encoded paths) keep
    their tile. A zero-byte label (a confirmed negative) counts as labelled:
    it is training data. A label file 03 cannot parse protects any tile whose
    name it contains, as a precaution.

WHAT IS DELETED (only with --apply)
    tiles/rgb/png/{tile}_rgb.png and tiles/ndwi/{tile}_ndwi.tif for every
    unlabelled tile, and their rows in tile_metadata.csv (backed up first as
    tile_metadata.before-purge-<stamp>.csv). Dropping the rows matters: the
    tile fetchers skip any tile id already in the metadata, so without it the
    targeted re-pull would skip tiles it should fetch again.
    Tiles are re-fetchable from NAIP; labels are never touched.

Usage (detection/.venv, from detection/pipeline/):
    python purge_unlabelled_tiles.py            # dry run: counts only
    python purge_unlabelled_tiles.py --apply    # delete
"""
import argparse
import importlib.util
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

_spec = importlib.util.spec_from_file_location(
    "prepare03", Path(__file__).resolve().parent / "03_prepare_dataset.py")
_prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prep)          # __name__ != "__main__": main() does not run


def labelled_tiles() -> tuple[set[str], list[str], int]:
    """Tile ids with a label file, plus the raw names of label files 03
    cannot parse (used as a substring guard)."""
    stems, unparsed, n = set(), [], 0
    for d in sorted(C.ANNOTATION_DIR.glob("labels*")):
        if not d.is_dir():
            continue
        for f in d.glob("*.txt"):
            n += 1
            r = _prep.parse_tile_stem(f.name)
            if r is None:
                unparsed.append(f.stem)
            else:
                stems.add(r[1])
    return stems, unparsed, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default: dry run, counts only)")
    args = ap.parse_args()

    labelled, unparsed, n_label_files = labelled_tiles()
    print(f"{n_label_files:,} label file(s) -> {len(labelled):,} labelled tile id(s)")
    if unparsed:
        print(f"  {len(unparsed)} label file(s) 03 cannot parse -- any tile whose "
              f"name they contain is kept: {unparsed[:5]}")

    def is_labelled(tid: str) -> bool:
        return tid in labelled or any(tid in u for u in unparsed)

    pngs = sorted(C.RGB_DIR.glob("*_rgb.png"))
    tifs = sorted(C.NDWI_DIR.glob("*_ndwi.tif")) if C.NDWI_DIR.exists() else []
    tid_png = {p.name[:-len("_rgb.png")]: p for p in pngs}
    tid_tif = {p.name[:-len("_ndwi.tif")]: p for p in tifs}
    drop = sorted(t for t in set(tid_png) | set(tid_tif) if not is_labelled(t))
    keep = sorted(t for t in tid_png if is_labelled(t))

    print(f"\n{len(pngs):,} RGB tile(s) in {C.RGB_DIR}")
    print(f"  keep   {len(keep):,} (labelled)")
    print(f"  delete {sum(t in tid_png for t in drop):,} RGB + "
          f"{sum(t in tid_tif for t in drop):,} NDWI (unlabelled)")
    missing = labelled - set(tid_png)
    if missing:
        print(f"  note: {len(missing):,} labelled tile(s) have no PNG here; "
              f"03 cannot train on those (e.g. {sorted(missing)[:3]})")

    meta = None
    if C.METADATA_CSV.exists():
        meta = pd.read_csv(C.METADATA_CSV, dtype=str)
        meta["tile_id"] = meta["tile_id"].fillna("")
        dropping = ~meta["tile_id"].map(is_labelled)
        why = Counter(zip(meta.loc[dropping, "source"].fillna(""),
                          meta.loc[dropping, "ll_uuid_alternates"].fillna("")))
        print(f"\ntile_metadata.csv: {len(meta):,} row(s), {int(dropping.sum()):,} "
              f"to drop (no label). By source / role:")
        for (src, role), n in why.most_common(15):
            print(f"  {n:>7,}  {src or '(blank)':<20} {role or '(blank)'}")

    if not args.apply:
        print("\nDry run -- nothing deleted. Re-run with --apply.")
        return

    for t in drop:
        for p in (tid_png.get(t), tid_tif.get(t)):
            if p is not None and p.exists():
                p.unlink()
    print(f"\nDeleted files for {len(drop):,} tile(s).")

    if meta is not None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = C.METADATA_CSV.with_name(f"{C.METADATA_CSV.stem}.before-purge-{stamp}.csv")
        shutil.copy2(C.METADATA_CSV, backup)
        kept = meta[~dropping]
        kept.to_csv(C.METADATA_CSV, index=False)
        print(f"tile_metadata.csv: {len(meta):,} -> {len(kept):,} rows (backup {backup.name})")

    print("\nNext, from review_app/ with detection/.venv's Python:")
    print("  python -m analysis.extract_review_tiles --dry-run")
    print("  python -m analysis.extract_review_tiles")
    print("then restart label_app.R.")


if __name__ == "__main__":
    main()
