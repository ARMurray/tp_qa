"""
fold_in_500m.py
=================
Copies convert_tiles_to_500m.py's output (rgb_500/, ndwi_500/, labels_500/,
tile_metadata_500.csv) into the SAME live folders label_app.R reads/writes
and 03_prepare_dataset.py/04_train_model.py train from -- data/tiles/rgb/png/,
data/tiles/ndwi/, annotation/ls_export/labels/, data/tile_metadata.csv.

After this runs, the 500m tiles are indistinguishable from any other tile to
every existing tool -- label_app.R can open and edit them like anything
else, 03_prepare_dataset.py picks them up automatically. No code changes
needed anywhere else. Distinguishable later only via tile_metadata.csv's
"source" column (upscaled_500m vs whatever 02_extract_tiles.py's own runs
write) or the _500m tile_id suffix, if you ever want to filter by tile size
(e.g. to check whether mixing 200m/333px and 500m/833px tiles in one YOLO
training run is affecting results -- both are captured at the same native
0.6m/px resolution, so a real object occupies the same pixel count in
either; the difference only shows up if training resizes to a common imgsz,
since 333->imgsz and 833->imgsz are different-direction rescales for the
same real-world object. Not fixed or worked around here -- informational).

SAFETY: never silently overwrites. If a destination file already exists,
it's skipped with a warning, not replaced -- re-running this after a
partial success is safe. tile_metadata.csv is appended and deduplicated on
tile_id (keep the newest), matching every other idempotent-append pattern
in this project.

Usage:
    python fold_in_500m.py
    python fold_in_500m.py --dry-run
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

SRC_RGB_DIR = C.TILES_DIR / "rgb_500"
SRC_NDWI_DIR = C.TILES_DIR / "ndwi_500"
SRC_LABEL_DIR = C.ANNOTATION_DIR / "labels_500"
SRC_METADATA_CSV = C.DATA_DIR / "tile_metadata_500.csv"

DEST_RGB_DIR = C.RGB_DIR
DEST_NDWI_DIR = C.NDWI_DIR
DEST_LABEL_DIR = C.ANNOTATION_DIR / "labels"
DEST_METADATA_CSV = C.METADATA_CSV


def copy_if_absent(src: Path, dest: Path, dry_run: bool) -> str:
    if dest.exists():
        return "skipped (already exists)"
    if dry_run:
        return "would copy"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return "copied"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                     help="show what would happen without touching any files")
    args = ap.parse_args()

    print("=== fold_in_500m.py ===")
    if not SRC_METADATA_CSV.exists():
        print(f"ERROR: {SRC_METADATA_CSV} not found -- run convert_tiles_to_500m.py first.")
        sys.exit(1)

    new_meta = pd.read_csv(SRC_METADATA_CSV, dtype={"CWNS_ID": str, "ll_uuid_primary": str})
    print(f"Found {len(new_meta)} converted tile(s) to fold in")
    if args.dry_run:
        print("--dry-run: no files will be touched\n")

    counts = {"copied": 0, "would copy": 0, "skipped (already exists)": 0, "missing source": 0}

    for _, row in new_meta.iterrows():
        tile_id = row["tile_id"]
        for src_dir, dest_dir, suffix in [
            (SRC_RGB_DIR, DEST_RGB_DIR, "_rgb.png"),
            (SRC_NDWI_DIR, DEST_NDWI_DIR, "_ndwi.tif"),
            (SRC_LABEL_DIR, DEST_LABEL_DIR, "_rgb.txt"),
        ]:
            src = src_dir / f"{tile_id}{suffix}"
            dest = dest_dir / f"{tile_id}{suffix}"
            if not src.exists():
                print(f"  WARNING: {src} not found -- skipping this file")
                counts["missing source"] += 1
                continue
            result = copy_if_absent(src, dest, args.dry_run)
            counts[result] = counts.get(result, 0) + 1

    print(f"\nFile copy summary: {counts}")

    if not args.dry_run:
        if DEST_METADATA_CSV.exists():
            existing = pd.read_csv(DEST_METADATA_CSV, dtype={"CWNS_ID": str, "ll_uuid_primary": str})
            combined = pd.concat([existing, new_meta], ignore_index=True)
            before = len(combined)
            combined = combined.drop_duplicates(subset="tile_id", keep="last")
            dropped = before - len(combined)
            if dropped:
                print(f"  tile_metadata.csv: {dropped} tile_id(s) already present -- kept newest")
        else:
            combined = new_meta
        combined.to_csv(DEST_METADATA_CSV, index=False)
        print(f"\ntile_metadata.csv: {len(combined)} row(s) total "
              f"({len(new_meta)} from this fold-in)")
    else:
        print(f"\n--dry-run: tile_metadata.csv would gain up to {len(new_meta)} row(s)")

    print("\n=== complete ===")
    if not args.dry_run:
        print("The 500m tiles are now live -- label_app.R and "
              "03_prepare_dataset.py will pick them up with no further changes.")


if __name__ == "__main__":
    main()
