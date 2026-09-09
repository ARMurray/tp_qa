"""
prepare_dataset.py
------------------
Reconstructs minimal metadata from tile filenames, matches Label Studio
YOLO exports to source images, performs a plant-level train/val split,
and populates the dataset/ directory structure for YOLOv8 training.

Expected folder layout (all relative to this script):
  tiles/rgb/          - RGB PNG tiles
  tiles/ndwi/         - NDWI TIF tiles (not used yet, reserved for Phase 3b)
  ls_export/labels/   - YOLO .txt label files from Label Studio export
  ls_export/classes.txt
  dataset/            - created by this script
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

import os
import re
import shutil
import random
from pathlib import Path
from collections import defaultdict


# --- paths/config now come from config.py ---
RGB_DIR       = C.RGB_DIR
LS_LABELS_DIR = C.ANNOTATION_DIR / "labels"
CLASSES_FILE  = C.CLASSES_FILE
DATASET_DIR   = C.DATASET_DIR
YAML_PATH     = C.DATASET_YAML
RANDOM_SEED   = C.RANDOM_SEED

VAL_FRACTION    = 0.2     # 20% of plants held out for validation
RANDOM_SEED     = 42
INCLUDE_NEGATIVES = True  # include tiles with no annotations (empty label files)
# ---------------------------------------------------------------------------


_UUID = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'

# Parcel-keyed tiles. The trailing group is EITHER a row/col pair (the
# 200m grid and the current one-or-nine 500m grid) OR the literal "500m"
# (convert_tiles_to_500m.py names one tile per site, no row/col at all).
#
# The "500m" alternative was added 2026-09-03. Without it every one of the
# 65 folded-in 500m tiles failed to parse and was dropped from the dataset
# with only a "Could not parse" warning -- fold_in_500m.py's docstring
# claims 03 picks them up automatically, which was never actually true.
_RE_PARCEL = re.compile(rf'^(\d+)_({_UUID})_(r\d+_c\d+|500m)$')

# TRI hard negatives carry no parcel uuid and lead with letters, so they
# never matched the parcel pattern either. Unlabeled today, but they would
# have been silently dropped the moment one was annotated.
_RE_TRI = re.compile(r'^TRI_([A-Za-z0-9]+)_(r\d+_c\d+)$')


def parse_tile_stem(label_filename: str) -> tuple[str, str] | None:
    """
    Extract (site_id, tile_stem) from a label filename.

    Filename patterns handled:
      1. Path-encoded: {hash}__Users%5CAMURRA02%5Cpng%5C{tile_stem}_rgb.txt
      2. Clean:        {hash}-{tile_stem}_rgb.txt
      3. Clean, no hash: {tile_stem}_rgb.txt   (what label_app.R writes now)

    tile_stem forms:
      17000281001_6fcc5c83-...-143ad9895c1e_r01_c01   parcel tile
      17000281001_6fcc5c83-...-143ad9895c1e_500m      folded 500m tile
      TRI_12345ABCDE_r01_c01                          TRI negative

    site_id is the CWNS_ID for plant/review tiles and "TRI_{id}" for TRI
    ones. It is the train/val grouping key, so it must be stable per real
    site -- every tile of one site has to fall on the same side of the
    split or the val score is inflated by leakage.

    Returns (site_id, tile_stem), or None if parsing fails.
    """
    name = Path(label_filename).stem  # drop .txt

    # Pattern 1: URL-encoded path — grab everything after the last %5C
    if "%5C" in name:
        name = name.split("%5C")[-1]
    else:
        # Pattern 2: strip leading 8-char hex hash and separator.
        # Safe against clean names because CWNS_IDs are 11 digits, so the
        # 9th character is never the required [-_] separator.
        name = re.sub(r'^[0-9a-f]{8}[-_]', '', name)

    # name should now be "{tile_stem}_rgb" or just "{tile_stem}"
    name = re.sub(r'_rgb$', '', name)

    m = _RE_PARCEL.match(name)
    if m:
        cwns_id, ll_uuid, suffix = m.group(1), m.group(2), m.group(3)
        return cwns_id, f"{cwns_id}_{ll_uuid}_{suffix}"

    m = _RE_TRI.match(name)
    if m:
        tri_id, rowcol = m.group(1), m.group(2)
        return f"TRI_{tri_id}", f"TRI_{tri_id}_{rowcol}"

    return None


def load_classes(classes_file: Path) -> list[str]:
    with open(classes_file) as f:
        return [line.strip() for line in f if line.strip()]


def build_label_index(ls_labels_dir: Path) -> dict[str, Path]:
    """
    Returns {tile_stem: label_path} for every parseable label file.
    Logs any files that couldn't be parsed.
    """
    index = {}
    unparsed = []
    for label_file in ls_labels_dir.glob("*.txt"):
        result = parse_tile_stem(label_file.name)
        if result is None:
            unparsed.append(label_file.name)
            continue
        cwns_id, tile_stem = result
        index[tile_stem] = label_file

    if unparsed:
        print(f"  WARNING: Could not parse {len(unparsed)} label filenames:")
        for f in unparsed[:10]:
            print(f"    {f}")
        if len(unparsed) > 10:
            print(f"    ... and {len(unparsed) - 10} more")

    return index


def build_rgb_index(rgb_dir: Path) -> dict[str, Path]:
    """Returns {tile_stem: rgb_path} for every PNG in tiles/rgb/."""
    index = {}
    for png in rgb_dir.glob("*.png"):
        stem = re.sub(r'_rgb$', '', png.stem)
        index[stem] = png
    return index


def split_by_plant(
    matched: list[tuple[str, Path, Path]],
    val_fraction: float,
    seed: int
) -> tuple[list, list]:
    """
    matched: list of (cwns_id, rgb_path, label_path)
    Splits by CWNS_ID so no plant appears in both train and val.
    Returns (train_list, val_list) of the same tuples.
    """
    # Group tile indices by plant
    plant_to_tiles = defaultdict(list)
    for i, (cwns_id, rgb_path, label_path) in enumerate(matched):
        plant_to_tiles[cwns_id].append(i)

    plants = list(plant_to_tiles.keys())
    random.seed(seed)
    random.shuffle(plants)

    n_val = max(1, round(len(plants) * val_fraction))
    val_plants  = set(plants[:n_val])
    train_plants = set(plants[n_val:])

    train = [matched[i] for p in train_plants for i in plant_to_tiles[p]]
    val   = [matched[i] for p in val_plants   for i in plant_to_tiles[p]]

    return train, val


def copy_split(split: list, split_name: str, dataset_dir: Path):
    img_dir = dataset_dir / "images" / split_name
    lbl_dir = dataset_dir / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for cwns_id, rgb_path, label_path in split:
        shutil.copy2(rgb_path,   img_dir / rgb_path.name)
        shutil.copy2(label_path, lbl_dir / label_path.name.replace(
            label_path.stem, rgb_path.stem  # make label filename match image filename
        ))


def write_yaml(yaml_path: Path, dataset_dir: Path, classes: list[str]):
    lines = [
        f"path: {dataset_dir.as_posix()}",
        "train: images/train",
        "val:   images/val",
        "",
        f"nc: {len(classes)}",
        f"names: {classes}",
        "",
    ]
    with open(yaml_path, "w") as f:
        f.write("\n".join(lines))


def is_negative(label_path: Path) -> bool:
    """True if the label file is empty (no annotations)."""
    return label_path.stat().st_size == 0


def main():
    print("=== prepare_dataset.py ===\n")

    # 1. Load classes
    classes = load_classes(CLASSES_FILE)
    print(f"Classes ({len(classes)}): {classes}\n")

    # 2. Build indexes
    print("Indexing label files...")
    label_index = build_label_index(LS_LABELS_DIR)
    print(f"  Parsed {len(label_index)} label files\n")

    print("Indexing RGB tiles...")
    rgb_index = build_rgb_index(RGB_DIR)
    print(f"  Found {len(rgb_index)} RGB tiles\n")

    # 3. Match labels to images
    matched = []
    missing_rgb = []
    n_positive = 0
    n_negative = 0

    for tile_stem, label_path in label_index.items():
        if tile_stem not in rgb_index:
            missing_rgb.append(tile_stem)
            continue

        negative = is_negative(label_path)
        if negative and not INCLUDE_NEGATIVES:
            n_negative += 1
            continue

        result = parse_tile_stem(label_path.name)
        cwns_id = result[0] if result else "unknown"

        matched.append((cwns_id, rgb_index[tile_stem], label_path))
        if negative:
            n_negative += 1
        else:
            n_positive += 1

    print(f"Matched {len(matched)} tiles to labels")
    print(f"  Positive (annotated): {n_positive}")
    print(f"  Negative (empty):     {n_negative}")

    # Break down by tile family. The 500m count here is the check that
    # matters after a fold-in: if it reads 0, the folded tiles are being
    # dropped again and nothing else in this output will say so.
    families = defaultdict(int)
    for _, rgb_path, _ in matched:
        stem = rgb_path.stem
        if stem.startswith("TRI_"):
            families["TRI negatives"] += 1
        elif "_500m" in stem:
            families["500m folded (833px)"] += 1
        else:
            families["parcel grid tiles"] += 1
    print("  By tile family:")
    for fam, k in sorted(families.items()):
        print(f"    {fam:24s} {k}")
    if missing_rgb:
        print(f"  WARNING: {len(missing_rgb)} label files had no matching RGB tile")
        for s in missing_rgb[:5]:
            print(f"    {s}")
    print()

    if not matched:
        print("ERROR: No matched tiles found. Check your paths and filename patterns.")
        return

    # 4. Plant-level train/val split
    train, val = split_by_plant(matched, VAL_FRACTION, RANDOM_SEED)

    # Count plants in each split
    train_plants = len(set(c for c, _, _ in train))
    val_plants   = len(set(c for c, _, _ in val))
    print(f"Train/val split (by plant):")
    print(f"  Train: {len(train)} tiles across {train_plants} plants")
    print(f"  Val:   {len(val)} tiles across {val_plants} plants")
    print()

    # 5. Copy files into dataset/
    print(f"Copying files to {DATASET_DIR} ...")
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    copy_split(train, "train", DATASET_DIR)
    copy_split(val,   "val",   DATASET_DIR)
    print("  Done.\n")

    # 6. Write dataset.yaml
    write_yaml(YAML_PATH, DATASET_DIR, classes)
    print(f"Written: {YAML_PATH}\n")

    # 7. Summary
    print("=== Summary ===")
    print(f"  dataset/images/train/ : {len(train)} images")
    print(f"  dataset/images/val/   : {len(val)} images")
    print(f"  dataset/labels/train/ : {len(train)} label files")
    print(f"  dataset/labels/val/   : {len(val)} label files")
    print(f"  dataset.yaml          : {YAML_PATH}")
    print("\nReady for train_model.py")


if __name__ == "__main__":
    main()