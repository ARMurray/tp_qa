"""
04_train_model.py
=================
Trains a YOLOv8s object detection model on annotated NAIP tiles for wastewater
treatment plant infrastructure detection.

Requires: ultralytics, torch (CUDA-enabled)

Outputs (under models/runs/{RUN_NAME}/):
    weights/best.pt   - best checkpoint (used by 05_run_inference.py)
    weights/last.pt   - final epoch checkpoint
    results.csv       - per-epoch metrics
    *.png / val_batch*.jpg - training plots and sample predictions
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

from ultralytics import YOLO
import torch

import torch
_orig_save = torch.save
def _save_legacy_format(*args, **kwargs):
    kwargs.setdefault("_use_new_zipfile_serialization", False)
    return _orig_save(*args, **kwargs)
torch.save = _save_legacy_format

# ---------------------------------------------------------------------------
# Shared config comes from config.py:
#   C.DATASET_YAML, C.DATASET_DIR, C.CLASSES_FILE, C.RUNS_DIR, C.RUN_NAME,
#   C.IMAGE_PX, C.RANDOM_SEED
# ---------------------------------------------------------------------------

# Model size — yolov8s balances 8GB VRAM and limited training data.
# Increasing size/power: yolov8n < yolov8s < yolov8m < yolov8l < yolov8x
MODEL      = "yolov8s.pt"   # downloads pretrained COCO weights automatically

# Training hyperparameters (training-specific; kept local)
EPOCHS     = 100            # temporarily set to 2 for the RUNS_DIR permission test -- reverted
PATIENCE   = 20             # stop if val mAP stalls for this many epochs
BATCH_SIZE = 16             # fits comfortably in 8GB VRAM with yolov8s
LR         = 0.001          # initial LR (Adam; fine for transfer learning)
WORKERS    = 4              # dataloader workers

# YOLO's backbone requires imgsz to be a multiple of its max stride (32).
# C.IMAGE_PX (333px) comes from tile geometry (200m / 0.6m/px), not from any
# YOLO constraint, so it isn't a multiple of 32 -- Ultralytics would silently
# round it up to 352 on every call and log a warning each time. Resolving it
# once here avoids that spam; detected box coordinates still come back in
# the original C.IMAGE_PX pixel space (Ultralytics rescales internally), so
# nothing downstream (pixel_to_lonlat, etc.) needs to change.
MODEL_IMGSZ = -(-C.IMAGE_PX // 32) * 32   # ceiling to nearest multiple of 32 -> 352

# Augmentation — important with limited data; applied on-the-fly
AUGMENT_CONFIG = dict(
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
    degrees=90,              # infrastructure has no canonical orientation
    fliplr=0.5, flipud=0.5,  # aerial — both flips valid
    mosaic=0.5, mixup=0.1,
    scale=0.3, translate=0.1,
    shear=0.0, perspective=0.0,   # orthographic imagery
)

# --- Class filtering -------------------------------------------------------
# Set to a list of class NAMES to train on a subset without re-annotating
# anything. Excluded-class boxes are dropped from a filtered COPY of
# dataset/; a tile whose only box was an excluded class becomes a genuine
# negative for the classes you kept (your annotators already confirmed
# nothing else was in that tile). Original dataset/, ls_export/, and all
# annotations are left untouched -- this only affects what THIS training
# run sees. Set to None to train on every class in classes.txt as before.
#
# oxidation_pond and (implicitly) any lagoon-scale class are excluded here
# because they need a larger tile size than this model's 200m tiles support --
# that's a separate future model, not something this filter can fix.
KEEP_CLASSES = ["aeration_basin", "clarifier", "digester"]
FILTERED_DATASET_DIR = C.DATASET_DIR.parent / "dataset_filtered"
# ---------------------------------------------------------------------------


def load_all_classes(classes_file: Path) -> list[str]:
    with open(classes_file) as f:
        return [line.strip() for line in f if line.strip()]


def build_filtered_dataset(source_dataset_dir: Path, all_classes: list[str],
                            keep_classes: list[str], filtered_dir: Path) -> Path:
    """
    Builds a class-filtered COPY of source_dataset_dir under filtered_dir.
    Label lines for excluded classes are dropped; kept classes are remapped
    to a clean 0..N-1 range in ALPHABETICAL order (matching the existing
    ls_export/classes.txt convention, so this doesn't introduce a second,
    inconsistent class-ordering scheme). A tile that had annotations ONLY
    for excluded classes becomes a legitimate empty/negative label file,
    not a dropped tile -- the image is still copied and still trains the
    model on "nothing here" for the kept classes.
    """
    keep_sorted = sorted(keep_classes)
    old_id_to_name = dict(enumerate(all_classes))
    name_to_new_id = {name: i for i, name in enumerate(keep_sorted)}

    unknown = set(keep_classes) - set(all_classes)
    if unknown:
        raise ValueError(f"KEEP_CLASSES has names not in classes.txt: {unknown}")

    if filtered_dir.exists():
        shutil.rmtree(filtered_dir)

    n_boxes_kept = n_boxes_dropped = n_tiles_became_negative = n_images = 0

    for split in ("train", "val"):
        img_src, lbl_src = source_dataset_dir / "images" / split, source_dataset_dir / "labels" / split
        img_dst, lbl_dst = filtered_dir / "images" / split, filtered_dir / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for label_path in lbl_src.glob("*.txt"):
            had_any_box = label_path.stat().st_size > 0
            kept_lines = []
            if had_any_box:
                for line in label_path.read_text().splitlines():
                    if not line.strip():
                        continue
                    parts = line.split()
                    name = old_id_to_name.get(int(parts[0]))
                    if name in name_to_new_id:
                        parts[0] = str(name_to_new_id[name])
                        kept_lines.append(" ".join(parts))
                        n_boxes_kept += 1
                    else:
                        n_boxes_dropped += 1

            (lbl_dst / label_path.name).write_text(
                "\n".join(kept_lines) + ("\n" if kept_lines else "")
            )
            if had_any_box and not kept_lines:
                n_tiles_became_negative += 1

            img_path = img_src / (label_path.stem + ".png")
            if img_path.exists():
                shutil.copy2(img_path, img_dst / img_path.name)
                n_images += 1

    yaml_path = filtered_dir / "dataset.yaml"
    lines = [
        f"path: {filtered_dir.as_posix()}",
        "train: images/train",
        "val:   images/val",
        "",
        f"nc: {len(keep_sorted)}",
        f"names: {keep_sorted}",
        "",
    ]
    yaml_path.write_text("\n".join(lines))

    print(f"  Filtered classes ({len(keep_sorted)}): {keep_sorted}")
    print(f"  Images copied         : {n_images}")
    print(f"  Boxes kept            : {n_boxes_kept}")
    print(f"  Boxes dropped (excluded classes, e.g. oxidation_pond): {n_boxes_dropped}")
    print(f"  Tiles now negative     : {n_tiles_became_negative} "
          f"(previously annotated only with an excluded class)")
    print(f"  Wrote: {yaml_path}")
    return yaml_path


def main():
    print("=== 04_train_model.py ===\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: no GPU found — training on CPU will be slow.")
    print()

    if not C.DATASET_YAML.exists():
        print(f"ERROR: {C.DATASET_YAML} not found. Run 03_prepare_dataset.py first.")
        return

    if KEEP_CLASSES:
        print(f"Class filter active -- building filtered dataset copy...")
        all_classes = load_all_classes(C.CLASSES_FILE)
        data_yaml = build_filtered_dataset(C.DATASET_DIR, all_classes, KEEP_CLASSES, FILTERED_DATASET_DIR)
        print()
    else:
        data_yaml = C.DATASET_YAML

    print(f"Loading model: {MODEL}")
    model = YOLO(MODEL)

    print(f"Training run '{C.RUN_NAME}' -> {C.RUNS_DIR / C.RUN_NAME}")
    print("NOTE: if a directory named C.RUN_NAME already exists, Ultralytics will")
    print("      auto-increment to e.g. RUN_NAME-2 rather than failing -- this script")
    print("      always resolves paths from the ACTUAL save_dir below, never from")
    print("      C.RUN_NAME directly, so an auto-increment can't point us at a stale run.\n")
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS, patience=PATIENCE,
        imgsz=MODEL_IMGSZ, batch=BATCH_SIZE, lr0=LR,
        device=device, workers=WORKERS,
        project=str(C.RUNS_DIR), name=C.RUN_NAME,
        exist_ok=False, pretrained=True, optimizer="Adam",
        verbose=True, seed=C.RANDOM_SEED, plots=True,
        **AUGMENT_CONFIG,
    )

    # Always resolve THIS run's actual directory from the trainer, never by
    # reconstructing C.RUNS_DIR / C.RUN_NAME -- if Ultralytics auto-incremented
    # the name (e.g. RUN_NAME already existed from a prior run), reconstructing
    # the path from C.RUN_NAME silently points at that OLD run's weights/classes
    # instead of the one we just trained.
    run_dir = model.trainer.save_dir
    print("\n=== Training Complete ===")
    print(f"Run directory: {run_dir}")
    if run_dir.name != C.RUN_NAME:
        print(f"NOTE: C.RUN_NAME='{C.RUN_NAME}' was already taken -- Ultralytics used "
              f"'{run_dir.name}' instead. Update C.RUN_NAME in config.py to avoid this "
              f"next time, or clear out old run folders under {C.RUNS_DIR}.")

    best = run_dir / "weights" / "best.pt"
    if best.exists():
        print(f"Best weights: {best}")
    else:
        print(f"ERROR: expected best.pt at {best} but it doesn't exist.")
        return

    print("\nValidating with best weights...")
    metrics = YOLO(str(best)).val(
        data=str(data_yaml), imgsz=MODEL_IMGSZ, device=device, plots=True,
        project=str(C.RUNS_DIR), name=run_dir.name + "_val", exist_ok=True,
    )
    print("\n=== Validation Metrics ===")
    print(f"  mAP50    : {metrics.box.map50:.3f}")
    print(f"  mAP50-95 : {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall   : {metrics.box.mr:.3f}")
    print("Per-class AP50:")
    for i, ap in enumerate(metrics.box.ap50):
        print(f"  {metrics.names[i]:<18}: {ap:.3f}")

    print("\nNext: 05_run_inference.py")


if __name__ == "__main__":
    main()