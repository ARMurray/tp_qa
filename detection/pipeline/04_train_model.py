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
import time
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
FORCE_DEPLOY = False        # deploy even if the new model scores below the deployed one
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
# Classes to train on. EMPTY = all of them, straight from C.DATASET_YAML with
# no filtered copy built -- which is also why this is [] rather than the six
# names spelled out: naming them would trigger build_filtered_dataset() and
# duplicate the whole ~1.2 GB dataset for no benefit.
#
# WAS ["aeration_basin", "clarifier", "digester"] until 2026-09-23. Annotation
# counts at the time of that change, across 1,004 label files (132 with boxes,
# 872 deliberate empties):
#
#     clarifier          264 boxes / 73 tiles     was trained
#     oxidation_pond     138 boxes / 67 tiles     was NOT
#     aeration_basin     105 boxes / 59 tiles     was trained
#     digester            82 boxes / 29 tiles     was trained
#     chlorine_contact    29 boxes / 22 tiles     was NOT
#     drying_bed          10 boxes /  6 tiles     was NOT
#
# oxidation_pond is the one that mattered: more instances than digester and
# more tiles than aeration_basin, excluded the whole time, and the dominant
# infrastructure at small plants -- exactly the plants the correction pipeline
# is worst at. chlorine_contact at 22 tiles is thin but sits in the same range
# as digester's 29.
#
# WATCH drying_bed. Six tiles is below where a YOLO class can learn anything,
# so expect its detections to be unreliable at first -- which matters because
# correction's models consume od_has_drying_bed / od_n_drying_bed as features.
# It is included so that new labels count immediately rather than needing
# another config change, and because targeted tile selection now accumulates
# examples where they actually occur. If 07/07b show it carrying weight before
# the count is up around 25+ tiles, that weight is noise: re-check then.
#
# correction/scripts/config.py's CLASSES already lists all six and always has
# -- it fixes the od_* FEATURE SCHEMA independently of what the detector was
# trained on. So those columns have existed all along, with three of them
# permanently False. Nothing changes there.
KEEP_CLASSES = []
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

    # ---- Validate the new model AND the deployed one on the same val set ---
    # 03's val split is stable (hash of the site id, 2026-10-05), so both are
    # scored on the same plants. Caveat printed below: the deployed model was
    # trained under the OLD random split, so it has seen some of these val
    # plants -- its score is optimistic, and the comparison favours it. A new
    # model that still wins is genuinely better.
    def _val(weights: Path, tag: str):
        m = YOLO(str(weights)).val(
            data=str(data_yaml), imgsz=MODEL_IMGSZ, device=device, plots=(tag == "new"),
            project=str(C.RUNS_DIR), name=f"{run_dir.name}_val_{tag}", exist_ok=True,
        )
        return {"map50": float(m.box.map50), "map": float(m.box.map),
                "p": float(m.box.mp), "r": float(m.box.mr),
                "ap50": {m.names[i]: float(ap) for i, ap in zip(m.box.ap_class_index, m.box.ap50)}}

    print("\nValidating the NEW model...")
    new = _val(best, "new")
    old = None
    if C.DEPLOY_MODEL_PATH.exists():
        print("\nValidating the DEPLOYED model on the same val set...")
        try:
            old = _val(C.DEPLOY_MODEL_PATH, "deployed")
        except Exception as e:  # e.g. a class list the new dataset no longer has
            print(f"  could not validate the deployed model: {e}")

    print("\n=== Validation: new vs deployed (same stable val split) ===")
    print(f"  {'metric':<18}{'new':>9}{'deployed':>10}")
    for k, label in (("map50", "mAP50"), ("map", "mAP50-95"), ("p", "Precision"), ("r", "Recall")):
        o = f"{old[k]:.3f}" if old else "--"
        print(f"  {label:<18}{new[k]:>9.3f}{o:>10}")
    print("  per-class AP50:")
    for cls in sorted(set(new["ap50"]) | set(old["ap50"] if old else {})):
        n = new["ap50"].get(cls)
        o = old["ap50"].get(cls) if old else None
        print(f"  {cls:<18}{(f'{n:.3f}' if n is not None else '--'):>9}"
              f"{(f'{o:.3f}' if o is not None else '--'):>10}")
    if old:
        print("  (the deployed model trained under the old random split and has seen")
        print("   some of these val plants -- its numbers are optimistic)")

    deploy = old is None or new["map50"] >= old["map50"] or FORCE_DEPLOY
    if not deploy:
        print(f"\nNOT deploying: new mAP50 {new['map50']:.3f} < deployed {old['map50']:.3f}.")
        print(f"  The new weights stay at {best}. Set FORCE_DEPLOY = True in this")
        print(f"  script (or copy by hand) to deploy anyway.")
        print("\nNext: 05_run_inference.py")
        return

    # ---- Deploy to the correction pipeline -------------------------------
    # Copy, don't ask -- once the new model has matched or beaten the deployed
    # one above. A manual copy is the step that gets forgotten, and
    # forgetting it means 01b/01c/01e keep running the previous model while
    # everything downstream looks perfectly normal. The model being replaced
    # is backed up under RUNS_DIR (not next to best.pt: every *.pt there
    # counts as "the deployed detector" for the freshness checks).
    if C.DEPLOY_MODEL_PATH.exists():
        backup = C.RUNS_DIR / "deployed_backups" / \
            f"best_replaced_{time.strftime('%Y%m%d-%H%M%S')}.pt"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(C.DEPLOY_MODEL_PATH, backup)
        print(f"\nBacked up the replaced model: {backup}")
    #
    # shutil.copy, NOT copy2 -- and this matters more than it looks.
    # check_od_freshness.py decides whether existing detection output is
    # stale by comparing each partition's mtime against the deployed model's,
    # and its docstring warns that anything preserving the SOURCE mtime
    # (scp -p, rsync -t) makes a freshly deployed model look old. copy2
    # preserves mtime and would do exactly that, so the freshness check would
    # pass on detection output produced by the previous weights. copy stamps
    # the copy with now, which is the deploy time -- what that check assumes.
    try:
        C.DEPLOY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, C.DEPLOY_MODEL_PATH)
        print(f"\nDeployed to correction pipeline: {C.DEPLOY_MODEL_PATH}")
        print(f"  size: {C.DEPLOY_MODEL_PATH.stat().st_size / 1e6:.1f} MB")
        print("  Every existing detection output is now STALE. Before the next")
        print("  02_feature_engineering run, re-run 01b / 01c / 01e and then")
        print("  check_od_freshness.py, or the feature tables will mix outputs")
        print("  from two different models.")
        print(f"  This file is tracked in git -- commit and push it so the HPC")
        print(f"  picks it up.")
    except OSError as e:
        # Not fatal: the model trained fine and is safe under RUNS_DIR. Only
        # the deploy copy failed, and that is recoverable by hand.
        print(f"\nWARNING: could not deploy to {C.DEPLOY_MODEL_PATH}: {e}")
        print(f"  Copy it manually from {best} before running 01b/01c/01e.")

    print("\nNext: 05_run_inference.py")


if __name__ == "__main__":
    main()