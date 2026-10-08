"""
compare_models_fresh.py
=======================
Re-scores a newly trained detector against the deployed one on val plants
the deployed model CANNOT have trained on.

WHY (2026-10-08)
    04_train_model.py compares new vs deployed on 03's stable (hash) val
    split. But the deployed model was trained under the OLD random split, so
    ~80% of the val plants that already had labels then were in its TRAINING
    set. On 2026-10-08, 35 of 43 val plants had labels when the deployed
    model was committed: its score is largely training-set fit, and 04's
    "NOT deploying" verdict is biased toward it.

    This splits 03's val set (detection/dataset/, as 04 just used it) in two:
      fresh  plants with no label file in git at the commit that last changed
             correction/models/object_detection/best.pt -- neither model has
             seen them
      seen   plants that did have labels then -- the deployed model probably
             trained on most of them; shown for reference only
    and validates both models on each. Decide on FRESH.

    Fresh is small (8 plants / 30 tiles on 2026-10-08), so read it as a sanity
    check on direction, not a precise score.

Usage (detection/.venv, from detection/pipeline/, after 04):
    python compare_models_fresh.py                       # newest run's best.pt
    python compare_models_fresh.py --new ..\\models\\runs\\wwtp_v2-8\\weights\\best.pt
"""
import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

_spec = importlib.util.spec_from_file_location(
    "prepare03", Path(__file__).resolve().parent / "03_prepare_dataset.py")
_prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prep)

GIT_ROOT = C.REPO_ROOT.parent
LABELS_GIT = "detection/annotation/ls_export/labels/"
DEPLOY_GIT = "correction/models/object_detection/best.pt"
OUT_DIR = C.DATASET_DIR.parent / "dataset_compare"
MODEL_IMGSZ = -(-C.IMAGE_PX // 32) * 32       # as 04_train_model.py


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=GIT_ROOT, capture_output=True,
                          text=True, check=True).stdout


def plants_labelled_at(rev: str) -> set[str]:
    out = set()
    for path in git("ls-tree", "-r", "--name-only", rev, LABELS_GIT).split():
        r = _prep.parse_tile_stem(path.rsplit("/", 1)[-1])
        if r:
            out.add(r[0])
    return out


def newest_run_best() -> Path:
    runs = [p for p in C.RUNS_DIR.glob("*/weights/best.pt")]
    if not runs:
        raise SystemExit(f"No */weights/best.pt under {C.RUNS_DIR}; pass --new")
    return max(runs, key=lambda p: p.stat().st_mtime)


def build_subset(name: str, imgs: list[Path], classes: list[str]) -> Path:
    d = OUT_DIR / name
    if d.exists():
        shutil.rmtree(d)
    (d / "images" / "val").mkdir(parents=True)
    (d / "labels" / "val").mkdir(parents=True)
    for img in imgs:
        shutil.copy2(img, d / "images" / "val" / img.name)
        lbl = C.DATASET_DIR / "labels" / "val" / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, d / "labels" / "val" / lbl.name)
    yaml = d / "dataset.yaml"
    yaml.write_text(f"path: {d.as_posix()}\ntrain: images/val\nval: images/val\n\n"
                    f"nc: {len(classes)}\nnames: {classes}\n")
    return yaml


def score(weights: Path, yaml: Path, tag: str) -> dict:
    from ultralytics import YOLO
    m = YOLO(str(weights)).val(data=str(yaml), imgsz=MODEL_IMGSZ, plots=False,
                               project=str(OUT_DIR / "runs"), name=tag, exist_ok=True,
                               verbose=False)
    return {"map50": float(m.box.map50), "map": float(m.box.map),
            "p": float(m.box.mp), "r": float(m.box.mr),
            "ap50": {m.names[i]: float(a) for i, a in zip(m.box.ap_class_index, m.box.ap50)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", type=Path, default=None, help="new weights (default: newest run)")
    ap.add_argument("--deployed", type=Path, default=C.DEPLOY_MODEL_PATH)
    args = ap.parse_args()
    new_w = args.new or newest_run_best()

    rev = git("log", "-1", "--format=%h", "--", DEPLOY_GIT).strip()
    if not rev:
        raise SystemExit(f"No git history for {DEPLOY_GIT}")
    before = plants_labelled_at(rev)
    print(f"Deployed model last committed in {rev}; {len(before)} plants had labels then")

    val_imgs = sorted((C.DATASET_DIR / "images" / "val").glob("*.png"))
    if not val_imgs:
        raise SystemExit(f"No val images in {C.DATASET_DIR}; run 03_prepare_dataset.py")
    fresh, seen = [], []
    for img in val_imgs:
        r = _prep.parse_tile_stem(img.name.replace(".png", ".txt"))
        (seen if r and r[0] in before else fresh).append(img)
    def n_plants(imgs):
        return len({_prep.parse_tile_stem(i.name.replace(".png", ".txt"))[0] for i in imgs})
    print(f"Val: {len(val_imgs)} tiles -> fresh {len(fresh)} tiles / {n_plants(fresh)} plants, "
          f"seen {len(seen)} tiles / {n_plants(seen)} plants")

    classes = _prep.load_classes(C.CLASSES_FILE)
    res = {}
    for name, imgs in (("fresh", fresh), ("seen", seen)):
        if not imgs:
            continue
        y = build_subset(name, imgs, classes)
        res[name] = {"new": score(new_w, y, f"{name}_new"),
                     "deployed": score(args.deployed, y, f"{name}_deployed")}

    print(f"\nnew      = {new_w}\ndeployed = {args.deployed}")
    for name, r in res.items():
        note = ("neither model trained on these -- DECIDE ON THIS" if name == "fresh"
                else "deployed model likely trained on most of these -- reference only")
        print(f"\n=== {name.upper()} val plants ({note}) ===")
        print(f"  {'metric':<18}{'new':>9}{'deployed':>10}")
        for k, label in (("map50", "mAP50"), ("map", "mAP50-95"), ("p", "Precision"), ("r", "Recall")):
            print(f"  {label:<18}{r['new'][k]:>9.3f}{r['deployed'][k]:>10.3f}")
        print("  per-class AP50:")
        for cls in sorted(set(r["new"]["ap50"]) | set(r["deployed"]["ap50"])):
            n, o = r["new"]["ap50"].get(cls), r["deployed"]["ap50"].get(cls)
            print(f"  {cls:<18}{(f'{n:.3f}' if n is not None else '--'):>9}"
                  f"{(f'{o:.3f}' if o is not None else '--'):>10}")


if __name__ == "__main__":
    main()
