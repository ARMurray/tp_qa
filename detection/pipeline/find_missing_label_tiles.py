"""
find_missing_label_tiles.py
===========================
Lists every label file whose RGB tile is not in tiles/rgb/png/ (the
"label files had no matching RGB tile" count in 03_prepare_dataset.py) and
says where each one went, so it can be put back before training. 03 drops
these from the dataset silently apart from that one count.

For each missing tile it reports:
    boxes        number of boxes in the label (0 = confirmed empty)
    classes      which classes those boxes are
    found_at     any file with the same name elsewhere under detection/data/
                 (e.g. tiles/_pruned/<stamp>/rgb/ from extract_review_tiles
                 --prune)
    in_metadata  which tile_metadata*.csv (current or a backup) still lists it,
                 with its source / role

Writes detection/annotation/missing_label_tiles.csv.
--restore copies every tile found elsewhere back into tiles/rgb/png/ (and
its NDWI, if found, into tiles/ndwi/). Copies, never moves.

Usage (detection/.venv, from detection/pipeline/):
    python find_missing_label_tiles.py
    python find_missing_label_tiles.py --restore
"""
import argparse
import importlib.util
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

_spec = importlib.util.spec_from_file_location(
    "prepare03", Path(__file__).resolve().parent / "03_prepare_dataset.py")
_prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prep)

OUT_CSV = C.ANNOTATION_DIR.parent / "missing_label_tiles.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--restore", action="store_true",
                    help="copy tiles found elsewhere back into tiles/rgb/png/")
    args = ap.parse_args()

    classes = _prep.load_classes(C.CLASSES_FILE)
    labels = _prep.build_label_index(C.ANNOTATION_DIR / "labels")
    rgb = _prep.build_rgb_index(C.RGB_DIR)
    missing = {stem: p for stem, p in labels.items() if stem not in rgb}
    print(f"{len(labels):,} parsed label file(s), {len(rgb):,} RGB tile(s); "
          f"{len(missing)} label(s) with no tile in {C.RGB_DIR}")
    if not missing:
        return

    # Every PNG / TIF anywhere under data/ except the live folders, by name.
    elsewhere = {}
    for p in C.DATA_DIR.rglob("*"):
        if p.suffix.lower() in (".png", ".tif") and p.parent not in (C.RGB_DIR, C.NDWI_DIR):
            elsewhere.setdefault(p.name, []).append(p)

    metas = []
    for m in sorted(C.DATA_DIR.glob("tile_metadata*.csv")):
        try:
            d = pd.read_csv(m, dtype=str, usecols=lambda c: c in
                            ("tile_id", "source", "ll_uuid_alternates"))
            d["file"] = m.name
            metas.append(d)
        except Exception as e:
            print(f"  could not read {m.name}: {e}")
    meta = pd.concat(metas) if metas else pd.DataFrame(columns=["tile_id"])

    rows = []
    for stem, lp in sorted(missing.items()):
        lines = [ln.split() for ln in lp.read_text().splitlines() if ln.strip()]
        cls = Counter(classes[int(r[0])] if int(r[0]) < len(classes) else r[0] for r in lines)
        found = elsewhere.get(f"{stem}_rgb.png", [])
        m = meta[meta["tile_id"] == stem]
        rows.append(dict(
            tile_id=stem, label_file=lp.name, boxes=len(lines),
            classes=", ".join(f"{k} {v}" for k, v in sorted(cls.items())),
            found_at=";".join(str(p) for p in found),
            in_metadata=";".join(f"{r.file} [{r.get('source', '')}/{r.get('ll_uuid_alternates', '')}]"
                                 for _, r in m.iterrows()),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    n_box = int((df["boxes"] > 0).sum())
    n_found = int((df["found_at"] != "").sum())
    n_meta = int((df["in_metadata"] != "").sum())
    print(f"  {n_box} have boxes, {len(df) - n_box} are confirmed-empty labels")
    print(f"  {n_found} found elsewhere under {C.DATA_DIR} (restorable)")
    print(f"  {n_meta} still listed in a tile_metadata*.csv (current or backup)")
    print(f"  {len(df) - n_found} not on disk anywhere under data/")
    print(f"\nWritten: {OUT_CSV}")
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(df[["tile_id", "boxes", "classes", "in_metadata"]].head(40).to_string(index=False))

    if args.restore and n_found:
        n = 0
        for r in df[df["found_at"] != ""].itertuples():
            src = Path(r.found_at.split(";")[0])
            shutil.copy2(src, C.RGB_DIR / src.name)
            n += 1
            tif = elsewhere.get(f"{r.tile_id}_ndwi.tif", [])
            if tif:
                C.NDWI_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(tif[0], C.NDWI_DIR / tif[0].name)
        print(f"\nRestored {n} tile(s) into {C.RGB_DIR}. Re-run 03_prepare_dataset.py.")
    elif n_found:
        print("\nRe-run with --restore to copy the found tiles back.")


if __name__ == "__main__":
    main()
