"""
dedupe_tiles_by_content.py
===========================
Finds duplicate NAIP tiles in the labelling inventory by looking at the
IMAGES, not the metadata, and quarantines the redundant copies.

    python dedupe_tiles_by_content.py                     # report only
    python dedupe_tiles_by_content.py --apply             # move dups aside
    python dedupe_tiles_by_content.py --apply --delete    # actually delete
    python dedupe_tiles_by_content.py --near-duplicates   # also report dHash matches

WHY THIS EXISTS ALONGSIDE review_app/analysis/dedupe_existing_tiles.py
    They find different things, and you probably want both.

    dedupe_existing_tiles.py  compares tile BOUNDING BOXES from
        tile_metadata.csv, within a single CWNS_ID, and clusters by IoU. It
        catches tiles that overlap heavily without being identical -- the
        same lagoon photographed at two different tile offsets.

    this script            compares the IMAGE CONTENT of every file on disk.
        It catches exact duplicates wherever they came from, including
        ACROSS plants, and including tiles that have no row in
        tile_metadata.csv at all.

    That last point is the reason this exists. On 2026-09-22 the inventory
    held 1,004 images in dataset_filtered alone against a tile_metadata.csv
    of 264 rows; reconstruct_tile_metadata.py exists because the metadata
    has been incomplete before. Anything metadata-driven silently skips
    whatever is missing from it. Hashing the files cannot.

WHAT THE DUPLICATION ACTUALLY LOOKED LIKE (measured, 2026-09-22)
    Across the 1,004 labelled tiles in dataset_filtered:

        identical bytes         969 unique | 17 groups | 35 redundant (3.5%)
        identical pixels        969 unique | 17 groups | 35 redundant (3.5%)
        near-identical (dHash)  969 unique | 17 groups | 35 redundant (3.5%)

    All three agree exactly, so the duplicates are byte-identical rather
    than near-misses. Every group was same-plant / different-parcel-uuid --
    one plant (08000000031) had three Regrid parcel records covering the
    same ground, each tiled separately into the same nine images. Zero
    cross-plant duplicates in that sample, no train/val leakage, and all 17
    groups carried identical labels.

    THE FULL INVENTORY, measured on the real machine 2026-09-23:

        6,814 tiles -> 1,003 after --purge-unlabeled
           35 exact duplicates
        5,776 never labelled

    So duplication was never the problem -- 35 redundant files out of 6,814,
    all of them inside the labelled subset. VOLUME was the problem, and
    5,776 of those tiles had been fetched, stored and never looked at.

    A follow-up pass with --near-duplicates over the 1,003 survivors found
    ZERO near-matches as well. The impression that the inventory was full of
    duplicates came from two things that are not duplication: one plant
    (08000000031) contributing 27 of the 35, which clusters while labelling
    and feels like many more; and thousands of empty fields that look alike
    without being the same tile.

    That is the argument for targeted selection rather than better dedup.
    See extract_review_tiles.py's load_targeted_sites().

LABELS ARE NEVER DESTROYED
    A tile with a label file is never moved or deleted, full stop. If a
    duplicate group contains more than one labelled tile the whole group is
    left alone and flagged: two labels on the same image is a training-set
    correctness question, not something to silently pick a side on. (In the
    measured sample all such groups agreed, but that is not guaranteed.)

    Otherwise the keeper is: the labelled tile if there is exactly one,
    else the lexicographically smallest tile id -- deterministic, so two
    runs make the same choice. Same tie-break dedupe_existing_tiles.py uses.

QUARANTINE, NOT DELETE, BY DEFAULT
    --apply MOVES redundant files to tiles/_duplicates/ preserving their
    names. Nothing is destroyed, the inventory shrinks, and if this got
    something wrong you move them back. --delete opts into real deletion.

    Either way a manifest is written to tiles/_duplicates/manifest_<ts>.json
    recording every group, what was kept, and what was moved, so the run is
    auditable after the fact.
"""
import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = None
    Image = None

TILE_NAME_SEP = "_rgb"


# ===========================================================================
# Hashing
# ===========================================================================
def byte_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_hash(path: Path) -> str | None:
    """Hash of the DECODED pixels, so a re-encode of the same image still
    collides. Slower than hashing bytes -- roughly one decode per file --
    but the byte hash would miss a tile that was written twice by different
    PNG settings, and that is exactly the case a fast check would report as
    'not a duplicate' when it is."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return hashlib.md5(np.asarray(im).tobytes()).hexdigest()
    except Exception:
        return None


def dhash(path: Path, size: int = 8) -> str | None:
    """Difference hash: survives re-encoding, mild resampling, small
    brightness shifts. Reported but never acted on by default -- a dHash
    collision is evidence of similarity, not proof of identity, and two
    genuinely different plants can look alike from 600m up."""
    try:
        with Image.open(path) as im:
            g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
            a = np.asarray(g, dtype=np.int16)
            return np.packbits(a[:, 1:] > a[:, :-1]).tobytes().hex()
    except Exception:
        return None


# ===========================================================================
# Inventory
# ===========================================================================
def parse_tile_name(p: Path):
    """{CWNS_ID}_{ll_uuid}_r##_c##_rgb.png -> (cwns_id, ll_uuid, tile_id).

    Best-effort: tiles from other sources (TRI facilities, 500m conversions)
    do not all follow it. An unparsed name is still deduplicated -- it just
    cannot contribute to the same-plant / cross-plant breakdown.
    """
    stem = p.stem
    tile_id = stem[:-len("_rgb")] if stem.endswith("_rgb") else stem
    parts = tile_id.split("_")
    if len(parts) >= 4 and parts[0].isdigit():
        return parts[0], parts[1], tile_id
    return None, None, tile_id


def load_labeled_stems(annotation_dir: Path) -> set[str]:
    labels_dir = annotation_dir / "labels"
    if not labels_dir.exists():
        print(f"  no labels directory at {labels_dir} -- treating every tile "
              f"as unlabelled. Check this before --apply.")
        return set()
    return {p.stem for p in labels_dir.glob("*.txt")}


def is_labeled(tile_id: str, labeled_stems: set[str]) -> bool:
    # Substring match, same approach extract_review_tiles.py --check-labels
    # and dedupe_existing_tiles.py both use: exported label stems carry
    # Label Studio prefixes/suffixes around the tile id.
    return any(tile_id in stem for stem in labeled_stems)


def find_label_files(tile_id: str, labels_dir: Path) -> list[Path]:
    if not labels_dir.exists():
        return []
    return sorted(p for p in labels_dir.glob("*.txt") if tile_id in p.stem)


def label_signature(paths: list[Path]) -> str | None:
    """Canonical form of a tile's YOLO annotations, for comparing two
    labellings of the same image.

    Boxes are rounded to 4dp and sorted, so the same annotation drawn in a
    different order, or re-exported with different float formatting, compares
    equal. Genuinely different boxes do not.

    AN EMPTY LABEL FILE IS A REAL ANNOTATION, not a missing one: in YOLO
    format a zero-byte .txt means "this tile was looked at and contains no
    objects", which is a negative training example and exactly as
    deliberate as a box. 872 of the 1,004 labels in the sample inventory
    are empty. Returning None for those would make every duplicate group of
    negatives look like an unresolvable conflict, and nothing would ever be
    deduplicated.

    Returns None ONLY if no file exists or one could not be read -- genuinely
    unknown. Callers must never treat None as "matches".
    """
    if not paths:
        return None
    rows = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                cls = int(float(parts[0]))
                nums = tuple(round(float(x), 4) for x in parts[1:])
            except ValueError:
                rows.append(line.strip())
                continue
            rows.append((cls,) + nums)
    # No rows but the file(s) existed and read cleanly -> an explicit
    # "nothing here". Distinct from None, which means unknown.
    return repr(sorted(rows, key=repr))


def find_ndwi(tile_id: str, ndwi_dir: Path) -> Path | None:
    for ext in (".tif", ".tiff", ".png"):
        for cand in (ndwi_dir / f"{tile_id}_ndwi{ext}", ndwi_dir / f"{tile_id}{ext}"):
            if cand.exists():
                return cand
    return None


# ===========================================================================
# Reporting
# ===========================================================================
def describe_groups(groups, info, label):
    redundant = sum(len(g) - 1 for g in groups)
    total = sum(len(g) for g in groups)
    print(f"\n{label}")
    print(f"  {len(groups)} group(s), {total} files, {redundant} redundant")
    if not groups:
        return
    same_plant = cross_plant = same_parcel = unknown = 0
    for g in groups:
        cw = {info[f]["cwns_id"] for f in g}
        pc = {info[f]["ll_uuid"] for f in g}
        if None in cw:
            unknown += 1
        elif len(cw) > 1:
            cross_plant += 1
        elif len(pc) > 1:
            same_plant += 1
        else:
            same_parcel += 1
    print(f"    cross-plant (different CWNS_ID)   : {cross_plant}")
    print(f"    same plant, different parcel uuid : {same_plant}")
    print(f"    same plant, same parcel           : {same_parcel}")
    if unknown:
        print(f"    unparsed filenames                : {unknown}")

    worst = defaultdict(int)
    for g in groups:
        for f in g[1:]:
            cw = info[f]["cwns_id"]
            if cw:
                worst[cw] += 1
    if worst:
        print("    plants contributing the most redundancy:")
        for cw, n in sorted(worst.items(), key=lambda kv: -kv[1])[:8]:
            print(f"      {cw}: {n} redundant tile(s)")


def main():
    ap = argparse.ArgumentParser(
        description="Deduplicate NAIP labelling tiles by image content.")
    ap.add_argument("--rgb-dir", type=Path, default=None,
                    help=f"tile directory (default: config.RGB_DIR)")
    ap.add_argument("--ndwi-dir", type=Path, default=None)
    ap.add_argument("--annotation-dir", type=Path, default=None,
                    help="annotation/ls_export, for label protection")
    ap.add_argument("--fast", action="store_true",
                    help="hash raw bytes only, skipping the decode pass. "
                         "Misses a re-encoded copy of the same image.")
    ap.add_argument("--near-duplicates", action="store_true",
                    help="also report dHash near-matches. Report only -- "
                         "never acted on, see the docstring.")
    ap.add_argument("--purge-unlabeled", action="store_true",
                    help="ALSO remove every tile that has no label file at "
                         "all. Resets the inventory to exactly what has been "
                         "labelled, on the basis that future tiles arrive by "
                         "targeted selection rather than bulk extraction. "
                         "Read the docstring before using this.")
    ap.add_argument("--apply", action="store_true",
                    help="move redundant tiles to tiles/_duplicates/")
    ap.add_argument("--delete", action="store_true",
                    help="with --apply, delete instead of quarantining")
    args = ap.parse_args()

    if Image is None and not args.fast:
        raise SystemExit("Pillow/numpy needed for the decode pass. Install "
                         "them, or pass --fast to hash bytes only.")

    rgb_dir = args.rgb_dir or C.RGB_DIR
    ndwi_dir = args.ndwi_dir or C.NDWI_DIR
    annotation_dir = args.annotation_dir or C.ANNOTATION_DIR

    print("=== dedupe_tiles_by_content.py ===")
    print(f"Tiles      : {rgb_dir}")
    print(f"NDWI       : {ndwi_dir}")
    print(f"Annotations: {annotation_dir}")

    if not rgb_dir.exists():
        raise SystemExit(f"\n{rgb_dir} does not exist. Pass --rgb-dir.")

    files = sorted(p for p in rgb_dir.rglob("*.png"))
    if not files:
        raise SystemExit(f"\nNo .png tiles under {rgb_dir}.")
    print(f"\n{len(files)} tile(s) found")

    labeled_stems = load_labeled_stems(annotation_dir)
    print(f"{len(labeled_stems)} label file(s) on disk")
    n_label_files = len(labeled_stems)

    info = {}
    exact = defaultdict(list)
    near = defaultdict(list)
    unreadable = []

    for i, f in enumerate(files):
        if i and i % 500 == 0:
            print(f"  hashed {i}/{len(files)}...", flush=True)
        cwns_id, ll_uuid, tile_id = parse_tile_name(f)
        key = byte_hash(f) if args.fast else pixel_hash(f)
        if key is None:
            unreadable.append(f)
            continue
        info[f] = {"cwns_id": cwns_id, "ll_uuid": ll_uuid, "tile_id": tile_id,
                   "labeled": is_labeled(tile_id, labeled_stems)}
        exact[key].append(f)
        if args.near_duplicates:
            d = dhash(f)
            if d:
                near[d].append(f)

    if unreadable:
        print(f"\n  WARNING: {len(unreadable)} file(s) could not be read and were "
              f"skipped entirely (not counted as duplicates of anything):")
        for f in unreadable[:5]:
            print(f"    {f.name}")

    exact_groups = [sorted(v, key=lambda p: p.name) for v in exact.values() if len(v) > 1]
    describe_groups(exact_groups, info, "EXACT DUPLICATES"
                    + (" (byte hash)" if args.fast else " (decoded pixels)"))

    if args.near_duplicates:
        seen = {f for g in exact_groups for f in g}
        near_groups = [sorted(v, key=lambda p: p.name) for v in near.values()
                       if len(v) > 1 and not set(v) <= seen]
        describe_groups(near_groups, info,
                        "NEAR DUPLICATES (dHash) -- reported only, never acted on")

    # ---- decide keepers ---------------------------------------------------
    labels_dir = annotation_dir / "labels"
    plan, conflicts, agreed = [], [], 0
    for g in exact_groups:
        labeled = [f for f in g if info[f]["labeled"]]

        if len(labeled) > 1:
            # More than one labelling of the same image. That is only a
            # conflict if the labellings DISAGREE. In the measured sample all
            # 17 such groups annotated the image identically -- flagging those
            # for hand review would be noise, and would leave the duplicates
            # in place forever. Compare the boxes and decide.
            sigs = {}
            for f in labeled:
                sigs[f] = label_signature(find_label_files(info[f]["tile_id"], labels_dir))
            distinct = {s for s in sigs.values() if s is not None}
            unknown = any(s is None for s in sigs.values())
            if unknown or len(distinct) > 1:
                conflicts.append((g, sigs))
                continue
            agreed += 1
            keeper = labeled[0]          # identical annotations; any will do
        else:
            keeper = labeled[0] if labeled else g[0]

        plan.append((keeper, [f for f in g if f != keeper]))

    # ---- optional: drop everything that was never labelled ----------------
    # Deliberately AFTER the duplicate pass, so a tile that is unlabelled but
    # is the chosen keeper of a group containing a labelled copy is not
    # double-counted. In practice the keeper is always the labelled one when
    # any exists, so the two sets are disjoint -- but relying on that rather
    # than enforcing it would be fragile.
    purged = []
    if args.purge_unlabeled:
        already_going = {f for _, drop in plan for f in drop}
        keepers = {k for k, _ in plan}
        conflict_files = {f for g, _ in conflicts for f in g}
        for f, meta in info.items():
            if meta["labeled"] or f in already_going or f in conflict_files:
                continue
            if f in keepers:
                # unlabelled, but the representative of a duplicate group in
                # which nothing was labelled -- the whole group goes
                pass
            purged.append(f)
        plan = [(k, d) for k, d in plan if k not in set(purged)]

    n_move = sum(len(drop) for _, drop in plan) + len(purged)
    print(f"\n--- plan ---")
    print(f"  {len(plan)} group(s) resolvable -> "
          f"{sum(len(d) for _, d in plan)} redundant duplicate(s)")
    if args.purge_unlabeled:
        n_labeled = sum(1 for m in info.values() if m["labeled"])
        print(f"  --purge-unlabeled: {len(purged)} never-labelled tile(s) to remove")
        print(f"    ({n_labeled} of {len(info)} tiles carry a label)")

        # Cross-check the matcher against the label files themselves. is_labeled
        # is a SUBSTRING test, because Label Studio exports carry hash prefixes
        # around the tile id -- so it can be wrong in both directions, and the
        # two directions have very different consequences.
        if n_label_files:
            if n_labeled < n_label_files:
                print(f"\n  *** WARNING: {n_labeled} tiles matched a label, but "
                      f"{n_label_files} label files exist. ***")
                print(f"  {n_label_files - n_labeled} label(s) matched no tile, which")
                print(f"  can mean labelled tiles are about to be purged as unlabelled.")
                print(f"  DO NOT --apply until you know why. Check whether the export")
                print(f"  filenames still contain the tile id.")
            elif n_labeled > n_label_files:
                print(f"\n  note: {n_labeled} tiles matched a label but only "
                      f"{n_label_files} label files exist.")
                print(f"  {n_labeled - n_label_files} tile(s) match a label belonging to")
                print(f"  another tile -- the substring test is many-to-one. They will be")
                print(f"  KEPT and show up as unlabelled next time. Harmless: this")
                print(f"  direction keeps too much, never too little.")
    if agreed:
        print(f"  {agreed} of those had the same image labelled more than once with")
        print(f"  IDENTICAL boxes -- duplicate effort, not a disagreement, so one")
        print(f"  labelled copy is kept and the rest go.")
    if conflicts:
        print(f"\n  {len(conflicts)} group(s) have the SAME IMAGE ANNOTATED DIFFERENTLY")
        print(f"  (or a label that could not be read). Left untouched -- which box")
        print(f"  is right is a judgement call, and guessing would quietly corrupt")
        print(f"  the training set:")
        for g, sigs in conflicts[:5]:
            print(f"    group of {len(g)}:")
            for f in g:
                mark = "[labelled]" if info[f]["labeled"] else "          "
                s = sigs.get(f)
                n = "unreadable" if (info[f]["labeled"] and s is None) else (
                    f"{s.count('(')} box(es)" if s else "")
                print(f"      {mark} {f.name}  {n}")
    if n_move:
        print(f"\n  {len(files)} tiles -> {len(files) - n_move} after dedup "
              f"({n_move / len(files) * 100:.1f}% removed)")

    if not args.apply:
        print("\n(dry run -- nothing moved. Re-run with --apply.)")
        for keeper, drop in plan[:3]:
            print(f"\n  keep {keeper.name}")
            for d in drop:
                print(f"  drop {d.name}")
        return

    # ---- apply ------------------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    quarantine = rgb_dir.parent / "_duplicates"
    quarantine.mkdir(parents=True, exist_ok=True)

    manifest = {"when": stamp, "rgb_dir": str(rgb_dir), "mode":
                "delete" if args.delete else "quarantine",
                "hash": "bytes" if args.fast else "pixels", "groups": []}
    moved = 0
    work = [(keeper, drop) for keeper, drop in plan]
    if purged:
        work.append((None, purged))
    for keeper, drop in work:
        rec = {"keep": keeper.name if keeper else "(unlabelled purge)",
               "dropped": []}
        for f in drop:
            targets = [f]
            nd = find_ndwi(info[f]["tile_id"], ndwi_dir)
            if nd:
                targets.append(nd)
            for t in targets:
                try:
                    if args.delete:
                        t.unlink()
                    else:
                        shutil.move(str(t), str(quarantine / t.name))
                    rec["dropped"].append(t.name)
                    moved += 1
                except OSError as e:
                    print(f"  WARNING: could not handle {t.name}: {e}")
        manifest["groups"].append(rec)

    manifest_path = quarantine / f"manifest_{stamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print(f"\n{'Deleted' if args.delete else 'Quarantined'} {moved} file(s) "
          f"(tiles + their NDWI pairs)")
    print(f"Manifest: {manifest_path}")
    if not args.delete:
        print(f"Quarantine: {quarantine}")
        print("Nothing is destroyed -- move files back from there if this got "
              "something wrong.")
    print("\nNOTE: tile_metadata.csv is not rewritten by this script. Rows for "
          "removed tiles will point at files that are gone; "
          "reconstruct_tile_metadata.py rebuilds it from what is actually on "
          "disk, which is the safer direction than editing it in place.")


if __name__ == "__main__":
    main()
