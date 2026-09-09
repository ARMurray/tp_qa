"""
diagnose_label_matching.py
============================
convert_tiles_to_500m.py found 2026 tiles in metadata, 0 with a matching
label file. This checks WHY, by trying two different matching strategies
against the real label filenames on disk and reporting exactly what's
there -- rather than guessing at a fix a second time.

Strategy 1 (what convert_tiles_to_500m.py currently does): exact match on
{tile_id}_rgb.txt, matching label_app.R's clean-filename convention.

Strategy 2 (what 03_prepare_dataset.py does): parse_tile_stem() reverses
Label Studio's export-mangled filenames ({hash}__Users%5C...{tile_stem}_rgb.txt
or {hash}-{tile_stem}_rgb.txt) back to a clean tile_stem. Reused directly
from 03_prepare_dataset.py via importlib rather than reimplemented, so
there's no risk of the two scripts silently drifting apart on this logic.

Usage:
    python diagnose_label_matching.py
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

# Reuse 03_prepare_dataset.py's own filename parser rather than
# reimplementing it -- same importlib pattern already used elsewhere in this
# project (01c reusing 01b, 05 reusing 02) for exactly this reason.
_spec = importlib.util.spec_from_file_location(
    "prep", Path(__file__).resolve().parent / "03_prepare_dataset.py")
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def main():
    print("=== diagnose_label_matching.py ===")
    print(f"Metadata: {C.METADATA_CSV}")
    print(f"Labels  : {C.ANNOTATION_DIR / 'labels'}\n")

    meta = pd.read_csv(C.METADATA_CSV, dtype={"CWNS_ID": str, "ll_uuid_primary": str})
    print(f"Tiles in metadata: {len(meta)}")
    print(f"Sample tile_id values:")
    for t in meta["tile_id"].head(5):
        print(f"  {t}")

    label_files = sorted((C.ANNOTATION_DIR / "labels").glob("*.txt"))
    print(f"\nLabel files on disk: {len(label_files)}")
    print(f"Sample filenames (raw, as they actually exist):")
    for f in label_files[:5]:
        print(f"  {f.name}")

    # --- Strategy 1: exact match on {tile_id}_rgb.txt ---
    tile_ids = set(meta["tile_id"])
    label_stems_clean = {f.stem for f in label_files}  # drops .txt only
    strategy1_matches = sum(1 for tid in tile_ids if f"{tid}_rgb" in label_stems_clean)
    print(f"\n--- Strategy 1 (exact {{tile_id}}_rgb.txt match) ---")
    print(f"  Matches: {strategy1_matches} / {len(tile_ids)}")

    # --- Strategy 2: 03_prepare_dataset.py's parse_tile_stem() ---
    parsed_count, unparsed_count = 0, 0
    unparsed_samples = []
    tile_stem_to_label = {}
    for f in label_files:
        result = prep.parse_tile_stem(f.name)
        if result is None:
            unparsed_count += 1
            if len(unparsed_samples) < 5:
                unparsed_samples.append(f.name)
        else:
            cwns_id, tile_stem = result
            parsed_count += 1
            tile_stem_to_label[tile_stem] = f

    print(f"\n--- Strategy 2 (03_prepare_dataset.py's parse_tile_stem) ---")
    print(f"  Parsed successfully: {parsed_count} / {len(label_files)}")
    print(f"  Failed to parse    : {unparsed_count} / {len(label_files)}")
    if unparsed_samples:
        print(f"  Sample unparsed filenames:")
        for s in unparsed_samples:
            print(f"    {s}")

    # Cross-match parsed tile_stems against metadata's tile_id column.
    # tile_stem format: {cwns_id}_{ll_uuid}_r{row}_c{col} -- should equal
    # tile_id directly if metadata's tile_id was built the same way
    # (confirmed via 02_extract_tiles.py's f"{cwns}_{primary}_r{row:02d}_c{col:02d}").
    strategy2_matches = sum(1 for tid in tile_ids if tid in tile_stem_to_label)
    print(f"\n  Cross-matched against metadata tile_id: {strategy2_matches} / {len(tile_ids)}")

    if strategy2_matches > strategy1_matches:
        print(f"\n>>> Strategy 2 (Label Studio filename parsing) is the right one -- "
              f"convert_tiles_to_500m.py needs to use parse_tile_stem() instead of "
              f"assuming clean filenames.")
    elif strategy1_matches == 0 and strategy2_matches == 0:
        print(f"\n>>> NEITHER strategy matched anything. Something else is going on -- "
              f"send this script's full output, especially the sample filenames above, "
              f"before any further fix is attempted.")

    # --- CWNS-ID-level comparison: is this actually a filename-format
    # problem, or are labels/ and tile_metadata.csv tracking almost entirely
    # DIFFERENT tiles? (2/2026 matching strongly suggests the latter --
    # SAMPLE_GPKG is literally named training_sample_round2.gpkg, implying
    # labels/ may hold round 1's tiles while metadata.csv reflects round 2.) ---
    label_cwns_ids = set()
    for f in label_files:
        result = prep.parse_tile_stem(f.name)
        if result:
            label_cwns_ids.add(result[0])
    meta_cwns_ids = set(meta["CWNS_ID"])

    overlap = label_cwns_ids & meta_cwns_ids
    print(f"\n--- CWNS_ID-level comparison ---")
    print(f"  Distinct CWNS_IDs in labels/        : {len(label_cwns_ids)}")
    print(f"  Distinct CWNS_IDs in tile_metadata   : {len(meta_cwns_ids)}")
    print(f"  Overlap                              : {len(overlap)}")

    if len(overlap) < len(label_cwns_ids) * 0.5:
        print(f"\n>>> Most labeled plants aren't in tile_metadata.csv at ALL -- this "
              f"isn't a filename-parsing problem, labels/ and tile_metadata.csv are "
              f"tracking mostly DIFFERENT tiles. Likely cause: a different/earlier "
              f"extraction round (SAMPLE_GPKG is named training_sample_round2.gpkg, "
              f"implying a round 1 existed with a different tile grid). Check whether "
              f"an OLDER tile_metadata.csv exists somewhere (a backup, a different "
              f"folder, a round1 file) that actually corresponds to these 264 labels.")
    elif overlap:
        # Some CWNS_IDs DO appear in both -- check whether the parcel (ll_uuid)
        # resolved the same way both times, or whether the grid regenerated
        # differently even for shared plants.
        sample_overlap = list(overlap)[:5]
        print(f"\n  For overlapping CWNS_IDs, checking whether ll_uuid_primary matches "
              f"(sample of {len(sample_overlap)}):")
        for cwns in sample_overlap:
            label_uuids = {prep.parse_tile_stem(f.name)[1].split("_")[1] for f in label_files
                           if prep.parse_tile_stem(f.name) and prep.parse_tile_stem(f.name)[0] == cwns}
            meta_uuids = set(meta.loc[meta["CWNS_ID"] == cwns, "ll_uuid_primary"].astype(str))
            print(f"    CWNS {cwns}: label ll_uuid(s)={label_uuids}, "
                  f"metadata ll_uuid(s)={meta_uuids}, "
                  f"{'MATCH' if label_uuids & meta_uuids else 'DIFFERENT'}")

    print("\n=== complete ===")


if __name__ == "__main__":
    main()