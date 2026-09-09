"""
list_training_states.py
=========================
Prints every STATE_CODE present in the training universe (union of
'classes' + 'corrections' layers), with a plant count per state. Run this
before submitting a nationwide 01a/01b array job or a multi-state 02 run --
sizes the array, and the per-state counts (especially the largest one)
matter for 02's memory budget, since 01a's OH run alone pulled 935k parcel
rows at k=18.

Usage:
    python list_training_states.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def main():
    classes = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CLASSES)
    corrections = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CORRECTIONS)

    training_ids = set(classes["CWNS_ID"].astype(str)) | set(corrections["CWNS_ID"].astype(str))
    print(f"Training universe: {len(training_ids)} unique CWNS_IDs\n")

    loc = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt", dtype={"CWNS_ID": str}, encoding="latin1")
    loc = loc[loc["CWNS_ID"].isin(training_ids)]
    loc = loc.drop_duplicates(subset="CWNS_ID")

    counts = loc["STATE_CODE"].value_counts().sort_index()
    print(f"{'STATE_CODE':<12}{'plants':>8}")
    for state, n in counts.items():
        print(f"{state:<12}{n:>8}")

    print(f"\nTotal states/territories: {len(counts)}")
    print(f"Total plants (state-known): {counts.sum()}")
    print(f"Largest single state: {counts.idxmax()} with {counts.max()} plants")

    print("\n--- Copy-paste ready ---")
    print("Space-separated (for --array + 01a/01b):")
    print(" ".join(counts.index.tolist()))
    print("\nComma-separated (for 02's --states):")
    print(",".join(counts.index.tolist()))
    print(f"\nArray range for sbatch: --array=0-{len(counts) - 1}")


if __name__ == "__main__":
    main()
