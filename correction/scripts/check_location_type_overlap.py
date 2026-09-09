"""
check_location_type_overlap.py
================================
check_location_type.py found no duplicate CWNS_IDs in PHYSICAL_LOCATION.txt,
but also found that 26% of its rows have LOCATION_TYPE != "Point" (City/
County/Watershed/State) -- presumably administrative-area centroids, not the
plant's actual coordinate. load_treatment_plants() doesn't filter on
LOCATION_TYPE at all. This checks whether any of those non-Point rows
actually survive into the treatment-plant-filtered universe the pipeline
scores, and if so, how many, and whether they're concentrated in any
particular label bin (classes/corrections) where it would most directly
distort training.

Usage:
    python check_location_type_overlap.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def main():
    print("=== check_location_type_overlap.py ===")

    facility_types = pd.read_csv(C.CWNS_DIR / "FACILITY_TYPES.txt", dtype=str, encoding="latin1")
    treatment_ids = set(
        facility_types.loc[facility_types["FACILITY_TYPE"] == "Treatment Plant", "CWNS_ID"])
    print(f"Treatment plants (FACILITY_TYPES): {len(treatment_ids)}")

    loc = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt", dtype=str, encoding="latin1")
    loc_tp = loc[loc["CWNS_ID"].isin(treatment_ids)]
    print(f"\nPHYSICAL_LOCATION rows for treatment plants: {len(loc_tp)}")
    print(loc_tp["LOCATION_TYPE"].value_counts(dropna=False).to_string())

    non_point = loc_tp[loc_tp["LOCATION_TYPE"] != "Point"]
    n_non_point = len(non_point)
    pct = 100 * n_non_point / len(loc_tp) if len(loc_tp) else 0
    print(f"\nNon-Point rows within the treatment-plant universe: "
          f"{n_non_point} ({pct:.2f}%)")

    if n_non_point:
        print("\nSample non-Point treatment-plant rows (up to 10):")
        cols = [c for c in ["CWNS_ID", "LOCATION_TYPE", "LATITUDE", "LONGITUDE",
                             "STATE_CODE", "CITY", "COUNTY_NAME"] if c in non_point.columns]
        print(non_point[cols].head(10).to_string(index=False))

        # Overlap with label bins, if the training gpkg is available
        if C.TRAINING_GPKG.exists():
            non_point_ids = set(non_point["CWNS_ID"])
            for layer in (C.TRAINING_LAYER_CLASSES, C.TRAINING_LAYER_CORRECTIONS):
                try:
                    gdf = gpd.read_file(C.TRAINING_GPKG, layer=layer)
                except Exception as e:
                    print(f"\n  Could not read layer {layer}: {e}")
                    continue
                gdf["CWNS_ID"] = gdf["CWNS_ID"].astype(str)
                overlap = gdf["CWNS_ID"].isin(non_point_ids).sum()
                print(f"\n  Layer '{layer}': {overlap} / {len(gdf)} plants "
                      f"have a non-Point reported location")
        else:
            print(f"\n  {C.TRAINING_GPKG} not found -- skipping label-bin overlap check")
    else:
        print("\nNo overlap -- non-Point rows are entirely outside the treatment-plant "
              "universe. This concern does not apply to the pipeline as currently scoped.")

    print("\n=== complete ===")


if __name__ == "__main__":
    main()
