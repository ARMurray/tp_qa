"""
08_build_training_manifest_hpc.py
==================================
Builds the TRAINING manifest that 09_run_manifest_pipeline_hpc.py consumes.

Counterpart to 00_build_full_plant_list_hpc.py, but for the Stage 1 training
set rather than the production universe. The difference matters:

  00 reads plants.gpkg, whose geometry is the BEST-AVAILABLE location per
     plant -- including manual corrections. Correct for production inference.

  08 reads training_locations.gpkg layer 'classes', whose geometry is the
     location that MATCHES THE LABEL:
        class == "Correct"    -> Corrected_X/Y (correct.1) or Original_X/Y
                                 (correct.2/correct.3)
        class == "Incorrect"  -> Original_X/Y  (incorrect.1/incorrect.2)

     For the ~302 corrected plants, the same CWNS_ID appears TWICE -- once as
     Incorrect at its original point, once as Correct at its corrected point.
     That is the matched pair we want: the same plant's wrong parcel and right
     parcel, both tiled, both scored.

USING plants.gpkg HERE WOULD INVERT THE LABEL. Plants labeled Incorrect would
have detection computed on their CORRECTED parcel, teaching Stage 1 that
"infrastructure present => incorrect location." Do not substitute the inputs.

Because CWNS_ID is no longer unique, the primary key throughout the training
pipeline is `sample_id` = "{CWNS_ID}__{label_class}". CWNS_ID is retained as a
GROUPING key -- any cross-validation split must group on it, or a plant's
Correct and Incorrect rows land in opposite folds and the detection feature
will look far better than it is.

Output: a Parquet manifest with one row per sample, parcel already resolved.

    sample_id, CWNS_ID, label_class, st, geoid, state_fips,
    ll_uuid, parcel_found, n_parcel_matches, parcel_area_m2,
    rep_lon, rep_lat

Rows where the point intersects no parcel are KEPT with parcel_found=False.
They cannot be tiled, but they are real Stage 1 observations (the "no parcel"
condition) and dropping them here would silently bias the training set toward
parcel-covered -- i.e. urban, i.e. larger -- plants.

Usage:
    python 08_build_training_manifest_hpc.py \
        --classes-gpkg data/training_locations.gpkg \
        --out data/samples/training_manifest.parquet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import from_wkb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

# The _hpc renaming convention left 07's import statement pointing at the old
# module name; tolerate either so this script doesn't inherit that landmine.
try:
    from state_fips import fips_to_abbr, FIPS_TO_ABBR
except ImportError:
    from state_fips_hpc import fips_to_abbr, FIPS_TO_ABBR


def validate_cwns_ids(cwns_ids: pd.Series):
    """Same leading-zero-truncation guard as 00_build_full_plant_list_hpc.py.
    Worth repeating rather than importing: the training gpkg comes from a
    different R script (Update_Training_Plants.R) and can lose leading zeros
    independently of plants.gpkg."""
    lengths = cwns_ids.str.len().value_counts()
    if len(lengths) > 1:
        print(f"  WARNING: CWNS_ID lengths inconsistent -- possible truncation:\n{lengths}")

    prefixes = cwns_ids.str[:2]
    bad = sorted(set(prefixes) - set(FIPS_TO_ABBR.keys()))
    if bad:
        n_bad = int(prefixes.isin(bad).sum())
        raise SystemExit(
            f"CWNS_ID validation FAILED: {n_bad} rows have a 2-character prefix that isn't "
            f"a known state FIPS code: {bad}\n"
            f"Almost always means CWNS_ID was stored numerically somewhere and lost a "
            f"leading zero. Re-export as character from Update_Training_Plants.R."
        )
    print(f"  CWNS_ID validation OK -- all {len(cwns_ids)} prefixes match a state FIPS code")


def attach_county_geoid(pts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Mirrors 00's approach. Needed to locate the right parcel parquet file."""
    counties = gpd.read_file(C.COUNTIES_GPKG)[["GEOID", "geometry"]].to_crs(C.EXPORT_CRS)
    counties["GEOID"] = counties["GEOID"].astype(str).str.zfill(5)
    joined = gpd.sjoin(pts, counties, how="left", predicate="intersects")
    joined = joined.drop(columns=["index_right"]).rename(columns={"GEOID": "geoid"})
    # A point on a county boundary can match twice; keep one deterministically.
    return joined.sort_values(["sample_id", "geoid"]).drop_duplicates("sample_id")


def load_parcels(state: str, geoid: str):
    """Identical contract to 07's loader."""
    path = C.PARCEL_BASE / f"state={state}" / f"{geoid}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, columns=[C.PARCEL_ID_FIELD, C.PARCEL_WKB_FIELD])
    except Exception:
        df = pd.read_parquet(path)
    if C.PARCEL_WKB_FIELD not in df.columns:
        return None
    geom = from_wkb(df[C.PARCEL_WKB_FIELD].to_numpy())
    df = df.drop(columns=[C.PARCEL_WKB_FIELD])
    return gpd.GeoDataFrame(df, geometry=geom, crs=C.EXPORT_CRS)


def resolve_parcels(pts: gpd.GeoDataFrame) -> pd.DataFrame:
    """Point-in-parcel resolution, one county parquet read at a time.

    Regrid parcels overlap in places, so a point can intersect several. 07 used
    `uuids[0]` off an unsorted unique() -- non-deterministic if parquet row
    order ever shifts, which silently changes which parcel got tiled between
    runs. Here the match is sorted by (descending area, ll_uuid): prefer the
    parcel that actually contains the site over a sliver, break ties on a
    stable string. n_parcel_matches is recorded so ambiguous cases stay
    auditable rather than disappearing.
    """
    out_rows = []
    counties = pts[["st", "geoid"]].drop_duplicates().dropna()
    print(f"  Counties to resolve: {len(counties)}")

    for ci, (_, cc) in enumerate(counties.iterrows(), start=1):
        st, geoid = cc["st"], cc["geoid"]
        sub = pts[(pts["st"] == st) & (pts["geoid"] == geoid)]

        parcels = load_parcels(st, geoid)
        if parcels is None or len(parcels) == 0:
            for _, r in sub.iterrows():
                out_rows.append(dict(sample_id=r["sample_id"], ll_uuid=None,
                                     n_parcel_matches=0, parcel_area_m2=np.nan))
            if ci % 100 == 0 or ci == len(counties):
                print(f"    [{ci}/{len(counties)}] {st}/{geoid}: no parcel file")
            continue

        parcels = parcels.to_crs(C.PROJECTED_CRS)[[C.PARCEL_ID_FIELD, "geometry"]]
        parcels["parcel_area_m2"] = parcels.geometry.area

        joined = gpd.sjoin(sub.to_crs(C.PROJECTED_CRS), parcels,
                           how="left", predicate="intersects")

        for sid, grp in joined.groupby("sample_id"):
            matched = grp.dropna(subset=[C.PARCEL_ID_FIELD])
            if len(matched) == 0:
                out_rows.append(dict(sample_id=sid, ll_uuid=None,
                                     n_parcel_matches=0, parcel_area_m2=np.nan))
                continue
            matched = matched.sort_values(
                ["parcel_area_m2", C.PARCEL_ID_FIELD], ascending=[False, True])
            best = matched.iloc[0]
            out_rows.append(dict(
                sample_id=sid,
                ll_uuid=str(best[C.PARCEL_ID_FIELD]),
                n_parcel_matches=int(len(matched)),
                parcel_area_m2=float(best["parcel_area_m2"]),
            ))

        if ci % 100 == 0 or ci == len(counties):
            print(f"    [{ci}/{len(counties)}] {st}/{geoid} done")

    return pd.DataFrame(out_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes-gpkg", default=str(
        getattr(C, "TRAINING_LOCATIONS_GPKG", C.DATA_DIR / "training_locations.gpkg")),
        help="training_locations.gpkg written by Update_Training_Plants.R")
    ap.add_argument("--layer", default="classes")
    ap.add_argument("--out", default=str(
        getattr(C, "TRAIN_MANIFEST", C.SAMPLES_DIR / "training_manifest.parquet")))
    args = ap.parse_args()

    print("=== 08_build_training_manifest_hpc.py ===\n")

    pts = gpd.read_file(args.classes_gpkg, layer=args.layer)
    print(f"Loaded {len(pts)} labeled rows from {args.classes_gpkg} (layer '{args.layer}')")
    print(f"  Source CRS: {pts.crs}")

    if "class" not in pts.columns:
        raise SystemExit(
            f"Expected a 'class' column in layer '{args.layer}'. Found: {list(pts.columns)}. "
            f"This script requires the label-matched geometry from Update_Training_Plants.R, "
            f"not plants.gpkg -- see module docstring."
        )

    pts["CWNS_ID"] = pts["CWNS_ID"].astype(str)
    validate_cwns_ids(pts["CWNS_ID"])

    pts["label_class"] = pts["class"].astype(str).str.strip().str.title()
    unknown = set(pts["label_class"]) - {"Correct", "Incorrect"}
    if unknown:
        raise SystemExit(f"Unexpected label values in 'class': {sorted(unknown)}")

    pts["sample_id"] = pts["CWNS_ID"] + "__" + pts["label_class"]
    dupes = pts["sample_id"].duplicated().sum()
    if dupes:
        print(f"  NOTE: {dupes} duplicate sample_id rows collapsed "
              f"(same plant, same label, repeated by the rbind in Update_Training_Plants.R)")
        pts = pts.drop_duplicates("sample_id")

    pts = pts.to_crs(C.EXPORT_CRS)
    pts["rep_lon"] = pts.geometry.x
    pts["rep_lat"] = pts.geometry.y
    pts["state_fips"] = pts["CWNS_ID"].str[:2]
    pts["st"] = pts["state_fips"].map(fips_to_abbr)

    print("\nLabel balance:")
    print(pts["label_class"].value_counts().to_string())
    n_paired = pts.groupby("CWNS_ID")["label_class"].nunique().eq(2).sum()
    print(f"  Matched pairs (same CWNS_ID, both labels): {n_paired}")
    print("  -> CV must group on CWNS_ID, not sample_id.\n")

    print("Attaching county GEOID...")
    pts = attach_county_geoid(pts)
    n_nogeo = int(pts["geoid"].isna().sum())
    if n_nogeo:
        print(f"  WARNING: {n_nogeo} points fell outside every county polygon")

    print("\nResolving point-in-parcel...")
    resolved = resolve_parcels(pts[pts["geoid"].notna()])

    man = (pts.drop(columns=["geometry", "class"], errors="ignore")
              .merge(resolved, on="sample_id", how="left"))
    man["parcel_found"] = man["ll_uuid"].notna()
    man["n_parcel_matches"] = man["n_parcel_matches"].fillna(0).astype(int)

    keep = ["sample_id", "CWNS_ID", "label_class", "st", "geoid", "state_fips",
            "ll_uuid", "parcel_found", "n_parcel_matches", "parcel_area_m2",
            "rep_lon", "rep_lat"]
    man = man[keep].sort_values(["state_fips", "geoid", "sample_id"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    man.to_parquet(out_path, engine="pyarrow", index=False)

    print("\n=== Manifest summary ===")
    print(f"  Total samples          : {len(man)}")
    print(f"  Parcel resolved        : {int(man['parcel_found'].sum())} "
          f"({man['parcel_found'].mean():.1%})")
    print(f"  No parcel (untileable) : {int((~man['parcel_found']).sum())}")
    print(f"  Ambiguous (>1 parcel)  : {int((man['n_parcel_matches'] > 1).sum())}")
    print(f"  States present         : {man['state_fips'].nunique()}")
    print("\n  Tileable samples per state (array job sizing):")
    print(man[man["parcel_found"]]["state_fips"].value_counts().sort_index().to_string())
    print(f"\nWrote {out_path}")
    print("\nNext: 09_run_manifest_pipeline_hpc.py --state <FIPS>")


if __name__ == "__main__":
    main()
