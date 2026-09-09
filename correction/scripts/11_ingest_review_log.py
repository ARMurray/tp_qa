"""
11_ingest_review_log.py
=========================
Converts reviewed, non-holdout verdicts (review_log_round{N}.parquet, from
the local review app's sync/push_review_log.py, uploaded via MobaXterm) into
the SAME classes/corrections schema build_training_bins.py already produces
from Updates.gdb. This is the single feed point for ALL THREE models --
Stage 1, Stage 2a, and Stage 2b all train from training_locations.gpkg's
classes/corrections layers, so augmenting that file (rather than a
model-specific derived table) means every model benefits from review data
through the exact same, already-proven feature engineering pipeline. No
schema mismatch, no missing-feature gaps -- 02_feature_engineering.py builds
these rows' features identically to every other row's.

SUPERSEDES an earlier version of this script (2026-08-26) that emitted a
Stage-2b-specific 17_review_training.parquet consumed only by
06_build_stage2b_training.py's union step. That approach is now DEPRECATED --
06's union block should be reverted (see that script's own history) to avoid
double-counting the same plant through two feedback paths simultaneously.
The Stage-2b-only approach also structurally could not supply OD features or
reclassified LBCS categories for review-derived rows; routing through
training_locations.gpkg avoids that problem entirely, since these rows go
through 01a/01b/01c/02 exactly like any Updates.gdb row.

VERDICT -> BIN, matching build_training_bins.py's Correct/Incorrect/
Corrections definitions exactly:
  reported_correct          classes: class="Correct", Original_X/Y=reported point
  candidate_correct         classes: class="Incorrect", Original_X/Y=reported point
                             corrections: Corrected_X/Y = the SELECTED CANDIDATE
                             PARCEL'S CENTROID, looked up live against the real
                             PARCEL_BASE store (the app only captured a parcel ID,
                             not a point -- this is where that point comes from)
  truth_outside_candidates  classes: class="Incorrect", Original_X/Y=reported point
                             corrections: Corrected_X/Y = the reviewer's clicked
                             point (already exact, no lookup needed) + a row in
                             candidate_recall_failures (feeds the K_RINGS decision)
  needs_info                skipped entirely -- stays effectively "unverified,"
                             no usable signal

OUTPUT: data/training/review/review_derived_locations.gpkg (classes +
corrections layers, same column contract as build_training_bins.py's output)
plus data/features/candidate_recall_failures.parquet (unchanged from the
prior version of this script -- this part was never Stage-2b-specific).

NOT run standalone against training -- see build_training_bins.py's new
--review-gpkg argument, which unions this file's rows with Updates.gdb's,
Updates.gdb winning on any CWNS_ID conflict (it's the stated single source of
truth; a review verdict on an already-labeled plant is informational, not an
automatic override).

HOLDOUT: belt and braces, same as before -- push_review_log.py already
routes holdout plants to a separate file and never includes them in
review_log_round{N}.parquet, so this should be unreachable. Checked anyway.

Usage:
    python 11_ingest_review_log.py --round 1
    python 11_ingest_review_log.py --round 1 --dry-run
"""
import argparse
import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from holdout import exclude_holdout

INCOMING_DIR = C.DATA_DIR / "review_log_incoming"  # where MobaXterm uploads land
REVIEW_GPKG_DIR = C.TRAINING_DIR / "review"
REVIEW_GPKG_PATH = REVIEW_GPKG_DIR / "review_derived_locations.gpkg"
RECALL_FAILURES_PATH = C.FEATURES_OUTPUT_DIR / "candidate_recall_failures.parquet"

REPORTED_CRS = "EPSG:4269"  # matches build_training_bins.py's REPORTED_CRS exactly


def load_inputs(round_num: int):
    plant_path = INCOMING_DIR / f"review_log_round{round_num}.parquet"
    if not plant_path.exists():
        raise FileNotFoundError(
            f"{plant_path} not found. Upload it from the review app's "
            f"data/outgoing/ (via sync/push_review_log.py) into {INCOMING_DIR} first.")
    plants = pd.read_parquet(plant_path)
    plants["cwns_id"] = plants["cwns_id"].astype(str)
    return plants


def get_parcel_representative_points(con, plants_needing_lookup: pd.DataFrame) -> dict:
    """{(state, ll_uuid): (lon, lat)} for every selected_ll_uuid needing a
    training coordinate -- candidate_correct verdicts only. Looked up against
    the REAL PARCEL_BASE store on HPC, not any local mirror. Grouped by
    state, one query per state rather than one per plant.

    ST_PointOnSurface, NOT ST_Centroid (2026-08-27, caught before this ever
    shipped): a polygon's geometric centroid can fall OUTSIDE the polygon
    entirely for concave/L-shaped/multi-part parcels -- a real risk here,
    not a theoretical one, since Regrid parcels routinely have irregular
    boundaries. ST_PointOnSurface guarantees a point that is actually inside
    the polygon (GEOS's "interior point" algorithm), at the cost of being a
    less "central" point than a true centroid would be for a convex shape --
    an acceptable tradeoff, since a training coordinate that's outside the
    parcel entirely is a correctness bug, while one that's merely off-center
    is not."""
    out = {}
    for state, grp in plants_needing_lookup.groupby("state_code"):
        uuids = grp["selected_ll_uuid"].dropna().unique().tolist()
        if not uuids:
            continue
        uuid_list = ", ".join(f"'{u}'" for u in uuids)
        try:
            df = con.execute(f"""
                SELECT ll_uuid,
                       ST_X(ST_PointOnSurface(ST_GeomFromWKB(wkb_geometry))) AS cx,
                       ST_Y(ST_PointOnSurface(ST_GeomFromWKB(wkb_geometry))) AS cy
                FROM read_parquet('{C.PARCEL_BASE.as_posix()}/state={state}/*.parquet')
                WHERE ll_uuid IN ({uuid_list})
            """).df()
        except Exception as e:
            print(f"  WARNING: point-on-surface lookup failed for state={state}: {e} -- "
                  f"{len(uuids)} candidate_correct plant(s) in this state will be "
                  f"dropped rather than written with a missing/wrong coordinate")
            continue
        for _, row in df.iterrows():
            out[(state, row["ll_uuid"])] = (row["cx"], row["cy"])
    return out


def build_review_bins(plants: pd.DataFrame, con) -> tuple:
    classes_rows = []
    corrections_rows = []
    failures = []

    needs_centroid = plants[plants["plant_verdict"] == "candidate_correct"]
    centroids = get_parcel_representative_points(con, needs_centroid) if len(needs_centroid) else {}

    for _, plant in plants.iterrows():
        cwns_id = plant["cwns_id"]
        verdict = plant["plant_verdict"]
        orig_x, orig_y = plant["longitude"], plant["latitude"]

        if verdict == "needs_info":
            continue

        elif verdict == "reported_correct":
            classes_rows.append({"CWNS_ID": cwns_id, "class": "Correct",
                                 "Original_X": orig_x, "Original_Y": orig_y})

        elif verdict == "candidate_correct":
            key = (plant["state_code"], plant["selected_ll_uuid"])
            if key not in centroids:
                print(f"  WARNING: {cwns_id} selected_ll_uuid "
                      f"{plant['selected_ll_uuid']} has no resolvable point-on-surface "
                      f"(lookup failed or parcel not found in PARCEL_BASE) -- "
                      f"skipping this plant entirely rather than writing a "
                      f"corrections row with no real coordinate")
                continue
            cx, cy = centroids[key]
            classes_rows.append({"CWNS_ID": cwns_id, "class": "Incorrect",
                                 "Original_X": orig_x, "Original_Y": orig_y})
            corrections_rows.append({"CWNS_ID": cwns_id, "Original_X": orig_x, "Original_Y": orig_y,
                                     "Corrected_X": cx, "Corrected_Y": cy})

        elif verdict == "truth_outside_candidates":
            tx, ty = plant.get("truth_longitude"), plant.get("truth_latitude")
            if pd.isna(tx) or pd.isna(ty):
                print(f"  WARNING: {cwns_id} is truth_outside_candidates but has no "
                      f"truth_latitude/truth_longitude captured -- skipping (this "
                      f"shouldn't happen; the app's verdict validation requires "
                      f"these for this verdict type)")
                continue
            classes_rows.append({"CWNS_ID": cwns_id, "class": "Incorrect",
                                 "Original_X": orig_x, "Original_Y": orig_y})
            corrections_rows.append({"CWNS_ID": cwns_id, "Original_X": orig_x, "Original_Y": orig_y,
                                     "Corrected_X": tx, "Corrected_Y": ty})
            failures.append({"cwns_id": cwns_id, "review_round": plant["review_round"],
                             "reviewed_at": plant["reviewed_at"], "reviewer": plant["reviewer"],
                             "truth_latitude": ty, "truth_longitude": tx,
                             "n_candidates_shown": int(plant.get("n_candidates", 0) or 0)})
        else:
            print(f"  WARNING: {cwns_id} has unrecognized plant_verdict "
                  f"'{verdict}' -- skipping")

    def to_gdf(rows, x_col, y_col):
        if not rows:
            return gpd.GeoDataFrame(columns=["CWNS_ID"], geometry=[], crs=REPORTED_CRS)
        df = pd.DataFrame(rows)
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x_col], df[y_col]), crs=REPORTED_CRS)

    classes_gdf = to_gdf(classes_rows, "Original_X", "Original_Y")
    corrections_gdf = to_gdf(corrections_rows, "Original_X", "Original_Y")
    failures_df = pd.DataFrame(failures)

    return classes_gdf, corrections_gdf, failures_df


def append_deduped_parquet(df: pd.DataFrame, path: Path, dedup_cols: list, label: str):
    if df.empty:
        print(f"  {label}: nothing to write this round")
        return
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
        before = len(combined)
        combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
        dropped = before - len(combined)
        if dropped:
            print(f"  {label}: {dropped} row(s) already present (safe re-run) -- kept newest")
    else:
        combined = df
    combined.to_parquet(path, index=False)
    print(f"  {label}: {len(combined)} row(s) total ({len(df)} added/updated this run)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=== 11_ingest_review_log.py ===")
    print(f"Round: {args.round}")

    plants = load_inputs(args.round)
    print(f"\nLoaded {len(plants)} reviewed plant(s)")

    n_before = len(plants)
    plants = exclude_holdout(plants.rename(columns={"cwns_id": "CWNS_ID"}), "review-ingest") \
                .rename(columns={"CWNS_ID": "cwns_id"})
    if len(plants) < n_before:
        print(f"  [holdout] removed {n_before - len(plants)} plant(s) that should "
              f"never have reached this file -- push_review_log.py has a bug if "
              f"this fires with a nonzero count")

    print(f"\nVerdict breakdown:\n{plants['plant_verdict'].value_counts().to_string()}")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; SET enable_geoparquet_conversion = false;")

    classes_gdf, corrections_gdf, failures_df = build_review_bins(plants, con)
    con.close()

    print(f"\nBuilt classes: {len(classes_gdf)} rows "
          f"({int((classes_gdf['class']=='Correct').sum()) if len(classes_gdf) else 0} Correct, "
          f"{int((classes_gdf['class']=='Incorrect').sum()) if len(classes_gdf) else 0} Incorrect)")
    print(f"Built corrections: {len(corrections_gdf)} rows")
    print(f"Built candidate_recall_failures: {len(failures_df)} rows")

    if args.dry_run:
        print("\n--dry-run: not writing anything")
        return

    REVIEW_GPKG_DIR.mkdir(parents=True, exist_ok=True)

    # Merge with anything already in the review gpkg from a prior round --
    # same idempotency reasoning as everywhere else in this pipeline.
    for gdf, layer in ((classes_gdf, "classes"), (corrections_gdf, "corrections")):
        if gdf.empty:
            continue
        if REVIEW_GPKG_PATH.exists():
            try:
                existing = gpd.read_file(REVIEW_GPKG_PATH, layer=layer)
                combined = pd.concat([existing, gdf], ignore_index=True)
                combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=REPORTED_CRS)
                combined = combined.drop_duplicates(subset="CWNS_ID", keep="last")
            except Exception:
                combined = gdf  # layer doesn't exist yet in this file
        else:
            combined = gdf
        combined.to_file(REVIEW_GPKG_PATH, layer=layer, driver="GPKG")
        print(f"  {layer}: {len(combined)} row(s) total in {REVIEW_GPKG_PATH.name}")

    append_deduped_parquet(failures_df, RECALL_FAILURES_PATH,
                           dedup_cols=["cwns_id", "review_round"],
                           label=RECALL_FAILURES_PATH.name)

    print(f"\nNEXT:")
    print(f"  python build_training_bins.py --gdb <Updates.gdb> --layer <layer> "
          f"--out training_locations.gpkg --review-gpkg {REVIEW_GPKG_PATH}")
    print(f"  then re-run 01a/01c for any genuinely NEW plants/corrected points, "
          f"then 02, then 03/04/06/07.")
    print("=== complete ===")


if __name__ == "__main__":
    main()
