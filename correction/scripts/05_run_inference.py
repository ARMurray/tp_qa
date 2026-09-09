"""
05_run_inference.py
====================
Stage 1 -> Stage 2a inference over a full-universe plant set. Stage 2b is
DELIBERATELY EXCLUDED from this pilot round -- see TPQA_MASTER_REFERENCE.md
S3/S10: OD features contribute almost nothing to the current Stage 2b model
(od_max_conf_aeration_basin led OD features at ~0.005 importance vs.
parcel_bbox_max_dim_m at 0.134), so building the not-yet-written OD-on-
arbitrary-top-K step to feed it is likely wasted effort until the review loop
(REVIEW_LOOP_PLAN.md Phase 5) supplies distribution-matched training data.
Revisit this decision once that data exists.

Reads the full-universe 05_plant_features.parquet / 10_parcel_features.parquet
that 02_feature_engineering.py --full-universe already built for the requested
states -- does NOT regenerate them. Run 02 first if they're missing or stale
for the states you're asking for.

Reuses point_in_parcel_lookup() and add_name_matching() from
02_feature_engineering.py via importlib (matching 01c_run_od_corrected_
locations.py's existing pattern for reusing 01b's logic) rather than
reimplementing them, since drift between two copies of the same join logic
is exactly the failure mode that broke Stage 1 training silently before
(the H3-shortcut bug documented in 02's point_in_parcel_lookup docstring).

KNOWN WART, replicated deliberately: both deployed Stage 1 and Stage 2a
models were trained with raw Albers-projected coordinates (x_5070, y_5070)
as live numeric features -- they were added only for spatial-CV clustering
in 03_train_stage1.py/04_train_stage2.py and never dropped before the model
saw them. The fitted pipelines will KeyError without these columns, so
inference computes and supplies them here to match the deployed models.
This is flagged, not fixed -- fixing it means retraining, which changes the
currently-deployed models' behavior. See TPQA_MASTER_REFERENCE.md S1/S10.

NEW IN THIS SCRIPT, not present anywhere else in the pipeline yet:
  - Reported-parcel exclusion: a candidate parcel equal to the plant's own
    reported parcel is dropped before scoring. 02's build_stage2_training has
    an `is_reported` column but never actually sets it True or filters on it
    -- this was a placeholder, not a working exclusion (confirmed 2026-08-25).
  - Candidate competition resolution: when two plants' candidate pools
    overlap on the same parcel, only the highest-scoring plant keeps it in
    the competition-resolved top pick. Resolved per state (parcels are
    state-scoped by construction, so cross-state competition can't occur
    within K_RINGS). Full candidate pool is preserved regardless -- this
    only affects which single candidate is reported as "the" top pick.

Output (in data/inference/), names chosen fresh rather than matching the old
app.R's stage1_results.parquet/stage2_results.parquet convention, since that
app no longer exists and its replacement's needs aren't designed yet
(TPQA_MASTER_REFERENCE.md S10, open question):
    plant_summary.parquet       one row per plant scored
    stage2_candidates.parquet   one row per (plant, candidate) pair, all
                                 flagged plants, full candidate pool

Usage:
    python 05_run_inference.py --states OH,MS,DE
    python 05_run_inference.py --states OH,MS,DE --top-k 20
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import duckdb
import geopandas as gpd
import h3
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

# ---------------------------------------------------------------------------
# Reuse 02_feature_engineering.py's join logic directly rather than
# reimplementing it -- filename starts with a digit, so a normal `import`
# statement isn't available; same importlib pattern 01c already uses for 01b.
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "feat_eng", Path(__file__).resolve().parent / "02_feature_engineering.py")
feat_eng = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(feat_eng)

point_in_parcel_lookup = feat_eng.point_in_parcel_lookup
add_name_matching = feat_eng.add_name_matching


# ===========================================================================
# Loading
# ===========================================================================
def load_full_universe_plants(states: list[str]) -> pd.DataFrame:
    """Every treatment plant in the requested states, reported LAT/LON +
    STATE_CODE + h3_res9. Mirrors 02's load_treatment_plants(training_only=
    False) but standalone here since that function isn't cleanly separable
    from the module's sys.path/import setup for a clean reuse."""
    facility_types = pd.read_csv(C.CWNS_DIR / "FACILITY_TYPES.txt", dtype=str, encoding="latin1")
    treatment_ids = set(
        facility_types.loc[facility_types["FACILITY_TYPE"] == "Treatment Plant", "CWNS_ID"])
    loc = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt", dtype={"CWNS_ID": str}, encoding="latin1")
    loc = loc[loc["CWNS_ID"].isin(treatment_ids)]
    loc["LATITUDE"] = pd.to_numeric(loc["LATITUDE"], errors="coerce")
    loc["LONGITUDE"] = pd.to_numeric(loc["LONGITUDE"], errors="coerce")
    n_before = len(loc)
    loc = loc.dropna(subset=["LATITUDE", "LONGITUDE"]).drop_duplicates(subset="CWNS_ID")
    if len(loc) < n_before:
        print(f"  Dropped {n_before - len(loc)} rows with non-numeric/missing coordinates")

    # point_location (2026-08-26) -- see 02_feature_engineering.py's
    # load_treatment_plants for the full reasoning. Same derivation, kept in
    # sync deliberately rather than shared via importlib, since this
    # function's docstring already notes it's standalone for import-setup
    # reasons.
    loc["point_location"] = loc["LOCATION_TYPE"] == "Point"

    loc = loc[["CWNS_ID", "STATE_CODE", "LATITUDE", "LONGITUDE", "point_location"]]
    loc = loc[loc["STATE_CODE"].isin(states)].reset_index(drop=True)
    loc["h3_res9"] = loc.apply(
        lambda r: h3.latlng_to_cell(r["LATITUDE"], r["LONGITUDE"], 9), axis=1)
    return loc


def load_od_features(od_dir: Path) -> pd.DataFrame | None:
    """Same union/dedup pattern 02's main() uses for 01b's append-only
    per-partition output -- duplicated here rather than extracted, since it's
    simple, already well-exercised, and not the part of the codebase that's
    been fragile. See 02_feature_engineering.py's own inline comments for why
    each step (one-file-at-a-time read, category->str fix, mtime-sorted
    keep='last' dedup) is there."""
    plants_dir = od_dir / "plants"
    if not plants_dir.exists() or not any(plants_dir.rglob("*.parquet")):
        return None
    part_files_by_partition = {}
    for f in plants_dir.rglob("part-*.parquet"):
        part_files_by_partition.setdefault(f.parent, []).append(f)
    frames = []
    for files in part_files_by_partition.values():
        for f in sorted(files, key=lambda f: f.stat().st_mtime):
            df_part = pd.read_parquet(f)
            for col in df_part.select_dtypes(include=["category"]).columns:
                df_part[col] = df_part[col].astype(str)
            frames.append(df_part)
    if not frames:
        return None
    od = pd.concat(frames, ignore_index=True)
    od["CWNS_ID"] = od["CWNS_ID"].astype(str)
    od = od.drop_duplicates(subset="CWNS_ID", keep="last")
    od = od.drop(columns=["state", "orig_lon", "orig_lat", "n_objects_total",
                          "n_objects_in_parcel", "processed_at"], errors="ignore")
    return od


def load_model(name: str) -> dict:
    path = C.MODELS_DIR / f"{name}_rf_model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Train {name} first (03_train_stage1.py / "
            f"04_train_stage2.py) before running inference.")
    bundle = joblib.load(path)
    threshold = joblib.load(C.MODELS_DIR / f"{name}_optimal_threshold.joblib")
    print(f"  Loaded {path.name}: {len(bundle['feature_columns'])} feature columns, "
          f"threshold={threshold:.3f}")
    return bundle, threshold


def add_projected_coords(df: pd.DataFrame, lon_col="LONGITUDE", lat_col="LATITUDE") -> pd.DataFrame:
    """x_5070/y_5070 -- see module docstring's KNOWN WART note. Required to
    match the deployed models' trained feature schema, not because these are
    good features."""
    coords = gpd.GeoSeries(
        gpd.points_from_xy(df[lon_col], df[lat_col]), crs=4326).to_crs(5070)
    df = df.copy()
    df["x_5070"] = coords.x.to_numpy()
    df["y_5070"] = coords.y.to_numpy()
    return df


def score(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Reindex to the model's exact trained column set/order before scoring
    -- safe regardless of how the fitted ColumnTransformer selects columns
    internally, and fails loudly (KeyError) if a required feature is
    missing, rather than silently scoring on the wrong columns."""
    X = df.reindex(columns=bundle["feature_columns"])
    missing = X.columns[X.isna().all()].tolist()
    truly_missing = [c for c in missing if c not in df.columns]
    if truly_missing:
        raise KeyError(
            f"Model expects columns not present in the inference frame at all: "
            f"{truly_missing}. Feature engineering has drifted from what the "
            f"model was trained on.")

    # Belt and braces against unhashable values reaching OneHotEncoder. The
    # known offender (geom_wkb) is dropped at source in run_stage1, but this
    # frame is assembled from a wide external-data join chain and a single
    # bytes/bytearray/list cell anywhere in a categorical column takes the
    # whole run down inside sklearn with a traceback that points at the
    # encoder rather than at the data. Fail here instead, naming the column.
    for col in X.columns:
        non_null = X[col].dropna()
        if len(non_null) == 0:
            continue
        first = non_null.iloc[0]
        if isinstance(first, (bytes, bytearray, memoryview, list, dict, set)):
            raise TypeError(
                f"Column '{col}' contains non-scalar values (found "
                f"{type(first).__name__}) and cannot be one-hot encoded. "
                f"Something upstream is leaking a raw/binary column into the "
                f"feature frame -- fix it at the source rather than coercing "
                f"here, since a silently str()-ified blob is a garbage feature.")

    return bundle["pipeline"].predict_proba(X)[:, 1]


# ===========================================================================
# Stage 1
# ===========================================================================
def run_stage1(con, plants: pd.DataFrame, plant_features: pd.DataFrame,
               parcel_features: pd.DataFrame, od_features: pd.DataFrame | None,
               bundle: dict, threshold: float) -> pd.DataFrame:
    print("\n=== Stage 1: reported-location scoring ===")
    reported = []
    for state, grp in plants.groupby("STATE_CODE"):
        pts = pd.DataFrame({
            "CWNS_ID": grp["CWNS_ID"], "h3_res9": grp["h3_res9"],
            "geom_wkb": gpd.points_from_xy(grp["LONGITUDE"], grp["LATITUDE"]).map(lambda g: g.wkb),
        })
        matched = point_in_parcel_lookup(con, state, pts)
        if len(matched):
            reported.append(matched)
    reported_parcels = pd.concat(reported, ignore_index=True) if reported else \
        pd.DataFrame(columns=["CWNS_ID", "ll_uuid"])
    # point_in_parcel_lookup does `SELECT pts.*, p.ll_uuid`, so everything we
    # registered into `pts` comes back too -- including geom_wkb, which DuckDB
    # returns as a bytearray. If that column survives into the scored frame,
    # build_preprocessor()'s select_dtypes picks it up as an object-dtype
    # "categorical" feature and OneHotEncoder dies on it with
    # "TypeError: unhashable type: 'bytearray'" (confirmed 2026-08-25).
    # 02's training path never hit this because its Stage 1 builder consumes
    # the frame differently; drop the lookup scaffolding explicitly here
    # rather than relying on a downstream drop that may not exist.
    reported_parcels = reported_parcels.drop(
        columns=["geom_wkb", "h3_res9"], errors="ignore")
    reported_parcels = reported_parcels.drop_duplicates(subset="CWNS_ID") \
        .rename(columns={"ll_uuid": "reported_ll_uuid"})

    out = plants.merge(reported_parcels, on="CWNS_ID", how="left")
    no_parcel_mask = out["reported_ll_uuid"].isna()
    n_no_parcel = int(no_parcel_mask.sum())
    print(f"  Plants: {len(out)}  |  reported parcel found: {len(out) - n_no_parcel}  "
          f"|  no parcel: {n_no_parcel} ({100 * n_no_parcel / len(out):.1f}%)")

    scoreable = out[~no_parcel_mask].copy()
    if len(scoreable):
        scoreable = scoreable.merge(
            plant_features.drop(columns=["STATE_CODE", "LATITUDE", "LONGITUDE"], errors="ignore"),
            on="CWNS_ID", how="left"
        ).merge(
            parcel_features.rename(columns={"ll_uuid": "reported_ll_uuid"}).drop(columns=["state"], errors="ignore"),
            on="reported_ll_uuid", how="left"
        )
        if od_features is not None and len(od_features):
            n_before = len(scoreable)
            scoreable = scoreable.merge(od_features, on="CWNS_ID", how="left")
            scoreable["od_ran"] = scoreable["od_ran"].fillna(False)
            scoreable["od_has_detection"] = scoreable["od_has_detection"].fillna(False)
            for c in [c for c in scoreable.columns if c.startswith("od_has_")]:
                scoreable[c] = scoreable[c].fillna(False)
            for c in [c for c in scoreable.columns if c.startswith("od_n_")]:
                scoreable[c] = scoreable[c].fillna(0)
            assert len(scoreable) == n_before, "OD join changed row count -- duplicate CWNS_IDs in od_features"
        else:
            print("  WARNING: no OD features available -- Stage 1 od_* columns will be all-default")

        scoreable = add_name_matching(scoreable)
        scoreable = add_projected_coords(scoreable)

        probs = score(bundle, scoreable)
        scoreable["stage1_prob_correct"] = probs
    else:
        scoreable["stage1_prob_correct"] = pd.Series(dtype=float)

    out = out.merge(
        scoreable[["CWNS_ID", "stage1_prob_correct"]] if len(scoreable) else
        pd.DataFrame(columns=["CWNS_ID", "stage1_prob_correct"]),
        on="CWNS_ID", how="left")

    out["trigger_reason"] = np.select(
        [no_parcel_mask, out["stage1_prob_correct"] < threshold],
        ["no_parcel", "low_confidence"], default="none")

    n_flagged = int((out["trigger_reason"] != "none").sum())
    print(f"  Flagged for Stage 2: {n_flagged} "
          f"({int((out['trigger_reason'] == 'no_parcel').sum())} no_parcel + "
          f"{int((out['trigger_reason'] == 'low_confidence').sum())} low_confidence)")
    return out


# ===========================================================================
# Stage 2a
# ===========================================================================
def build_candidates_for_plant(plant_row, state_candidates: pd.DataFrame) -> pd.DataFrame | None:
    """Mirrors 02_feature_engineering.py's build_stage2_training inner loop
    body (candidate selection within K_RINGS, distance features from the
    REPORTED point) -- not extracted into a shared function because that
    loop is tightly woven into build_stage2_training's corrections-layer-
    specific setup (correct_id/label columns) which doesn't apply here.
    Replicated deliberately rather than diverging in behavior."""
    plant_h3_cells = set(h3.grid_disk(plant_row["h3_res9"], C.K_RINGS))
    candidates = state_candidates[state_candidates["h3_index_9"].isin(plant_h3_cells)].copy()
    if len(candidates) == 0:
        return None

    # Reported-parcel exclusion -- NEW, see module docstring. Never implemented
    # anywhere in the pipeline before this (02's `is_reported` column exists
    # but is always False and unused).
    reported_uuid = plant_row.get("reported_ll_uuid")
    if pd.notna(reported_uuid):
        candidates = candidates[candidates["ll_uuid"] != reported_uuid]
    if len(candidates) == 0:
        return None

    centers = candidates["h3_index_9"].apply(h3.cell_to_latlng)
    candidates["centroid_lat"] = centers.apply(lambda c: c[0])
    candidates["centroid_lng"] = centers.apply(lambda c: c[1])

    cand_sf = gpd.GeoDataFrame(
        candidates, geometry=gpd.points_from_xy(candidates["centroid_lng"], candidates["centroid_lat"]),
        crs=4326).to_crs(5070)
    reported_pt = gpd.GeoSeries(
        gpd.points_from_xy([plant_row["LONGITUDE"]], [plant_row["LATITUDE"]]), crs=4326
    ).to_crs(5070).iloc[0]
    distances = cand_sf.geometry.distance(reported_pt).to_numpy()

    candidates["CWNS_ID"] = plant_row["CWNS_ID"]
    candidates["distance_m"] = distances
    candidates["log_distance"] = np.log1p(distances)
    candidates["within_1km"] = distances <= 1000
    candidates["within_5km"] = distances <= 5000
    return candidates


def run_stage2a(flagged: pd.DataFrame, plant_features: pd.DataFrame,
                parcel_features: pd.DataFrame, bundle: dict, top_k: int) -> pd.DataFrame:
    print("\n=== Stage 2a: candidate scoring ===")
    rows = []
    for state in flagged["STATE_CODE"].dropna().unique():
        nlcd_path = C.NLCD_OUTPUT_DIR / f"nlcd_{state}_k{C.K_RINGS}.parquet"
        if not nlcd_path.exists():
            print(f"  No 01a output for {state} -- skipping its flagged plants")
            continue
        state_candidates = pd.read_parquet(nlcd_path, columns=["ll_uuid", "h3_index_9"])
        state_candidates = state_candidates.merge(
            parcel_features.drop(columns=["state", "h3_index_9"], errors="ignore"),
            on="ll_uuid", how="left")

        state_plants = flagged[flagged["STATE_CODE"] == state]
        t0 = time.time()
        for i, (_, plant_row) in enumerate(state_plants.iterrows(), 1):
            cands = build_candidates_for_plant(plant_row, state_candidates)
            if cands is not None:
                rows.append(cands)
            if i % 100 == 0:
                print(f"    {state}: {i}/{len(state_plants)} plants "
                      f"({(time.time() - t0) / 60:.1f} min elapsed)")
        print(f"  {state}: {len(state_plants)} flagged plants processed")

    if not rows:
        print("  No candidates generated for any flagged plant.")
        return pd.DataFrame()

    cand_df = pd.concat(rows, ignore_index=True)
    print(f"  Raw candidates (post reported-parcel exclusion): {len(cand_df)}")

    cand_df = cand_df.merge(
        flagged[["CWNS_ID", "STATE_CODE", "LATITUDE", "LONGITUDE"]], on="CWNS_ID", how="left")

    # Plant-level features, mirroring 02's build_stage2_training, which merges
    # its stage2_plants frame before calling add_name_matching. This step was
    # missing here (found 2026-08-25): the merge above brings in only
    # STATE_CODE/LATITUDE/LONGITUDE, so cand_df reached add_name_matching with
    # no subdivision/place/county and died on KeyError: 'subdivision'. Even
    # past that, Stage 2a is TRAINED on pop_served and is_rural as well as the
    # six name-match booleans, so scoring without this merge would have failed
    # in score()'s "columns not present in the inference frame" check -- the
    # Stage 1 path a hundred lines up already does exactly this merge, and the
    # two paths had drifted. STATE_CODE/LATITUDE/LONGITUDE are dropped from the
    # right side because `flagged` just supplied them and pandas would suffix
    # both copies to _x/_y, leaving neither under its plain name.
    cand_df = cand_df.merge(
        plant_features.drop(columns=["STATE_CODE", "LATITUDE", "LONGITUDE"], errors="ignore"),
        on="CWNS_ID", how="left")

    cand_df = add_name_matching(cand_df)
    cand_df = add_projected_coords(cand_df)

    probs = score(bundle, cand_df)
    cand_df["stage2_prob_correct"] = probs

    return cand_df


def resolve_competition(cand_df: pd.DataFrame) -> pd.DataFrame:
    """Per-state: when two plants' candidate pools overlap on the same
    parcel, only the highest-scoring plant keeps it as ITS top pick. Full
    candidate pool (cand_df) is untouched -- this only picks, per plant, its
    single best surviving candidate for the plant-level summary. Mirrors the
    R-era build_results.R design (see TPQA_MASTER_REFERENCE.md S3): resolve
    only at final-assignment time, never on the candidate pool itself."""
    if cand_df.empty:
        return cand_df
    out = []
    for state, grp in cand_df.groupby("STATE_CODE"):
        # parcel goes to its single highest-scoring claimant
        winners = grp.sort_values("stage2_prob_correct", ascending=False) \
                     .drop_duplicates(subset="ll_uuid", keep="first")
        # then each plant's best surviving candidate
        top = winners.sort_values("stage2_prob_correct", ascending=False) \
                     .drop_duplicates(subset="CWNS_ID", keep="first")
        out.append(top)
    return pd.concat(out, ignore_index=True)


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=str, required=True,
                     help="comma-separated, e.g. --states OH,MS,DE. Must already have "
                          "full-universe 05_plant_features.parquet/10_parcel_features.parquet "
                          "from 02_feature_engineering.py --full-universe, and 01a NLCD "
                          "output, for these states.")
    ap.add_argument("--top-k", type=int, default=20,
                     help="max candidates kept per flagged plant in the output table "
                          "(default 20 -- REVIEW_LOOP_PLAN.md's K=5 is the REVIEWED count, "
                          "not the stored candidate-pool size; keep this larger so "
                          "candidate_recall can be measured against a real pool).")
    args = ap.parse_args()
    states = [s.strip() for s in args.states.split(",")]

    C.ensure_dirs()
    inference_dir = C.DATA_DIR / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)

    print("=== 05_run_inference.py ===")
    print(f"States: {states}\n")

    print("Loading Stage 1 / Stage 2a models...")
    stage1_bundle, stage1_threshold = load_model("stage1")
    stage2_bundle, stage2_threshold = load_model("stage2")

    print("\nLoading full-universe plant list...")
    plants = load_full_universe_plants(states)
    print(f"  {len(plants)} plants across {plants['STATE_CODE'].nunique()} state(s)")

    print("\nLoading plant/parcel features (from 02_feature_engineering.py --full-universe)...")
    plant_features = pd.read_parquet(C.FEATURES_OUTPUT_DIR / "05_plant_features.parquet")
    parcel_features = pd.read_parquet(C.FEATURES_OUTPUT_DIR / "10_parcel_features.parquet")
    missing_states = set(states) - set(plant_features["STATE_CODE"].dropna().unique())
    if missing_states:
        print(f"  WARNING: {missing_states} not present in 05_plant_features.parquet -- "
              f"those plants will have no plant-level features. Re-run 02 with "
              f"--full-universe --states including them.")
    print(f"  Plant features: {len(plant_features)} rows  |  "
          f"Parcel features: {len(parcel_features)} rows")

    print("\nLoading OD features (01b reported-location output)...")
    od_features = load_od_features(C.OD_OUTPUT_DIR)
    if od_features is None:
        print("  WARNING: no 01b output found -- proceeding without OD features")
    else:
        print(f"  {len(od_features)} plants with OD output")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; SET enable_geoparquet_conversion = false;")

    stage1_out = run_stage1(con, plants, plant_features, parcel_features,
                            od_features, stage1_bundle, stage1_threshold)
    con.close()

    flagged = stage1_out[stage1_out["trigger_reason"] != "none"].copy()
    cand_df = run_stage2a(flagged, plant_features, parcel_features,
                          stage2_bundle, args.top_k)

    if not cand_df.empty:
        top_picks = resolve_competition(cand_df)
        top_picks = top_picks.rename(columns={
            "ll_uuid": "top_candidate_ll_uuid", "stage2_prob_correct": "top_stage2_prob",
            "distance_m": "top_distance_m", "within_1km": "top_within_1km",
            "within_5km": "top_within_5km",
        })[["CWNS_ID", "top_candidate_ll_uuid", "top_stage2_prob", "top_distance_m",
            "top_within_1km", "top_within_5km"]]

        n_cands_per_plant = cand_df.groupby("CWNS_ID").size().rename("n_candidates")
        # keep only the top --top-k candidates per plant in the stored table
        cand_out = cand_df.sort_values(["CWNS_ID", "stage2_prob_correct"], ascending=[True, False])
        cand_out = cand_out.groupby("CWNS_ID", group_keys=False).head(args.top_k)

        stage1_out = stage1_out.merge(top_picks, on="CWNS_ID", how="left")
        stage1_out = stage1_out.merge(n_cands_per_plant, on="CWNS_ID", how="left")
        stage1_out["n_candidates"] = stage1_out["n_candidates"].fillna(0).astype(int)
    else:
        cand_out = pd.DataFrame()
        stage1_out["top_candidate_ll_uuid"] = None
        stage1_out["top_stage2_prob"] = np.nan
        stage1_out["top_distance_m"] = np.nan
        stage1_out["top_within_1km"] = False
        stage1_out["top_within_5km"] = False
        stage1_out["n_candidates"] = 0

    summary_path = inference_dir / "plant_summary.parquet"
    cand_path = inference_dir / "stage2_candidates.parquet"
    stage1_out.to_parquet(summary_path, index=False)
    cand_out.to_parquet(cand_path, index=False)

    print("\n=== Summary ===")
    print(f"  Plants scored          : {len(stage1_out)}")
    print(f"  Flagged for Stage 2     : {len(flagged)}")
    print(f"  Plants with >=1 candidate: {int((stage1_out['n_candidates'] > 0).sum())}")
    print(f"  Plants with 0 candidates : {int((stage1_out['n_candidates'] == 0).sum())} "
          f"(flagged but no parcel survived K_RINGS + reported-parcel exclusion)")
    print(f"\nWritten: {summary_path}")
    print(f"Written: {cand_path}")


if __name__ == "__main__":
    main()
