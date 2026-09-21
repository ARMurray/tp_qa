"""
update_master_locations.py
===========================
Folds completed review verdicts from app.db back into the master CWNS
locations file, writing a new dated layer.

Python port of data/pull_reviews.R, with the master moved from
Updates.gdb to Updates.gpkg -- read AND write are the same file now, so
each run builds on the previous one instead of re-reading a frozen gdb
and silently dropping the last round (the R version read the gdb and
wrote the gpkg, so round 2 would have lost round 1).

WHY THIS EXISTS
---------------
Reviewing model output has two payoffs: it documents how the model is
doing, and it grows the pool of verified-correct locations. This script
is what makes the second one stick -- a verdict submitted in the review
app is worthless to the next training run until it lands in the master
file that build_training_bins.py reads.

    review app (app.db)
        -> update_master_locations.py   [this script]
        -> Updates.gpkg / CWNS_Locations_YYYYMMDD
        -> build_training_bins.py
        -> training_locations.gpkg (classes / corrections / unverified)
        -> 01a / 01b / 01c / 02 / 03 / 04 ...

VERDICT -> MASTER COLUMNS
-------------------------
Matches 11_ingest_review_log.py's verdict handling exactly -- that script
derives the same truth from the same verdicts for the HPC-side review
gpkg, and the two must not disagree about what a verdict means.

  reported_correct          Original_Correct = Yes, Corrected = No
                            No Corrected_X/Y. The reported point stands.

  candidate_correct         Original_Correct = No, Corrected = Yes
                            Corrected_X/Y = the SELECTED PARCEL's
                            point-on-surface, looked up live against the
                            local Regrid mirror. The app only ever stores
                            a parcel id (selected_ll_uuid), never a
                            coordinate -- `plants.latitude/longitude` is
                            the REPORTED point (see queue_loader.py's
                            _insert_plant), so using it as the correction
                            writes Corrected == Original and teaches the
                            model that a wrong point is right. This lookup
                            is the only place a real corrected coordinate
                            comes from.
                            How_Corrected = "Model"

  truth_outside_candidates  Original_Correct = No, Corrected = Yes
                            Corrected_X/Y = the reviewer's clicked point,
                            already exact, no lookup needed.
                            How_Corrected = "Manual_Review"

  needs_info                skipped -- no decision was reached, so the row
                            stays exactly as it was (unverified).

Every verdict above sets Verified = "Yes" on the master row.

POINT-ON-SURFACE, NOT CENTROID: a polygon's geometric centroid can fall
outside the polygon for concave / L-shaped / multipart parcels, which
Regrid has plenty of. ST_PointOnSurface guarantees a point inside. Same
reasoning and same function as 11_ingest_review_log.py.

UPDATE IN PLACE, NOT REBUILD: reviewed rows are updated on the existing
master row rather than reconstructed from app.db columns joined back to
the master (what the R version did). The master carries ~15 CWNS
attribute columns this script has no business rewriting, and a rebuild
silently drops any column not named in its select list.

GEOMETRY is best-available: Corrected_X/Y where present, else
Original_X/Y -- which is what the existing master already holds and what
build_training_bins.py's docstring assumes. The R version rebuilt
geometry from the reported LONGITUDE/LATITUDE for every row, which
regressed already-corrected rows back to their wrong point.

CRS: EPSG:4269 (NAD83), matching build_training_bins.REPORTED_CRS and
what 02_feature_engineering.py's build_stage2_training() assumes when it
reprojects Original_X/Y and Corrected_X/Y. The R version hardcoded 4326.

HOLDOUT: excluded by default. The holdout is frozen evaluation data
(correction/scripts/holdout.py) and push_review_log.py already routes it
to a separate file that never reaches the training feed; letting it in
here would smuggle it back in through the master. Pass --include-holdout
only if you have deliberately decided the master should carry holdout
truth too, and remember 12_score_holdout.py's numbers stop meaning
anything the moment those rows reach training.

ENVIRONMENT: geopandas + pyogrio for GPKG I/O, duckdb (with spatial) for
the parcel lookup. Run from the review_app directory:

    python -m sync.update_master_locations --dry-run
    python -m sync.update_master_locations
    python -m sync.update_master_locations --master /path/to/Updates.gpkg
"""
import argparse
import re
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from backend import parcels

# ---------------------------------------------------------------------------
# Master file conventions
# ---------------------------------------------------------------------------
# Dated layers in the master gpkg. YYYYMMDD -- sorts lexicographically in
# date order, which is the whole point of the convention and what
# correction/scripts/config.py's MASTER_LAYER already uses. (The R version
# wrote %m%d%Y while reading a %Y%m%d layer, so the file accumulated two
# conventions no sort could order.)
LAYER_PREFIX = "CWNS_Locations_"
LAYER_DATE_FMT = "%Y%m%d"
# The _vN tail is for same-day re-runs (see next_layer_name). It MUST be
# matched here too -- correction/scripts/config.py's MASTER_LAYER_RE matches
# it, and if the two regexes disagree the HPC build and this script would
# resolve different "newest" layers from the same file.
LAYER_RE = re.compile(rf"^{re.escape(LAYER_PREFIX)}(\d{{8}})(?:_v\d+)?$")

REPORTED_CRS = "EPSG:4269"  # matches build_training_bins.py exactly

# Written into How_Corrected. Kept as named constants because they are a
# vocabulary shared with whatever reads the master downstream -- if the
# existing file uses different strings, change them here, in one place.
HOW_CORRECTED_MODEL = "Model"
HOW_CORRECTED_MANUAL = "Manual_Review"

# Columns this script is allowed to write on a master row. Everything else
# on the row (the CWNS attribute columns) is left untouched.
VERDICT_COLUMNS = [
    "Verified", "Original_Correct", "Corrected",
    "Corrected_X", "Corrected_Y", "How_Corrected",
]

VERDICTS_WITH_TRUTH = ("reported_correct", "candidate_correct",
                       "truth_outside_candidates")

# Diagnostic, not a label. Every truth_outside_candidates verdict is a case
# where Stage 2a's k-ring search did not surface the true parcel at all, so
# the running count of these is the empirical input to the K_RINGS decision
# (config.py section 6). Inherited from the retired 11_ingest_review_log.py,
# which was the only thing that produced it.
RECALL_FAILURES_NAME = "candidate_recall_failures.parquet"


# ---------------------------------------------------------------------------
# Master layer discovery
# ---------------------------------------------------------------------------
def parse_layer_date(layer: str):
    """Date encoded in a CWNS_Locations_YYYYMMDD layer name, or None.

    Tolerates the %m%d%Y layers the R script may already have written into
    an existing file: an 8-digit string is tried as YYYYMMDD first, then
    MMDDYYYY. Anything that parses as neither is not a dated layer and is
    ignored rather than silently sorting to an arbitrary position.
    """
    m = LAYER_RE.match(layer)
    if not m:
        return None
    digits = m.group(1)
    for fmt in (LAYER_DATE_FMT, "%m%d%Y"):
        try:
            return datetime.strptime(digits, fmt).date()
        except ValueError:
            continue
    return None


def find_latest_layer(master_path: Path) -> str:
    """Newest dated CWNS_Locations layer in the master gpkg.

    Sorted by the PARSED date, not by name -- see parse_layer_date. Raises
    rather than falling back to "whatever the driver listed first", which
    is what the R version effectively did once its mdy() parse returned NA
    for every layer and its arrange(desc(date)) became a no-op.
    """
    import pyogrio

    if not master_path.exists():
        raise FileNotFoundError(f"Master gpkg not found: {master_path}")

    names = [str(n) for n in pyogrio.list_layers(master_path)[:, 0]]
    dated = [(parse_layer_date(n), n) for n in names]
    dated = [(d, n) for d, n in dated if d is not None]
    if not dated:
        raise ValueError(
            f"No {LAYER_PREFIX}YYYYMMDD layers in {master_path}. "
            f"Found: {names}")
    # Sort key is (date, name) so a same-day _v2 sorts after the unsuffixed
    # layer written earlier that day -- same rule as config.latest_master_layer.
    dated.sort()
    return dated[-1][1]


def next_layer_name(existing_layers: list[str], run_date: date) -> str:
    """Today's layer name, suffixed if today's already exists.

    Re-running on the same day is normal (review a few more, push again),
    and overwriting the layer written an hour ago loses that run's rows.
    """
    base = f"{LAYER_PREFIX}{run_date.strftime(LAYER_DATE_FMT)}"
    if base not in existing_layers:
        return base
    for i in range(2, 100):
        candidate = f"{base}_v{i}"
        if candidate not in existing_layers:
            return candidate
    raise RuntimeError(f"Too many layers already written for {base}")


# ---------------------------------------------------------------------------
# Verdict loading
# ---------------------------------------------------------------------------
def load_verdicts(db_path: Path, review_round: int | None,
                  include_holdout: bool) -> pd.DataFrame:
    """Reviewed plants from app.db, as the reviewer left them."""
    if not db_path.exists():
        raise FileNotFoundError(f"app.db not found: {db_path}")

    query = "SELECT * FROM plants WHERE reviewed = 1"
    params: list = []
    if review_round is not None:
        query += " AND review_round = ?"
        params.append(review_round)

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return df
    df["cwns_id"] = df["cwns_id"].astype(str)

    n_before = len(df)
    if not include_holdout:
        df = df[~df["is_holdout"].astype(bool)].copy()
        if len(df) < n_before:
            print(f"  holdout: excluded {n_before - len(df)} reviewed plant(s) "
                  f"(pass --include-holdout to keep them -- read this script's "
                  f"docstring first)")

    n_before = len(df)
    df = df[df["plant_verdict"].isin(VERDICTS_WITH_TRUTH)].copy()
    n_skipped = n_before - len(df)
    if n_skipped:
        print(f"  needs_info (and any unrecognized verdict): skipped "
              f"{n_skipped} plant(s) -- no decision reached, master row "
              f"left unverified")
    return df


def resolve_corrected_points(df: pd.DataFrame) -> dict:
    """{cwns_id: (lon, lat)} for every plant whose correction needs one.

    candidate_correct rows resolve their selected parcel to a
    point-on-surface against the local Regrid mirror, grouped by state so
    it's one query per state rather than one per plant.
    truth_outside_candidates rows already carry an exact clicked point and
    need no lookup.
    """
    out: dict[str, tuple[float, float]] = {}

    for _, r in df[df["plant_verdict"] == "truth_outside_candidates"].iterrows():
        lon, lat = r.get("truth_longitude"), r.get("truth_latitude")
        if pd.isna(lon) or pd.isna(lat):
            print(f"  WARNING: {r['cwns_id']} is truth_outside_candidates but has "
                  f"no truth_latitude/truth_longitude -- skipping (the app's "
                  f"verdict validation requires these, so this shouldn't happen)")
            continue
        out[r["cwns_id"]] = (float(lon), float(lat))

    picks = df[df["plant_verdict"] == "candidate_correct"]
    if picks.empty:
        return out

    con = parcels.get_connection()
    for state, grp in picks.groupby("state_code"):
        wanted = grp.dropna(subset=["selected_ll_uuid"])
        uuids = wanted["selected_ll_uuid"].unique().tolist()
        if not uuids:
            continue
        glob_pattern = str(C.REGRID_ROOT / C.REGRID_STATE_GLOB.format(state=state))
        uuid_list = ", ".join(f"'{u}'" for u in uuids)
        try:
            # ST_PointOnSurface, NOT ST_Centroid -- see module docstring.
            found = con.execute(f"""
                SELECT {C.REGRID_UUID_COL} AS ll_uuid,
                       ST_X(ST_PointOnSurface(ST_GeomFromWKB({C.REGRID_GEOM_COL}))) AS cx,
                       ST_Y(ST_PointOnSurface(ST_GeomFromWKB({C.REGRID_GEOM_COL}))) AS cy
                FROM read_parquet('{glob_pattern}')
                WHERE {C.REGRID_UUID_COL} IN ({uuid_list})
            """).df()
        except Exception as e:
            print(f"  WARNING: parcel lookup failed for state={state}: {e}")
            print(f"           glob tried: {glob_pattern}")
            print(f"           {len(wanted)} candidate_correct plant(s) in this "
                  f"state will be skipped rather than written with a missing "
                  f"or wrong corrected coordinate")
            continue

        by_uuid = {r["ll_uuid"]: (r["cx"], r["cy"]) for _, r in found.iterrows()}
        for _, r in wanted.iterrows():
            pt = by_uuid.get(r["selected_ll_uuid"])
            if pt is None:
                print(f"  WARNING: {r['cwns_id']} selected parcel "
                      f"{r['selected_ll_uuid']} not found in the local Regrid "
                      f"mirror for state={state} -- skipping this plant rather "
                      f"than writing a corrections row with no real coordinate")
                continue
            out[r["cwns_id"]] = (float(pt[0]), float(pt[1]))

    return out


def build_updates(df: pd.DataFrame, points: dict) -> pd.DataFrame:
    """One row per plant, holding only the VERDICT_COLUMNS to write."""
    rows = []
    for _, r in df.iterrows():
        cwns_id = r["cwns_id"]
        verdict = r["plant_verdict"]

        if verdict == "reported_correct":
            rows.append({
                "CWNS_ID": cwns_id, "Verified": "Yes",
                "Original_Correct": "Yes", "Corrected": "No",
                "Corrected_X": None, "Corrected_Y": None, "How_Corrected": None,
            })
            continue

        pt = points.get(cwns_id)
        if pt is None:
            # Already warned about in resolve_corrected_points.
            continue
        rows.append({
            "CWNS_ID": cwns_id, "Verified": "Yes",
            "Original_Correct": "No", "Corrected": "Yes",
            "Corrected_X": pt[0], "Corrected_Y": pt[1],
            "How_Corrected": (HOW_CORRECTED_MODEL
                              if verdict == "candidate_correct"
                              else HOW_CORRECTED_MANUAL),
        })

    return pd.DataFrame(rows, columns=["CWNS_ID"] + VERDICT_COLUMNS)


# ---------------------------------------------------------------------------
# Master update
# ---------------------------------------------------------------------------
def apply_updates(existing, updates: pd.DataFrame):
    """Existing master with verdict columns overwritten for reviewed rows.

    Updates IN PLACE on the matched row -- every other column keeps its
    value. CWNS_IDs not present in the master are reported and dropped:
    the master is the full CWNS universe, so a miss means an id mismatch
    worth looking at, not a new facility to invent a row for.
    """
    import geopandas as gpd

    out = existing.copy()
    out["CWNS_ID"] = out["CWNS_ID"].astype(str)

    # Widen the verdict columns before writing into them. A master layer
    # where Corrected_X/Y is entirely null reads back from GPKG as a STRING
    # column, and assigning a float into it raises rather than upcasting
    # (pandas 2.x arrow-backed str dtype). That is exactly the state of a
    # master that has never had a correction written to it -- i.e. the first
    # run of this script, which is the one that must not fail.
    numeric_cols = {"Corrected_X", "Corrected_Y"}
    for col in VERDICT_COLUMNS:
        if col not in out.columns:
            print(f"  NOTE: master has no '{col}' column -- adding it")
            out[col] = pd.Series([None] * len(out), dtype="float64"
                                 if col in numeric_cols else "object")
        elif col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        else:
            out[col] = out[col].astype("object")

    known = set(out["CWNS_ID"])
    missing = sorted(set(updates["CWNS_ID"]) - known)
    if missing:
        print(f"  WARNING: {len(missing)} reviewed CWNS_ID(s) are not in the "
              f"master layer at all and were NOT added: {missing}")
        updates = updates[updates["CWNS_ID"].isin(known)]

    idx = out.index[out["CWNS_ID"].isin(set(updates["CWNS_ID"]))]
    by_id = updates.set_index("CWNS_ID")
    ids = out.loc[idx, "CWNS_ID"]
    for col in VERDICT_COLUMNS:
        out.loc[idx, col] = ids.map(by_id[col]).values

    # Geometry = best available point: corrected where we have one, else the
    # reported point. Rebuilt for the WHOLE layer, not just updated rows, so
    # rows corrected in an earlier round keep their corrected geometry.
    lon = out["Original_X"].astype(float)
    lat = out["Original_Y"].astype(float)
    cx = pd.to_numeric(out["Corrected_X"], errors="coerce")
    cy = pd.to_numeric(out["Corrected_Y"], errors="coerce")
    has_corr = cx.notna() & cy.notna()
    lon = lon.where(~has_corr, cx)
    lat = lat.where(~has_corr, cy)

    n_nullpt = int((lon.isna() | lat.isna()).sum())
    if n_nullpt:
        print(f"  NOTE: {n_nullpt} row(s) have no usable point (null "
              f"Original_X/Y and no correction) -- written with null geometry")

    geom_col = getattr(out, "_geometry_column_name", None) or "geometry"
    return gpd.GeoDataFrame(
        out.drop(columns=[geom_col], errors="ignore"),
        geometry=gpd.points_from_xy(lon, lat),
        crs=REPORTED_CRS,
    ), len(updates)


def write_recall_failures(verdicts: pd.DataFrame, out_dir: Path) -> int:
    """Append this run's truth_outside_candidates rows to the recall-failure
    log, deduped on (cwns_id, review_round) so re-runs are safe.

    Upload this alongside the training gpkg: 08_diagnose_candidate_coverage.py
    reports how many corrections fall outside the k-ring window, and these
    rows are the reviewer-confirmed cases of exactly that.
    """
    toc = verdicts[verdicts["plant_verdict"] == "truth_outside_candidates"]
    if toc.empty:
        return 0
    df = pd.DataFrame({
        "cwns_id": toc["cwns_id"],
        "review_round": toc["review_round"],
        "reviewed_at": toc["reviewed_at"],
        "reviewer": toc["reviewer"],
        "truth_latitude": toc["truth_latitude"],
        "truth_longitude": toc["truth_longitude"],
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / RECALL_FAILURES_NAME
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df = df.drop_duplicates(subset=["cwns_id", "review_round"], keep="last")
    df.to_parquet(path, index=False)
    print(f"  {RECALL_FAILURES_NAME}: {len(df)} row(s) total "
          f"({len(toc)} from this run) -> {path}")
    return len(toc)


def summarize(gdf, label: str):
    verified = (gdf["Verified"] == "Yes").sum()
    correct = ((gdf["Verified"] == "Yes") & (gdf["Original_Correct"] == "Yes")).sum()
    incorrect = ((gdf["Verified"] == "Yes") & (gdf["Original_Correct"] == "No")).sum()
    corrected = (gdf["Corrected"] == "Yes").sum()
    print(f"  {label}: {len(gdf)} rows | verified {verified} "
          f"({correct} correct / {incorrect} incorrect) | corrected {corrected}")


def main():
    ap = argparse.ArgumentParser(
        description="Fold review-app verdicts into the master CWNS locations gpkg.")
    ap.add_argument("--master", default=None,
                    help="path to Updates.gpkg. Defaults to config.MASTER_GPKG.")
    ap.add_argument("--layer", default=None,
                    help="source layer to build on. Defaults to the newest "
                         "dated CWNS_Locations layer in the master.")
    ap.add_argument("--out-layer", default=None,
                    help="layer name to write. Defaults to "
                         "CWNS_Locations_<today>, suffixed if it exists.")
    ap.add_argument("--round", type=int, default=None,
                    help="only fold in verdicts from this review round. "
                         "Default: every reviewed plant in app.db.")
    ap.add_argument("--include-holdout", action="store_true",
                    help="also write holdout plants' verdicts. Off by "
                         "default -- read this script's docstring first.")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change and write nothing.")
    args = ap.parse_args()

    import geopandas as gpd
    import pyogrio

    master_path = Path(args.master) if args.master else C.MASTER_GPKG
    print("=== update_master_locations.py ===")
    print(f"Master: {master_path}")

    src_layer = args.layer or find_latest_layer(master_path)
    print(f"Source layer: {src_layer}")

    existing = gpd.read_file(master_path, layer=src_layer)
    summarize(existing, "before")

    print(f"\nLoading verdicts from {C.APP_DB_PATH} ...")
    verdicts = load_verdicts(C.APP_DB_PATH, args.round, args.include_holdout)
    if verdicts.empty:
        print("No usable verdicts to fold in -- nothing to do.")
        return
    print(f"  {len(verdicts)} usable verdict(s):")
    for v, n in verdicts["plant_verdict"].value_counts().items():
        print(f"    {v}: {n}")

    print("\nResolving corrected coordinates ...")
    points = resolve_corrected_points(verdicts)
    updates = build_updates(verdicts, points)
    print(f"  {len(updates)} row(s) ready to write "
          f"({int((updates['Corrected'] == 'Yes').sum())} with a corrected point)")

    if updates.empty:
        print("Nothing resolvable -- not writing a layer.")
        return

    out_gdf, n_applied = apply_updates(existing, updates)
    print(f"\nApplied {n_applied} update(s).")
    summarize(out_gdf, "after ")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    layers = [str(n) for n in pyogrio.list_layers(master_path)[:, 0]]
    out_layer = args.out_layer or next_layer_name(layers, date.today())
    print(f"\nWriting layer {out_layer} to {master_path} ...")
    out_gdf.to_file(master_path, layer=out_layer, driver="GPKG")

    print("\nRecall-failure diagnostics ...")
    n_rf = write_recall_failures(verdicts, C.OUTGOING_DIR)
    if not n_rf:
        print(f"  no truth_outside_candidates verdicts this run -- nothing to log")

    print("\n=== complete ===")
    print(f"NEXT: point correction/scripts/config.py at this file if it isn't "
          f"already, then rebuild training bins:")
    print(f"  python build_training_bins.py --out training_locations.gpkg")


if __name__ == "__main__":
    main()
