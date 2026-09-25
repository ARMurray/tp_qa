"""
extract_review_tiles.py
========================
Lives in review_app/analysis/, next to find_false_top_picks.py. Both
detection/ and review_app/ run locally (correction/ is the only HPC-side
piece), so this writes directly into the REAL detection/data/tiles/
inventory -- same rgb/png, ndwi, and tile_metadata.csv 02_extract_tiles.py
itself writes -- by loading that script's own tiling/fetch/metadata
functions in-process, rather than porting a second copy of that logic that
could drift out of sync.

WHAT THIS ADDS
    NAIP tiles (RGB PNG + NDWI GeoTIFF) for every parcel touched by a
    review-app verdict, for ALL reviewed plants -- not just the
    find_false_top_picks.py subset (which only pulls wrong-rank-1 cases):
      - reported_ll_uuid (plants table): present for every reviewed plant,
        regardless of verdict or review_task.
      - ll_uuid for every row in candidates (ALL ranks, not just rank 1)
        for that cwns_id -- exactly what the reviewer saw on screen.
    A parcel that's both (reported point sits on a candidate parcel, or the
    reviewer picked the parcel that IS the reported one) is tiled once;
    'source' in the metadata records review_reported/review_candidate,
    preferring review_reported when both apply.

TARGETED SELECTION IS NOW THE DEFAULT (2026-09-23)
    The behaviour described above -- tile the reported parcel plus every
    candidate -- is what --all-candidates still does, and it is where the
    unmanageable inventory came from: roughly 3,000 tiles per 150-plant
    round, of which 872 of the 1,004 eventually labelled turned out to be
    empty. Most of that volume is parcels the detector ignored and the
    reviewer rejected, where the model already agrees with the reviewer and
    the tile teaches nothing.

    The default now tiles only two things: wherever the reviewer established
    the plant actually is, and parcels where the detector found
    infrastructure but the reviewer said it was not the plant. See
    load_targeted_sites() for the full reasoning.

    That includes truth_outside_candidates clicks, which the old behaviour
    skipped entirely for want of a parcel. They are now tiled from a
    synthetic square centred on the click -- the same fallback 01b uses when
    it will not trust a parcel boundary. A verified true location is worth
    labelling whether or not Regrid has a polygon for it.

WHERE GEOMETRY COMES FROM
    backend.parcels.get_parcel_context() -- the SAME live DuckDB lookup
    against your local Regrid mirror the review app itself uses to draw
    parcels on screen. Using it here (rather than a fresh duckdb query)
    means "the parcel that gets tiled" is guaranteed to match "the parcel
    the reviewer actually saw."

WRITING INTO THE SAME INVENTORY AS 02_extract_tiles.py
    detection/pipeline/02_extract_tiles.py is loaded in-process (via
    importlib, same pattern reconstruct_tile_metadata.py and
    01c_run_od_corrected_locations.py already use elsewhere in this repo
    for exactly this reason: reusing deterministic tiling/fetch code beats
    a second hand-copy that can silently drift). That gets us, for free:
      - generate_tile_grid / tiles_to_wgs84 -- identical tile geometry
      - fetch_and_save / run_tasks -- identical NAIP fetch, retry, and
        cross-quad-boundary fallback
      - load_existing_tile_ids / existing_rgb -- real skip-existing against
        tile_metadata.csv and the real rgb/png folder, so a parcel already
        pulled by a normal 02_extract_tiles.py run (e.g. it was also in the
        190-plant training sample) is recognized and not re-fetched
      - append_metadata -- writes into the SAME tile_metadata.csv

    THE 'config' NAME COLLISION: review_app/config.py and
    detection/pipeline/config.py are two different files both imported
    under the bare module name 'config'. Python caches imports by name in
    sys.modules, so if review_app's config is imported first (needed here
    for APP_DB_PATH/backend.parcels), a later bare `import config` inside
    02_extract_tiles.py would silently reuse THAT cached module instead of
    detection/pipeline/'s own config.py -- tile output would point at
    review_app's paths with no error raised anywhere. load_detection_modules()
    below evicts sys.modules['config'] and puts detection/pipeline/ first
    on sys.path immediately before loading 02_extract_tiles.py, specifically
    to avoid this.

LABEL AWARENESS (--check-labels)
    Reports, per site, whether any of its tiles already has a matching
    label file in annotation/ls_export/labels/ -- best-effort match by
    tile-id substring, since 03_prepare_dataset.py's own docstring notes
    Label Studio's export filenames are URL-encoded/hash-prefixed rather
    than the plain tile_id. This does NOT gate fetching -- a plant with 5
    candidate parcels and 1 already labeled still needs tiles (and
    eventually labels) for the other 4 -- it's informational, so you can
    see at a glance which reviewed parcels still need annotating.

DEPENDENCIES
    Needs whatever environment 02_extract_tiles.py itself runs in locally
    (geopandas, shapely, rasterio, pyproj, pystac-client,
    planetary-computer, pillow -- see detection/requirements.txt). Run this
    script with that same interpreter/venv, not review_app's own (which
    only has fastapi/duckdb/pandas/pyarrow/pydantic).

OXIDATION PONDS / LAGOONS
    No pipeline change needed -- detection/pipeline/config.py's CLASSES
    already includes oxidation_pond (deferred from training, not deleted),
    and 02_extract_tiles.py already tiles at 500m, which was the blocker
    for lagoon-scale infrastructure under the old 200m geometry. Tiles this
    script writes are immediately usable for oxidation_pond annotation.
    Re-including the class in 04_train_model.py's KEEP_CLASSES is a
    separate, later step -- not part of pulling imagery.

OVERLAPPING PARCELS (2026-09-16)
    Two different ll_uuid candidates/reported parcels for the same
    cwns_id can be separate Regrid records for the same physical site
    (adjoining tracts, split lots). dedupe_overlapping_sites() collapses
    these before tiling so the same lagoons don't get fetched twice under
    two different tile-id prefixes -- see that function's docstring.
    Tune with --overlap-frac if it's merging too aggressively or missing
    real overlaps.

RESILIENCE (2026-09-16)
    build_tasks() runs resolve_item() sequentially for every site BEFORE
    any fetching starts -- a hard failure there used to crash the whole
    run and throw away all the planning work already done for a
    1000+-site pass. A single site's lookup failure (all of with_retry's
    attempts exhausted) is now caught, logged, and skipped rather than
    fatal -- see naip_fetch.py's own with_retry() for the matching fix to
    what counts as a retryable error in the first place (connection resets
    weren't being retried at all before that fix).

Usage:
    python extract_review_tiles.py                    # all reviewed plants
    python extract_review_tiles.py --dry-run
    python extract_review_tiles.py --reported-only     # skip candidate parcels
    python extract_review_tiles.py --candidates-only   # skip reported parcels
    python extract_review_tiles.py --check-labels      # also report label coverage
    python extract_review_tiles.py --limit 200         # cap new tiles, test run
    python extract_review_tiles.py --detection-root "D:\\somewhere\\else\\detection"
"""
import argparse
import importlib.util
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box, shape

# review_app/config.py + backend/ -- one level up from analysis/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C          # review_app's config (APP_DB_PATH, Regrid mirror)
from backend import parcels

TP_QA_ROOT = C.APP_ROOT.parent
DETECTION_PIPELINE_DIR_DEFAULT = TP_QA_ROOT / "detection" / "pipeline"


def load_detection_modules(pipeline_dir: Path):
    """Loads detection/pipeline/'s config.py, naip_fetch.py, and
    02_extract_tiles.py from a real local checkout, evicting the cached
    review_app 'config' module first -- see the module docstring's
    'config' NAME COLLISION section for why this matters."""
    if not pipeline_dir.exists():
        raise SystemExit(f"detection pipeline not found at {pipeline_dir} -- "
                          f"pass --detection-root if it lives somewhere else.")

    sys.modules.pop("config", None)
    sys.path.insert(0, str(pipeline_dir))

    det_config_spec = importlib.util.spec_from_file_location(
        "config", pipeline_dir / "config.py")
    det_config = importlib.util.module_from_spec(det_config_spec)
    sys.modules["config"] = det_config   # so 02_extract_tiles.py's own
                                          # `import config as C` reuses THIS
    det_config_spec.loader.exec_module(det_config)

    naip_spec = importlib.util.spec_from_file_location(
        "naip_fetch", pipeline_dir / "naip_fetch.py")
    naip_fetch = importlib.util.module_from_spec(naip_spec)
    sys.modules["naip_fetch"] = naip_fetch
    naip_spec.loader.exec_module(naip_fetch)

    extract_spec = importlib.util.spec_from_file_location(
        "extract02", pipeline_dir / "02_extract_tiles.py")
    extract02 = importlib.util.module_from_spec(extract_spec)
    extract_spec.loader.exec_module(extract02)   # __name__ != "__main__",
                                                   # so its own main() never runs

    return det_config, naip_fetch, extract02


# ===========================================================================
# Load reviewed plants + candidates from review_app's own app.db
# ===========================================================================
def load_reviewed_sites(reported_only: bool, candidates_only: bool) -> pd.DataFrame:
    conn = sqlite3.connect(C.APP_DB_PATH)
    plants = pd.read_sql_query(
        "SELECT cwns_id, state_code, reported_ll_uuid "
        "FROM plants WHERE reviewed = 1", conn)
    print(f"{len(plants)} reviewed plant(s) in {C.APP_DB_PATH.name}")

    rows = []
    if not candidates_only:
        rep = plants[plants["reported_ll_uuid"].notna()
                     & (plants["reported_ll_uuid"] != "")]
        for _, r in rep.iterrows():
            rows.append(dict(cwns_id=r["cwns_id"], st=r["state_code"],
                             ll_uuid=r["reported_ll_uuid"], role="reported"))
        print(f"  {len(rep)} reported-location parcel(s)")

    if not reported_only:
        cwns_ids = plants["cwns_id"].tolist()
        if cwns_ids:
            placeholders = ",".join("?" * len(cwns_ids))
            cands = pd.read_sql_query(
                f"SELECT cwns_id, candidate_rank, ll_uuid FROM candidates "
                f"WHERE cwns_id IN ({placeholders}) AND ll_uuid IS NOT NULL "
                f"AND ll_uuid != ''", conn, params=cwns_ids)
        else:
            cands = pd.DataFrame(columns=["cwns_id", "candidate_rank", "ll_uuid"])
        st_by_cwns = plants.set_index("cwns_id")["state_code"]
        for _, r in cands.iterrows():
            st = st_by_cwns.get(r["cwns_id"])
            if st is None:
                continue
            rows.append(dict(cwns_id=r["cwns_id"], st=st, ll_uuid=r["ll_uuid"],
                             role=f"candidate_rank{int(r['candidate_rank'])}"))
        print(f"  {len(cands)} candidate parcel row(s) across all ranks")

    conn.close()
    if not rows:
        return pd.DataFrame(columns=["cwns_id", "st", "ll_uuid", "role"])

    df = pd.DataFrame(rows)
    df["cwns_id"] = df["cwns_id"].astype(str)
    df["ll_uuid"] = df["ll_uuid"].astype(str)

    def pick_role(roles):
        return "reported" if "reported" in roles.values else roles.iloc[0]

    dedup = (df.groupby(["cwns_id", "ll_uuid"], as_index=False)
               .agg(st=("st", "first"), role=("role", pick_role)))
    n_dupe = len(df) - len(dedup)
    if n_dupe:
        print(f"  {n_dupe} row(s) were both reported and a candidate for the "
              f"same parcel -- collapsed to one site each")
    print(f"  {len(dedup)} distinct (plant, parcel) site(s) to tile")
    return dedup


def load_targeted_sites(point_halfwidth_m: float,
                        review_round: int | None = None) -> pd.DataFrame:
    """Sites worth labelling after a review round, rather than every parcel
    the reviewer saw.

    Two categories, and nothing else:

      VERIFIED TRUE LOCATION -- wherever the reviewer established the plant
      actually is. Three ways that happens:
          candidate_correct         the selected candidate parcel
          reported_correct          the reported parcel
          truth_outside_candidates  a clicked lat/lon with no parcel at all,
                                    tiled from a synthetic square (see below)
      These are the positive examples. Worth labelling whether or not the
      detector fired: if it did, the label confirms it; if it did not, the
      label is a miss it needs to learn from.

      DETECTION ON A PARCEL THAT WAS NOT THE ANSWER -- a candidate where the
      detector found infrastructure and the reviewer said this is not the
      plant. A false positive, and the most informative thing a round
      produces: a specific, confirmed case of the model seeing treatment
      infrastructure where there is none.

    DELIBERATELY EXCLUDED: candidates with no detection that were not
    selected. The detector and the reviewer already agree there is nothing
    there, so labelling one confirms what the model knows. That category is
    the overwhelming bulk of what the old "tile every candidate" behaviour
    produced -- measured 2026-09-22, 872 of 1,004 existing labels are empty
    -- and dropping it is the whole point of this function.

    ALSO EXCLUDED: needs_info plants entirely. Without a verdict there is no
    ground truth, so a detection on one of their candidates cannot be called
    a false positive. It might be the plant.
    """
    conn = sqlite3.connect(C.APP_DB_PATH)
    plants = pd.read_sql_query(
        "SELECT cwns_id, state_code, reported_ll_uuid, plant_verdict, "
        "       selected_ll_uuid, truth_latitude, truth_longitude "
        "FROM plants WHERE reviewed = 1"
        + (" AND review_round = ?" if review_round is not None else ""),
        conn, params=[review_round] if review_round is not None else None)
    cands = pd.read_sql_query(
        "SELECT cwns_id, candidate_rank, ll_uuid, od_ran, od_has_detection, "
        "       od_n_objects, od_max_confidence FROM candidates "
        "WHERE ll_uuid IS NOT NULL AND ll_uuid != ''", conn)
    conn.close()

    plants["cwns_id"] = plants["cwns_id"].astype(str)
    cands["cwns_id"] = cands["cwns_id"].astype(str)
    if review_round is not None:
        cands = cands[cands["cwns_id"].isin(set(plants["cwns_id"]))]
        print(f"  --round {review_round}: restricted to that round's plants")
    print(f"{len(plants)} reviewed plant(s) in {C.APP_DB_PATH.name}")

    rows, n_point, truth_parcel = [], 0, {}
    for _, p in plants.iterrows():
        v, cw, st = p["plant_verdict"], p["cwns_id"], p["state_code"]
        if v == "candidate_correct" and p["selected_ll_uuid"]:
            truth_parcel[cw] = str(p["selected_ll_uuid"])
            rows.append(dict(cwns_id=cw, st=st, ll_uuid=str(p["selected_ll_uuid"]),
                             role="verified_true", lat=None, lon=None))
        elif v == "reported_correct" and p["reported_ll_uuid"]:
            truth_parcel[cw] = str(p["reported_ll_uuid"])
            rows.append(dict(cwns_id=cw, st=st, ll_uuid=str(p["reported_ll_uuid"]),
                             role="verified_true", lat=None, lon=None))
        elif v == "truth_outside_candidates":
            if pd.isna(p["truth_latitude"]) or pd.isna(p["truth_longitude"]):
                print(f"  WARNING: {cw} is truth_outside_candidates with no "
                      f"clicked point -- skipped")
                continue
            # No parcel exists for a clicked point, so tile a synthetic square
            # centred on it. Same fallback shape 01b uses when it will not
            # trust a parcel boundary (CAPPED_PARCEL_FALLBACK_HALFWIDTH_M).
            rows.append(dict(cwns_id=cw, st=st, ll_uuid="",
                             role="verified_true_point",
                             lat=float(p["truth_latitude"]),
                             lon=float(p["truth_longitude"])))
            n_point += 1

    verdicts = plants.set_index("cwns_id")["plant_verdict"].to_dict()
    st_by_cwns = plants.set_index("cwns_id")["state_code"].to_dict()

    n_fp = 0
    for _, c in cands.iterrows():
        cw = c["cwns_id"]
        if verdicts.get(cw) in (None, "needs_info"):
            continue
        if not bool(c["od_has_detection"]):
            continue
        if truth_parcel.get(cw) == str(c["ll_uuid"]):
            continue                      # it fired on the right parcel: a TP,
                                          # already captured as verified_true
        st = st_by_cwns.get(cw)
        if st is None:
            continue
        rows.append(dict(cwns_id=cw, st=st, ll_uuid=str(c["ll_uuid"]),
                         role="detection_not_selected", lat=None, lon=None))
        n_fp += 1

    n_true = sum(1 for r in rows if r["role"].startswith("verified_true"))
    print(f"  {n_true} verified true location(s) ({n_point} from a clicked point)")
    print(f"  {n_fp} detection(s) on a parcel that was not the answer")

    if cands["od_ran"].isna().all() and len(cands):
        print("  WARNING: no candidate carries detection results. 01e had not run "
              "when this queue was built, so no false positives can be identified "
              "-- only verified true locations will be tiled.")

    if not rows:
        return pd.DataFrame(columns=["cwns_id", "st", "ll_uuid", "role", "lat", "lon"])

    df = pd.DataFrame(rows)
    df["point_halfwidth_m"] = point_halfwidth_m
    dedup = df.drop_duplicates(subset=["cwns_id", "ll_uuid", "role"])
    dedup = dedup.sort_values("role").drop_duplicates(subset=["cwns_id", "ll_uuid"],
                                                      keep="first")
    print(f"  {len(dedup)} distinct site(s) to tile "
          f"(was {len(cands) + len(plants)} under tile-every-candidate)")
    return dedup.reset_index(drop=True)


def resolve_geometries(sites: pd.DataFrame, projected_crs) -> gpd.GeoDataFrame:
    """backend.parcels.get_parcel_context() -- same live lookup the review
    app itself uses, so what gets tiled matches what the reviewer saw.
    Reprojects to detection's PROJECTED_CRS to feed generate_tile_grid()
    directly."""
    sites = sites.copy()
    has_points = "lat" in sites.columns and sites["lat"].notna().any()
    parcel_sites = sites[sites["ll_uuid"].astype(bool)] if "lat" in sites.columns else sites

    geoms = {}
    for state, grp in parcel_sites.groupby("st"):
        ctx = parcels.get_parcel_context(state, grp["ll_uuid"].unique().tolist())
        n_found = 0
        for uuid, entry in ctx.items():
            if entry.get("geometry"):
                geoms[(state, uuid)] = shape(entry["geometry"])
                n_found += 1
        print(f"  {state}: resolved {n_found}/{grp['ll_uuid'].nunique()} parcel(s)")

    def geom_for(r):
        # A clicked point has no parcel to look up. Represent it as the bare
        # Point here and square it off after reprojection, where a half-width
        # in metres actually means something -- buffering in degrees would
        # give a different real-world size at every latitude.
        if getattr(r, "lat", None) is not None and not pd.isna(getattr(r, "lat", None)):
            return Point(r.lon, r.lat)
        return geoms.get((r.st, r.ll_uuid))

    sites["geometry"] = [geom_for(r) for r in sites.itertuples()]
    n_missing = int(sites["geometry"].isna().sum())
    if n_missing:
        print(f"  {n_missing} site(s) had no matching parcel in the local "
              f"Regrid mirror -- skipped (not centered from lat/lon: a tile "
              f"centered off-parcel costs review time for nothing, same "
              f"reasoning 01_sample_sites.py applies to review_fp rows)")
        sites = sites.dropna(subset=["geometry"])

    gdf = gpd.GeoDataFrame(sites, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.to_crs(projected_crs)

    if has_points:
        hw = float(gdf["point_halfwidth_m"].iloc[0]) if "point_halfwidth_m" in gdf.columns else 250.0
        is_pt = gdf.geometry.geom_type == "Point"
        if is_pt.any():
            # cap_style=3 -> a square, not a circle: the tile grid is square,
            # and a round buffer just adds corners that tile to nothing.
            gdf.loc[is_pt, "geometry"] = gdf.loc[is_pt, "geometry"].buffer(hw, cap_style=3)
            print(f"  {int(is_pt.sum())} clicked point(s) squared off to "
                  f"{hw * 2:.0f}m boxes for tiling")
    return gdf


# ===========================================================================
# Overlapping-parcel dedup (added 2026-09-16)
# ===========================================================================
def dedupe_overlapping_sites(gdf: gpd.GeoDataFrame, overlap_frac: float = 0.3) -> gpd.GeoDataFrame:
    """For the same cwns_id, two different ll_uuid candidates/reported
    parcels can be separate Regrid records for what is physically the same
    lagoon/plant complex -- adjoining tracts, split lots under different
    deed records, etc. Confirmed 2026-09-16: two 'different' candidate
    parcels for one CWNS_ID produced near-identical tiles (same two
    lagoons, offset by about one tile) because their parcel polygons
    substantially overlap. Tiling both wastes fetch/labeling effort and
    risks the same real-world scene landing in both a train and a val
    split as if they were independent examples.

    Merges any two sites for the same cwns_id whose geometries overlap by
    more than overlap_frac of the smaller one's area, keeping one
    representative row and combining their roles (e.g. a parcel that's
    both a rank-2 candidate AND overlaps the reported parcel keeps the
    more informative 'reported' label -- same precedence as the
    (cwns_id, ll_uuid) dedup in load_reviewed_sites).
    """
    gdf = gdf.reset_index(drop=True)
    keep_rows = []
    used = set()
    n_dropped = 0

    for cwns_id, grp in gdf.groupby("cwns_id"):
        idxs = list(grp.index)
        for i in idxs:
            if i in used:
                continue
            base = gdf.loc[i]
            roles = {base["role"]}
            used.add(i)
            for j in idxs:
                if j in used:
                    continue
                other = gdf.loc[j]
                if not base["geometry"].intersects(other["geometry"]):
                    continue
                inter_area = base["geometry"].intersection(other["geometry"]).area
                smaller_area = min(base["geometry"].area, other["geometry"].area)
                if smaller_area > 0 and inter_area / smaller_area > overlap_frac:
                    roles.add(other["role"])
                    used.add(j)
                    n_dropped += 1
                    print(f"    {cwns_id}: {other['ll_uuid']} ({other['role']}) "
                          f"overlaps {base['ll_uuid']} ({base['role']}) by "
                          f"{inter_area / smaller_area:.0%} of the smaller parcel "
                          f"-- merged, same physical site")
            row = base.copy()
            row["role"] = "reported" if "reported" in roles else sorted(roles)[0]
            keep_rows.append(row)

    if n_dropped:
        print(f"  {n_dropped} parcel(s) collapsed into an overlapping sibling "
              f"parcel for the same plant -- avoided near-duplicate tiles")
    return gpd.GeoDataFrame(keep_rows, geometry="geometry", crs=gdf.crs).reset_index(drop=True)


# ===========================================================================
# Label coverage (--check-labels, informational only)
# ===========================================================================
def load_labeled_stems(labels_dir: Path) -> set:
    if not labels_dir.exists():
        print(f"  (labels dir not found at {labels_dir} -- skipping label check)")
        return set()
    return {p.stem for p in labels_dir.glob("*.txt")}


def report_label_coverage(sites: gpd.GeoDataFrame, labeled_stems: set):
    if not labeled_stems:
        return
    print("\n--- label coverage (informational, does not affect fetching) ---")
    n_any = 0
    for cwns_id, grp in sites.groupby("cwns_id"):
        parcels_labeled = [
            ll for ll in grp["ll_uuid"]
            if any(f"{cwns_id}_{ll}" in stem for stem in labeled_stems)
        ]
        if parcels_labeled:
            n_any += 1
        else:
            print(f"  {cwns_id}: none of its {len(grp)} reviewed parcel(s) "
                  f"labeled yet")
    print(f"  {n_any}/{sites['cwns_id'].nunique()} reviewed plant(s) have at "
          f"least one already-labeled parcel")


# ===========================================================================
# Build tile tasks -- mirrors 02_extract_tiles.py's build_parcel_tasks inner
# loop, minus the per-county load_parcels() step since resolve_geometries()
# already produced one geometry per site.
# ===========================================================================
def build_tasks(gdf: gpd.GeoDataFrame, existing_ids: set, extract, catalog,
                dry_run: bool) -> tuple[list, dict]:
    tasks = []
    stats = {"single": 0, "ring": 0, "no_item": 0, "lookup_failed": 0}
    print(f"\nBuilding tile grid for {len(gdf)} site(s)...")

    for i, site in enumerate(gdf.itertuples(), start=1):
        grid = extract.generate_tile_grid(site.geometry)
        # Keep only ring tiles that actually overlap the parcel.
        #
        # generate_tile_grid returns the full 3x3 ring (1.5 km x 1.5 km) as
        # soon as a parcel overruns the single centred 500 m tile by any
        # amount -- which most treatment parcels do. For a parcel that pokes
        # out on one side, most of the ring is neighbouring land with nothing
        # in it. On 2026-09-25 that turned ~325 review sites into several
        # thousand tiles. The centre tile (row 2, col 2) is always kept: it is
        # what resolve_item() fetches the NAIP item for, and it is the one
        # tile guaranteed to hold the parcel's centroid.
        n_ring = len(grid)
        grid = [t for t in grid
                if (t["row"] == 2 and t["col"] == 2)
                or box(t["x_min"], t["y_min"], t["x_max"], t["y_max"]).intersects(site.geometry)]
        stats["ring_tiles_dropped"] = stats.get("ring_tiles_dropped", 0) + (n_ring - len(grid))
        stats["ring" if len(grid) > 1 else "single"] += 1
        bboxes, centers = extract.tiles_to_wgs84(grid)

        # Centroid tile is always tagged row=2,col=2 by generate_tile_grid,
        # whether or not the ring of 8 exists.
        ctr_idx = next(k for k, t in enumerate(grid)
                       if t["row"] == 2 and t["col"] == 2)

        if dry_run:
            item_url, acq = "", ""
        else:
            try:
                item_url, acq = extract.resolve_item(catalog, bboxes[ctr_idx])
            except Exception as e:
                # A site-level lookup failure (all of with_retry's attempts
                # exhausted) shouldn't take down planning for every other
                # site in a 1000+ site run -- log it and move on, rather
                # than crashing and losing all the planning work already
                # done for sites processed before this one.
                print(f"    {site.cwns_id}/{site.ll_uuid}: item lookup FAILED "
                      f"({type(e).__name__}: {e}) -- skipping this site")
                stats["lookup_failed"] += 1
                continue
            if item_url is None:
                print(f"    {site.cwns_id}/{site.ll_uuid}: no NAIP item -- skipping")
                stats["no_item"] += 1
                continue

        source = "review_reported" if site.role == "reported" else "review_candidate"
        for t, bbox, ctr in zip(grid, bboxes, centers):
            tid = f"{site.cwns_id}_{site.ll_uuid}_r{t['row']:02d}_c{t['col']:02d}"
            if tid in existing_ids or extract.existing_rgb(tid):
                continue
            tasks.append(dict(
                tile_id=tid, bbox=bbox, ctr=ctr, item_url=item_url,
                catalog=catalog, acq_date=acq, source=source,
                cwns_id=site.cwns_id, tri_id="", ll_uuid=site.ll_uuid,
                alt=site.role, st=site.st, geoid="",
                row=t["row"], col=t["col"]))
            existing_ids.add(tid)

        if i % 25 == 0 or i == len(gdf):
            print(f"  [{i}/{len(gdf)}]")

    return tasks, stats


def prune_ring_tiles(DC, dry_run: bool) -> None:
    """Move UNLABELLED review tiles that do not touch their own parcel out of
    the inventory. The cleanup for runs made before build_tasks() filtered
    the 3x3 ring (2026-09-25).

    Only ever touches rows that are ALL of:
      - source review_candidate / review_reported (this script's output --
        never 02_extract_tiles.py's own sample tiles)
      - a ring tile, not the centre tile (row 2, col 2)
      - on a real parcel (clicked-point sites have no parcel to test against)
      - with no label file in any labels*/ folder -- labelled work is never
        moved, even if the tile turns out to be off-parcel
      - whose parcel geometry was found; a tile that cannot be checked stays

    Moves the PNG and NDWI into tiles/_pruned/, writes a manifest of every
    file moved, backs up tile_metadata.csv and drops the moved rows from it.
    Nothing is deleted: moving the files back and restoring the backup
    undoes it completely.
    """
    import shutil
    from datetime import datetime

    meta_path = Path(DC.METADATA_CSV)
    if not meta_path.exists():
        print(f"No {meta_path} -- nothing to prune.")
        return
    meta = pd.read_csv(meta_path, dtype=str)
    is_review = meta["source"].isin(["review_candidate", "review_reported"])
    is_ring = ~((meta["tile_row"].astype(str).str.lstrip("0") == "2")
                & (meta["tile_col"].astype(str).str.lstrip("0") == "2"))
    has_parcel = meta["ll_uuid_primary"].fillna("").astype(str).str.len() > 0
    cand = meta[is_review & is_ring & has_parcel].copy()
    print(f"{len(meta)} tile(s) in metadata; {int(is_review.sum())} from review; "
          f"{len(cand)} review ring tile(s) on a real parcel")

    labelled = set()
    for d in Path(DC.ANNOTATION_DIR).glob("labels*"):
        if d.is_dir():
            labelled |= {p.stem for p in d.glob("*.txt")}
    cand = cand[~(cand["tile_id"] + "_rgb").isin(labelled)]
    print(f"  {len(cand)} of them unlabelled ({len(labelled)} label file(s) found)")
    if cand.empty:
        return

    drop_ids, unchecked = [], 0
    for st, grp in cand.groupby("st"):
        ctx = parcels.get_parcel_context(st, grp["ll_uuid_primary"].unique().tolist())
        geoms = {u: shape(e["geometry"]) for u, e in ctx.items() if e.get("geometry")}
        for r in grp.itertuples():
            g = geoms.get(r.ll_uuid_primary)
            if g is None:
                unchecked += 1
                continue
            tile = box(float(r.bbox_xmin), float(r.bbox_ymin),
                       float(r.bbox_xmax), float(r.bbox_ymax))
            if not tile.intersects(g):
                drop_ids.append(r.tile_id)
    print(f"  {len(drop_ids)} do not touch their parcel -> prune"
          + (f"  ({unchecked} could not be checked -- parcel not in the local "
             f"Regrid mirror -- and are kept)" if unchecked else ""))
    if not drop_ids:
        return
    if dry_run:
        print("--dry-run: nothing moved.")
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = Path(DC.TILES_DIR) / "_pruned" / stamp
    (dest / "rgb").mkdir(parents=True, exist_ok=True)
    (dest / "ndwi").mkdir(parents=True, exist_ok=True)
    moved = []
    rows = meta.set_index("tile_id").loc[drop_ids]
    for tid, r in rows.iterrows():
        for col, sub, fallback in (("rgb_path", "rgb", Path(DC.RGB_DIR) / f"{tid}_rgb.png"),
                                   ("ndwi_path", "ndwi", Path(DC.NDWI_DIR) / f"{tid}_ndwi.tif")):
            src = Path(r[col]) if isinstance(r[col], str) and r[col] else fallback
            if not src.exists():
                src = fallback
            if src.exists():
                shutil.move(str(src), str(dest / sub / src.name))
                moved.append(dict(tile_id=tid, kind=sub, from_path=str(src),
                                  to_path=str(dest / sub / src.name)))
    pd.DataFrame(moved).to_csv(dest / "manifest.csv", index=False)
    backup = meta_path.with_name(f"{meta_path.stem}.before-prune-{stamp}.csv")
    shutil.copy2(meta_path, backup)
    meta[~meta["tile_id"].isin(set(drop_ids))].to_csv(meta_path, index=False)
    print(f"Moved {len(moved)} file(s) for {len(drop_ids)} tile(s) to {dest}")
    print(f"  manifest : {dest / 'manifest.csv'}")
    print(f"  metadata : {len(meta)} -> {len(meta) - len(drop_ids)} rows "
          f"(backup {backup.name})")
    print("Restart label_app.R to see the smaller inventory.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detection-root", type=Path,
                    default=DETECTION_PIPELINE_DIR_DEFAULT,
                    help=f"path to detection/pipeline/ "
                         f"(default: {DETECTION_PIPELINE_DIR_DEFAULT})")
    ap.add_argument("--all-candidates", action="store_true",
                    help="LEGACY: tile the reported parcel plus every candidate "
                         "shown, regardless of verdict or detections. This is "
                         "what produced ~3,000 tiles per round, most of them "
                         "empty. The default is targeted selection -- see "
                         "load_targeted_sites().")
    ap.add_argument("--point-halfwidth-m", type=float, default=250.0,
                    help="half-width of the synthetic square tiled around a "
                         "truth_outside_candidates click, which has no parcel. "
                         "250m -> a 500m box, matching one tile's footprint.")
    ap.add_argument("--reported-only", action="store_true",
                    help="tile only reported-location parcels, skip candidates")
    ap.add_argument("--candidates-only", action="store_true",
                    help="tile only candidate parcels, skip reported locations")
    ap.add_argument("--overlap-frac", type=float, default=0.3,
                    help="collapse two same-plant parcels into one site when "
                         "their geometries overlap by more than this fraction "
                         "of the smaller parcel's area (default: 0.3)")
    ap.add_argument("--check-labels", action="store_true",
                    help="report which reviewed plants already have a "
                         "labeled parcel (informational, doesn't skip fetching)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of NEW tiles fetched, for a test run")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the tile count without fetching anything")
    ap.add_argument("--round", type=int, default=None,
                    help="only tile plants from this review round. Default: "
                         "every reviewed round (tiles that already exist are "
                         "skipped either way).")
    ap.add_argument("--prune", action="store_true",
                    help="instead of fetching, move UNLABELLED review ring "
                         "tiles that do not touch their parcel into "
                         "tiles/_pruned/ (reversible; see prune_ring_tiles). "
                         "Combine with --dry-run to see the count first.")
    args = ap.parse_args()
    if args.reported_only and args.candidates_only:
        raise SystemExit("--reported-only and --candidates-only are mutually exclusive")

    print("=== extract_review_tiles.py ===")

    DC, NF, extract = load_detection_modules(args.detection_root)
    DC.ensure_dirs()
    print(f"Writing into: {DC.RGB_DIR}")
    print(f"          and {DC.NDWI_DIR}")
    print(f"Metadata:     {DC.METADATA_CSV}")

    if args.prune:
        prune_ring_tiles(DC, args.dry_run)
        return

    if args.all_candidates:
        sites = load_reviewed_sites(args.reported_only, args.candidates_only)
    else:
        if args.reported_only or args.candidates_only:
            raise SystemExit(
                "--reported-only / --candidates-only only apply to "
                "--all-candidates. Targeted selection chooses by verdict and "
                "detection, not by which list a parcel came from.")
        sites = load_targeted_sites(args.point_halfwidth_m, args.round)
    if sites.empty:
        print("Nothing to tile.")
        return

    gdf = resolve_geometries(sites, DC.PROJECTED_CRS)
    if gdf.empty:
        print("No parcels resolved against the local Regrid mirror.")
        return

    gdf = dedupe_overlapping_sites(gdf, overlap_frac=args.overlap_frac)

    if args.check_labels:
        labeled_stems = load_labeled_stems(DC.ANNOTATION_DIR / "labels")
        report_label_coverage(gdf, labeled_stems)

    existing = extract.load_existing_tile_ids()
    print(f"\n{len(existing)} tiles already in metadata (will skip)")

    catalog = None if args.dry_run else NF.open_catalog()
    tasks, stats = build_tasks(gdf, existing, extract, catalog, args.dry_run)

    print("\n--- tile plan ---")
    print(f"  {stats['single']} single-tile site(s), {stats['ring']} multi-tile site(s)")
    print(f"  {stats.get('ring_tiles_dropped', 0)} ring tile(s) skipped for not "
          f"touching their parcel")
    if stats["no_item"]:
        print(f"  {stats['no_item']} site(s) had no NAIP item -- skipped")
    if stats["lookup_failed"]:
        print(f"  {stats['lookup_failed']} site(s) failed item lookup after "
              f"retries -- skipped (re-run later to retry just these)")
    print(f"  TOTAL NEW TILES: {len(tasks)}")

    if args.limit and len(tasks) > args.limit:
        print(f"  --limit {args.limit}: dropping {len(tasks) - args.limit} tile(s)")
        tasks = tasks[:args.limit]

    if args.dry_run:
        print("\n--dry-run: nothing fetched.")
        return

    print(f"\nFetching with {DC.MAX_WORKERS} workers...\n")
    with ThreadPoolExecutor(max_workers=DC.MAX_WORKERS) as executor:
        rows = extract.run_tasks(tasks, executor)
        for _ in range(DC.MAX_WORKERS):
            executor.submit(NF.close_thread_datasets)
    extract.append_metadata(rows)

    failed = len(tasks) - len(rows)
    print(f"\nWrote {len(rows)} tile(s)" + (f", {failed} failed" if failed else ""))
    print(f"Metadata now indexes {len(extract.load_existing_tile_ids())} tiles: "
          f"{DC.METADATA_CSV}")
    print("\nNext: label the new tiles in label_app.R -- oxidation_pond is "
          "already in config.py's CLASSES -- then 03_prepare_dataset.py.")


if __name__ == "__main__":
    main()
