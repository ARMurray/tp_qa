"""
build_test_bundle.py
=====================
Assembles a self-contained, downloadable copy of the correction pipeline --
every script, plus a geographically-concentrated subset of every input it
reads -- into /work/GRDVULN/correction/testing/tpqa_test/.

Purpose: let the whole pipeline be run and debugged on a normal machine with
a fast edit/run loop, instead of one sbatch per hypothesis. The bundle mirrors
the HPC directory layout exactly, so scripts run unchanged apart from four
paths in config.py (see the generated MANIFEST.md).

WHAT THIS BUNDLE IS AND IS NOT
-------------------------------
It is a FIXTURE for finding schema/type/join bugs. Every stage should run to
completion against it.

It is NOT a training set. Any model trained on ~25 plants is meaningless as a
model -- a fixture-trained .joblib must never be treated as a real one, and
must never be copied back to HPC. Real training stays on HPC against the full
data. The bundle exists so that when 03/04/06/07 crash, they crash in two
seconds instead of two hours.

PLANT SELECTION
---------------
Deliberately mixed so every downstream stage has something to chew on:
  - corrections-bin plants (Stage 2/2b positives -- without these, 04/06/07
    have no positive labels at all and can't run)
  - classes-bin plants, both Correct and Incorrect (Stage 1's two classes)
  - holdout-manifest members (so the anti-joins in 03/04/06/07 actually
    fire and can be observed, rather than silently no-op'ing)
  - unlabeled full-universe plants (what inference actually runs against)

Then concentrated into the fewest counties possible, because the Regrid
parcel store is the size driver and it's stored per-county.

Usage:
    python build_test_bundle.py --state OH
    python build_test_bundle.py --state OH --n-plants 40 --n-counties 3
    python build_test_bundle.py --state OH --no-raster    # skip NLCD clip
"""
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

TESTING_ROOT = C.ROOT / "testing"
BUNDLE = TESTING_ROOT / "tpqa_test"

# Candidate parcels for a plant can sit in a neighbouring county, so the
# county set has to cover the k-ring radius around every selected plant, not
# just the counties the plants themselves land in. K_RINGS=18 measures to
# ~5.4km (see config.py); 10km of buffer is comfortable headroom.
COUNTY_BUFFER_M = 10_000


def log(msg=""):
    print(msg, flush=True)


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def copy_file(src: Path, dst_dir: Path, label: str = "") -> bool:
    """Copy one file, creating the destination dir. Returns success. Missing
    sources warn rather than raise -- a partial bundle that reports what's
    missing is more useful than no bundle."""
    if not src.exists():
        log(f"    MISSING (skipped): {src}")
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / src.name)
    size = src.stat().st_size / 1e6
    log(f"    {label or src.name}: {size:.1f} MB")
    return True


def copy_tree(src: Path, dst: Path, pattern: str = "*") -> int:
    if not src.exists():
        log(f"    MISSING (skipped): {src}")
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src.rglob(pattern):
        if f.is_file():
            rel = f.relative_to(src)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
            n += 1
    return n


# ===========================================================================
# 1. Plant selection
# ===========================================================================
def select_plants(state: str, n_plants: int, n_counties: int) -> pd.DataFrame:
    log("\n--- Selecting plants ---")

    # Full universe for the state (what inference runs against)
    facility_types = pd.read_csv(C.CWNS_DIR / "FACILITY_TYPES.txt", dtype=str, encoding="latin1")
    treatment_ids = set(
        facility_types.loc[facility_types["FACILITY_TYPE"] == "Treatment Plant", "CWNS_ID"])
    loc = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt", dtype={"CWNS_ID": str}, encoding="latin1")
    loc = loc[loc["CWNS_ID"].isin(treatment_ids)]
    loc["LATITUDE"] = pd.to_numeric(loc["LATITUDE"], errors="coerce")
    loc["LONGITUDE"] = pd.to_numeric(loc["LONGITUDE"], errors="coerce")
    loc = loc.dropna(subset=["LATITUDE", "LONGITUDE"]).drop_duplicates(subset="CWNS_ID")
    loc = loc[loc["STATE_CODE"] == state][
        ["CWNS_ID", "STATE_CODE", "LATITUDE", "LONGITUDE"]].reset_index(drop=True)
    log(f"  Full universe in {state}: {len(loc)} plants")

    # Label bins
    corrections_ids, classes_ids, holdout_ids = set(), set(), set()
    if C.TRAINING_GPKG.exists():
        corr = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CORRECTIONS)
        corrections_ids = set(corr["CWNS_ID"].astype(str))
        cls = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CLASSES)
        classes_ids = set(cls["CWNS_ID"].astype(str))
        log(f"  Training universe: {len(classes_ids)} classes, {len(corrections_ids)} corrections")
    else:
        log(f"  WARNING: {C.TRAINING_GPKG} missing -- bundle will have no labels, "
            f"and 03/04/06/07 will not be runnable against it")

    manifest_path = C.DATA_DIR / "holdout" / "holdout_manifest.parquet"
    if manifest_path.exists():
        holdout_ids = set(pd.read_parquet(manifest_path)["CWNS_ID"].astype(str))
        log(f"  Holdout manifest: {len(holdout_ids)} plants")

    loc["in_corrections"] = loc["CWNS_ID"].isin(corrections_ids)
    loc["in_classes"] = loc["CWNS_ID"].isin(classes_ids)
    loc["in_holdout"] = loc["CWNS_ID"].isin(holdout_ids)

    # Attach county so selection can be concentrated geographically
    counties = load_counties()
    pts = gpd.GeoDataFrame(
        loc, geometry=gpd.points_from_xy(loc["LONGITUDE"], loc["LATITUDE"]), crs=4326
    ).to_crs(counties.crs)
    joined = gpd.sjoin(pts, counties[["GEOID", "geometry"]], how="left", predicate="intersects")
    joined = joined.drop(columns=["index_right"]).rename(columns={"GEOID": "geoid"})
    joined = joined.drop_duplicates(subset="CWNS_ID")

    # Rank counties by how much labelled material they contain -- a county with
    # corrections in it is worth far more than one with only unlabelled plants,
    # because corrections are the only source of Stage 2 positive labels.
    score = (joined.groupby("geoid")
             .agg(n_corrections=("in_corrections", "sum"),
                  n_classes=("in_classes", "sum"),
                  n_total=("CWNS_ID", "size"))
             .reset_index())
    score["score"] = score["n_corrections"] * 10 + score["n_classes"] * 2 + score["n_total"]
    score = score.sort_values("score", ascending=False)
    log(f"\n  Top counties by label content:")
    log(score.head(8).to_string(index=False))

    chosen_geoids = score.head(n_counties)["geoid"].tolist()
    pool = joined[joined["geoid"].isin(chosen_geoids)].copy()
    log(f"\n  Chose {len(chosen_geoids)} county/counties: {chosen_geoids}")
    log(f"  Pool within them: {len(pool)} plants "
        f"({int(pool['in_corrections'].sum())} corrections, "
        f"{int(pool['in_classes'].sum())} classes, "
        f"{int(pool['in_holdout'].sum())} holdout)")

    # Take everything labelled (scarce and valuable), then top up with
    # unlabelled plants to reach n_plants.
    labelled = pool[pool["in_corrections"] | pool["in_classes"] | pool["in_holdout"]]
    unlabelled = pool[~pool.index.isin(labelled.index)]
    n_top_up = max(0, n_plants - len(labelled))
    rng = np.random.default_rng(42)
    if n_top_up and len(unlabelled):
        take = min(n_top_up, len(unlabelled))
        idx = rng.choice(len(unlabelled), size=take, replace=False)
        unlabelled = unlabelled.iloc[idx]
    else:
        unlabelled = unlabelled.iloc[:0]

    selected = pd.concat([labelled, unlabelled], ignore_index=True)
    log(f"\n  SELECTED {len(selected)} plants:")
    log(f"    corrections bin : {int(selected['in_corrections'].sum())}"
        f"   <-- Stage 2/2b positives")
    log(f"    classes bin     : {int(selected['in_classes'].sum())}"
        f"   <-- Stage 1 labels")
    log(f"    holdout members : {int(selected['in_holdout'].sum())}"
        f"   <-- exercises the anti-joins")
    log(f"    unlabelled      : {len(selected) - int((selected['in_corrections'] | selected['in_classes'] | selected['in_holdout']).sum())}"
        f"   <-- inference-only")
    if int(selected["in_corrections"].sum()) == 0:
        log("\n    WARNING: zero corrections-bin plants selected. 04/06/07 have no")
        log("    positive labels and will not train against this bundle. Raise")
        log("    --n-counties, or pick a state with more corrections.")
    return gpd.GeoDataFrame(selected, geometry="geometry", crs=counties.crs)


def load_counties() -> gpd.GeoDataFrame:
    if C.CENSUS_GDB.exists():
        return gpd.read_file(C.CENSUS_GDB, layer="County")[["NAMELSAD", "GEOID", "geometry"]].to_crs(5070)
    raise FileNotFoundError(
        f"{C.CENSUS_GDB} not found -- needed to attach counties to plants. "
        f"Cannot build the bundle without it.")


def counties_needed(selected: gpd.GeoDataFrame) -> list[str]:
    """Every county intersecting a COUNTY_BUFFER_M buffer around the selected
    plants -- candidate parcels within K_RINGS routinely cross county lines,
    and a missing neighbour county silently truncates the candidate pool."""
    counties = load_counties()
    buffered = selected.to_crs(5070).buffer(COUNTY_BUFFER_M).union_all()
    hit = counties[counties.intersects(buffered)]
    geoids = sorted(hit["GEOID"].tolist())
    log(f"\n  Counties needed (plants + {COUNTY_BUFFER_M / 1000:.0f}km buffer): "
        f"{len(geoids)} -> {geoids}")
    return geoids


# ===========================================================================
# 2. Collectors
# ===========================================================================
def collect_scripts():
    log("\n--- Scripts ---")
    n = copy_tree(C.SCRIPTS_DIR, BUNDLE / "scripts")
    log(f"    {n} files ({dir_size_mb(BUNDLE / 'scripts'):.1f} MB)")


def collect_cwns():
    log("\n--- CWNS source tables ---")
    n = copy_tree(C.CWNS_DIR, BUNDLE / "data" / "cwns")
    log(f"    {n} files ({dir_size_mb(BUNDLE / 'data' / 'cwns'):.1f} MB)")


def collect_training():
    log("\n--- Training labels ---")
    copy_file(C.TRAINING_GPKG, BUNDLE / "data" / "training")
    # Updates.gdb is a directory, and only build_training_bins.py reads it.
    if C.MASTER_GDB.exists():
        n = copy_tree(C.MASTER_GDB, BUNDLE / "data" / "training" / C.MASTER_GDB.name)
        log(f"    {C.MASTER_GDB.name}: {n} files "
            f"({dir_size_mb(BUNDLE / 'data' / 'training' / C.MASTER_GDB.name):.1f} MB)")


def collect_parcels(state: str, geoids: list[str]):
    """Whole per-county parquet files, unmodified. Deliberately NOT
    column-trimmed or row-filtered: 01a/01b/02 each SELECT different column
    sets and join against the store in different ways, so a trimmed copy
    would work for some scripts and mysteriously fail for others -- exactly
    the class of bug this bundle exists to eliminate."""
    log("\n--- Regrid parcels (whole county files, unmodified) ---")
    src_dir = C.PARCEL_BASE / f"state={state}"
    dst_dir = BUNDLE / "data" / "parcels" / f"state={state}"
    if not src_dir.exists():
        log(f"    MISSING: {src_dir}")
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied, missing = 0, []
    for geoid in geoids:
        matches = list(src_dir.glob(f"{geoid}*.parquet"))
        if not matches:
            missing.append(geoid)
            continue
        for f in matches:
            shutil.copy2(f, dst_dir / f.name)
            copied += 1
    log(f"    {copied} county file(s) ({dir_size_mb(dst_dir):.1f} MB)")
    if missing:
        log(f"    No parcel file for county geoid(s): {missing} "
            f"(may be genuinely absent from the store)")


def collect_nlcd_features(state: str, selected: pd.DataFrame):
    """01a's output, filtered to parcels within K_RINGS of a selected plant."""
    log("\n--- 01a NLCD features (filtered to selected plants' candidates) ---")
    src = C.NLCD_OUTPUT_DIR / f"nlcd_{state}_k{C.K_RINGS}.parquet"
    if not src.exists():
        log(f"    MISSING: {src}")
        return set()
    nlcd = pd.read_parquet(src)
    sel_wgs = selected.to_crs(4326)
    cells = set()
    for _, r in sel_wgs.iterrows():
        cell = h3.latlng_to_cell(r.geometry.y, r.geometry.x, 9)
        cells |= set(h3.grid_disk(cell, C.K_RINGS))
    keep = nlcd[nlcd["h3_index_9"].isin(cells)]
    dst_dir = BUNDLE / "data" / "nlcd_features"
    dst_dir.mkdir(parents=True, exist_ok=True)
    keep.to_parquet(dst_dir / src.name, index=False)
    log(f"    {len(keep)} of {len(nlcd)} rows kept "
        f"({(dst_dir / src.name).stat().st_size / 1e6:.1f} MB)")
    return set(keep["ll_uuid"])


def collect_od(selected_ids: set):
    log("\n--- 01b / 01c OD output (filtered to selected plants) ---")
    for src_root, label in ((C.OD_OUTPUT_DIR, "od_features"),
                            (C.OD_OUTPUT_DIR_CORRECTED, "od_features_corrected")):
        if not src_root.exists():
            log(f"    {label}: not present")
            continue
        n_rows = 0
        for table_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
            for part in table_dir.rglob("*.parquet"):
                try:
                    df = pd.read_parquet(part)
                except Exception as e:
                    log(f"    could not read {part.name}: {e}")
                    continue
                if "CWNS_ID" not in df.columns:
                    continue
                keep = df[df["CWNS_ID"].astype(str).isin(selected_ids)]
                if not len(keep):
                    continue
                rel = part.relative_to(src_root)
                out = BUNDLE / "data" / label / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                keep.to_parquet(out, index=False)
                n_rows += len(keep)
        log(f"    {label}: {n_rows} rows kept "
            f"({dir_size_mb(BUNDLE / 'data' / label):.1f} MB)")


def collect_features(selected_ids: set, keep_uuids: set):
    """02's outputs. Plant-keyed tables are filtered to the selected plants;
    parcel-keyed tables to the retained candidate parcels."""
    log("\n--- 02 feature tables ---")
    dst = BUNDLE / "data" / "features"
    dst.mkdir(parents=True, exist_ok=True)

    plant_keyed = ["05_plant_features.parquet", "14_stage1_training.parquet",
                   "15_stage2_training.parquet", "16_stage2b_training.parquet",
                   "17_review_training.parquet"]
    for name in plant_keyed:
        src = C.FEATURES_OUTPUT_DIR / name
        if not src.exists():
            log(f"    {name}: not present")
            continue
        df = pd.read_parquet(src)
        if "CWNS_ID" in df.columns:
            keep = df[df["CWNS_ID"].astype(str).isin(selected_ids)]
        else:
            keep = df
        keep.to_parquet(dst / name, index=False)
        log(f"    {name}: {len(keep)} of {len(df)} rows "
            f"({(dst / name).stat().st_size / 1e6:.1f} MB)")

    for name in ["10_parcel_features.parquet"]:
        src = C.FEATURES_OUTPUT_DIR / name
        if not src.exists():
            log(f"    {name}: not present")
            continue
        df = pd.read_parquet(src)
        keep = df[df["ll_uuid"].isin(keep_uuids)] if keep_uuids else df.iloc[:0]
        keep.to_parquet(dst / name, index=False)
        log(f"    {name}: {len(keep)} of {len(df)} rows "
            f"({(dst / name).stat().st_size / 1e6:.1f} MB)")

    # Per-state cache: deliberately NOT copied. It's a persistent cache keyed
    # by state, and a partial copy is exactly what caused the 5.47x row
    # inflation bug (see TPQA_MASTER_REFERENCE.md S8.2). Let 02 rebuild it.
    log("    10_parcel_features_by_state/: deliberately NOT copied "
        "(cache; 02 rebuilds it -- see MANIFEST.md)")


def collect_models():
    log("\n--- Trained models ---")
    dst = BUNDLE / "models"
    n = 0
    for f in C.MODELS_DIR.glob("*"):
        if f.is_file():
            copy_file(f, dst)
            n += 1
    if C.OD_MODEL_DIR.exists():
        for f in C.OD_MODEL_DIR.glob("*.pt"):
            copy_file(f, dst / "object_detection", f"object_detection/{f.name}")
            n += 1
    log(f"    {n} files ({dir_size_mb(dst):.1f} MB)")


def collect_holdout():
    log("\n--- Holdout ---")
    src = C.DATA_DIR / "holdout"
    if not src.exists():
        log("    not present")
        return
    n = copy_tree(src, BUNDLE / "data" / "holdout")
    log(f"    {n} files ({dir_size_mb(BUNDLE / 'data' / 'holdout'):.1f} MB)")


def collect_diagnostics():
    log("\n--- 08 diagnostics output (input to 01d) ---")
    src = C.DATA_DIR / "diagnostics"
    if not src.exists():
        log("    not present (run 08 first if 01d needs testing)")
        return
    n = copy_tree(src, BUNDLE / "data" / "diagnostics")
    log(f"    {n} files ({dir_size_mb(BUNDLE / 'data' / 'diagnostics'):.1f} MB)")


def collect_reference(state: str, geoids: list[str]):
    """Census layers clipped to the needed counties, OSM clipped to their
    extent. Written as a GeoPackage with the SAME LAYER NAMES 02 reads from
    the .gdb, so only the path changes locally, not the code."""
    log("\n--- Reference layers (clipped) ---")
    dst = BUNDLE / "data" / "reference"
    dst.mkdir(parents=True, exist_ok=True)

    if C.CENSUS_GDB.exists():
        counties = load_counties()
        keep_counties = counties[counties["GEOID"].isin(geoids)]
        extent = keep_counties.to_crs(5070).buffer(COUNTY_BUFFER_M).union_all()
        extent_gs = gpd.GeoSeries([extent], crs=5070)

        out_path = dst / "census_subset.gpkg"
        layers = {
            "County": ["NAMELSAD", "GEOID", "geometry"],
            "County_Subdivision": ["NAMELSAD", "geometry"],
            "Incorporated_Place": ["NAMELSAD", "geometry"],
            "Census_Designated_Place": ["NAMELSAD", "geometry"],
        }
        for layer, cols in layers.items():
            try:
                gdf = gpd.read_file(C.CENSUS_GDB, layer=layer)
            except Exception as e:
                log(f"    census layer {layer}: could not read ({e})")
                continue
            gdf = gdf[[c for c in cols if c in gdf.columns]]
            gdf_5070 = gdf.to_crs(5070)
            keep = gdf[gdf_5070.intersects(extent_gs.iloc[0]).to_numpy()]
            keep.to_file(out_path, layer=layer, driver="GPKG")
            log(f"    census {layer}: {len(keep)} of {len(gdf)} features")
        log(f"    census_subset.gpkg: {out_path.stat().st_size / 1e6:.1f} MB")
    else:
        log(f"    MISSING: {C.CENSUS_GDB}")

    if C.OSM_PATH.exists():
        try:
            osm = gpd.read_file(C.OSM_PATH, layer="Points")
            counties = load_counties()
            keep_counties = counties[counties["GEOID"].isin(geoids)]
            extent = keep_counties.to_crs(5070).buffer(COUNTY_BUFFER_M).union_all()
            osm_5070 = osm.to_crs(5070)
            keep = osm[osm_5070.intersects(extent).to_numpy()]
            out = dst / C.OSM_PATH.name
            keep.to_file(out, layer="Points", driver="GPKG")
            log(f"    {C.OSM_PATH.name}: {len(keep)} of {len(osm)} points "
                f"({out.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            log(f"    OSM clip failed ({e}) -- copying whole file instead")
            copy_file(C.OSM_PATH, dst)
    else:
        log(f"    MISSING: {C.OSM_PATH}")


def collect_nlcd_raster(geoids: list[str]):
    """Windowed clip of the continental NLCD raster to the needed counties,
    so 01a and 01d can actually RUN locally rather than only being tested
    against pre-baked output."""
    log("\n--- NLCD raster (clipped) ---")
    if not C.NLCD_PATH.exists():
        log(f"    MISSING: {C.NLCD_PATH}")
        return
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError:
        log("    rasterio not available -- skipping raster clip")
        return

    counties = load_counties()
    keep = counties[counties["GEOID"].isin(geoids)]
    dst = BUNDLE / "data" / "nlcd"
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / C.NLCD_PATH.name

    with rasterio.open(C.NLCD_PATH) as src:
        bounds = keep.to_crs(src.crs).total_bounds
        pad = COUNTY_BUFFER_M
        win = from_bounds(bounds[0] - pad, bounds[1] - pad,
                          bounds[2] + pad, bounds[3] + pad, src.transform)
        data = src.read(1, window=win)
        profile = src.profile.copy()
        profile.update(height=data.shape[0], width=data.shape[1],
                       transform=src.window_transform(win),
                       compress="lzw")
        with rasterio.open(out, "w", **profile) as dstf:
            dstf.write(data, 1)
    log(f"    {out.name}: {data.shape[1]}x{data.shape[0]} px "
        f"({out.stat().st_size / 1e6:.1f} MB)")


def collect_logs(n_recent: int = 25):
    log(f"\n--- Recent logs (last {n_recent}) ---")
    if not C.LOGS_DIR.exists():
        log("    not present")
        return
    logs = sorted(C.LOGS_DIR.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    dst = BUNDLE / "logs"
    dst.mkdir(parents=True, exist_ok=True)
    for f in logs[:n_recent]:
        shutil.copy2(f, dst / f.name)
    log(f"    {min(len(logs), n_recent)} files ({dir_size_mb(dst):.1f} MB)")


def collect_docs():
    log("\n--- Docs ---")
    dst = BUNDLE / "docs"
    for name in ("TPQA_MASTER_REFERENCE.md", "REVIEW_LOOP_PLAN.md"):
        for base in (C.ROOT, C.SCRIPTS_DIR, C.ROOT / "docs"):
            src = base / name
            if src.exists():
                copy_file(src, dst)
                break
        else:
            log(f"    MISSING: {name} (searched {C.ROOT}, {C.SCRIPTS_DIR}, {C.ROOT}/docs)")


# ===========================================================================
# 3. Manifest
# ===========================================================================
def write_manifest(state, selected, geoids, args):
    log("\n--- Writing MANIFEST.md ---")
    sel = selected.copy()
    n_corr = int(sel["in_corrections"].sum())
    n_cls = int(sel["in_classes"].sum())
    n_hold = int(sel["in_holdout"].sum())

    sel[["CWNS_ID", "STATE_CODE", "LATITUDE", "LONGITUDE", "geoid",
         "in_corrections", "in_classes", "in_holdout"]].to_csv(
        BUNDLE / "selected_plants.csv", index=False)

    text = f"""# tpqa_test — portable pipeline fixture

Generated {datetime.now():%Y-%m-%d %H:%M:%S} by `build_test_bundle.py`
from `{C.ROOT}` on the atmos HPC.

## What this is

A self-contained copy of the wastewater treatment plant location-correction
pipeline — every script, plus a geographically-concentrated subset of every
input it reads — so the whole thing can be run and debugged on a normal
machine with a fast edit/run loop.

**This is a fixture for finding bugs, not a training set.** {len(sel)} plants
is nowhere near enough to train a meaningful model. Any `.joblib` produced
here is a test artifact: it must never be treated as a real model and must
never be copied back to the HPC. Real training stays on HPC against full data.

## Contents

- **State:** {state}
- **Plants:** {len(sel)} ({n_corr} corrections-bin, {n_cls} classes-bin,
  {n_hold} holdout members, {len(sel) - int((sel['in_corrections'] | sel['in_classes'] | sel['in_holdout']).sum())} unlabelled)
- **Counties:** {len(geoids)} — `{', '.join(geoids)}`
- **Full plant list:** see `selected_plants.csv`

Plants were chosen to exercise every stage: corrections-bin plants supply the
only Stage 2/2b positive labels, classes-bin plants supply Stage 1's two
classes, holdout members make the anti-joins in 03/04/06/07 actually fire
rather than silently no-op, and unlabelled plants are what inference runs on.

## Directory layout

Mirrors the HPC layout exactly, so scripts run unchanged apart from paths:

```
tpqa_test/
  scripts/                  every pipeline script, verbatim
  models/                   trained .joblib models + object_detection/best.pt
  logs/                     recent HPC logs, for context
  docs/                     TPQA_MASTER_REFERENCE.md, REVIEW_LOOP_PLAN.md
  selected_plants.csv       exactly which plants are in this bundle
  data/
    cwns/                   CWNS source tables (whole, unmodified)
    training/               training_locations.gpkg (+ Updates.gdb if present)
    parcels/state={state}/       Regrid county files (whole, unmodified)
    nlcd_features/          01a output, filtered to these plants' candidates
    nlcd/                   NLCD raster, clipped to the counties above
    od_features/            01b output, filtered to these plants
    od_features_corrected/  01c output, filtered to these plants
    features/               02 output tables, filtered
    holdout/                holdout manifest + truth
    diagnostics/            08 output (01d's input), if present
    reference/              census_subset.gpkg, OSM points — both clipped
```

## Setup on the target machine

Four paths in `scripts/config.py` need to point inside this bundle. Everything
else in that file is already relative to `ROOT`:

```python
ROOT        = Path("/wherever/you/put/tpqa_test")
PARCEL_BASE = ROOT / "data" / "parcels"
NLCD_PATH   = ROOT / "data" / "nlcd" / "{C.NLCD_PATH.name}"
CENSUS_GDB  = ROOT / "data" / "reference" / "census_subset.gpkg"
```

`CENSUS_GDB` is a GeoPackage here rather than a File Geodatabase, but the
layer names are identical (`County`, `County_Subdivision`,
`Incorporated_Place`, `Census_Designated_Place`), so `02_feature_engineering.py`
reads it with no code change.

`_common.sh` and the `.slurm` wrappers assume SLURM and the HPC venv — off-HPC,
call the Python scripts directly. The `.slurm` files are included anyway
because their headers document each script's expected options and memory
profile.

## Known issues carried over

These are real, already-diagnosed problems present in this snapshot. They are
not artifacts of the subsetting:

1. **`geom_wkb` is a trained feature of `stage1_rf_model.joblib`.**
   `point_in_parcel_lookup` does `SELECT pts.*`, which returns the WKB blob
   column that was passed in; `03_train_stage1.py`'s `DROP_COLS` doesn't drop
   it, so the model was fitted with a per-row unique binary blob one-hot
   encoded as a categorical. Inference cannot supply it (DuckDB returns
   `bytearray`, which is unhashable and crashes `OneHotEncoder`) and cannot
   omit it (the fitted pipeline requires the column). Stage 1's threshold and
   permutation importances were computed with this contaminating feature
   present. **Fixing this requires retraining, not just a code patch.**

2. **`x_5070` / `y_5070` are trained features of Stage 1 and Stage 2a.**
   Added only for spatial-CV clustering in `03`/`04`, never dropped before
   the model saw them, so raw projected coordinates are live features — a
   plausible route to geographic memorisation rather than transferable signal.
   Same fix path: retrain.

3. **`05_plant_features.parquet` has `STATE_CODE_x`, `STATE_CODE_y` and
   `STATE_CODE`.** `build_discharge_features()` returns its own `STATE_CODE`
   column, which collides on the `CWNS_ID` merge in `build_plant_features()`;
   pandas suffixes both, and the defensive re-attach then adds a third. Only
   the unsuffixed one is correct.

4. **`14_stage1_training.parquet` / `15_stage2_training.parquet` in this
   bundle are the 3-state (OH/MS/DE) pilot versions, not nationwide.** Every
   `02` run overwrites these at a fixed path regardless of `--states` scope.
   See `docs/TPQA_MASTER_REFERENCE.md` §8.3.

5. **`10_parcel_features_by_state/` is deliberately absent.** It's a
   persistent per-state cache whose unscoped glob caused a 5.47x row inflation
   bug (master reference §8.2). Let `02` rebuild it locally.

## What can and can't run here

| Script | Runs locally? | Notes |
|---|---|---|
| `01a_extract_parcels.py` | yes | NLCD raster is clipped to these counties |
| `01b_run_object_detection.py` | yes | needs internet (Planetary Computer) + torch |
| `01c_run_od_corrected_locations.py` | yes | same |
| `01d_nlcd_topup.py` | yes | needs `data/diagnostics/` from 08 |
| `02_feature_engineering.py` | yes | rebuilds the by-state cache |
| `03` / `04` / `06` / `07` | yes | fixture-scale only — see warning above |
| `05_run_inference.py` | yes | currently blocked by issue 1 |
| `08` / `08b` / `09` | yes | |
| diagnostics / `inspect_*` | yes | |

Anything requiring a state's full parcel universe, or states other than
{state}, will not work — that data isn't here by design.
"""
    (BUNDLE / "MANIFEST.md").write_text(text)
    log(f"    MANIFEST.md written ({len((BUNDLE / 'MANIFEST.md').read_text())} chars)")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=str, default="OH")
    ap.add_argument("--n-plants", type=int, default=30,
                     help="target plant count (all labelled plants in the chosen "
                          "counties are taken regardless; this tops up with unlabelled)")
    ap.add_argument("--n-counties", type=int, default=2,
                     help="how many counties to draw plants from. More counties = "
                          "more labels but a bigger parcel-data payload.")
    ap.add_argument("--no-raster", action="store_true",
                     help="skip the NLCD raster clip (saves size; 01a/01d then "
                          "can't be run locally)")
    ap.add_argument("--clean", action="store_true",
                     help="delete any existing bundle first")
    args = ap.parse_args()

    log("=== build_test_bundle.py ===")
    log(f"Source ROOT : {C.ROOT}")
    log(f"Bundle      : {BUNDLE}")
    log(f"State       : {args.state}")

    if args.clean and BUNDLE.exists():
        log(f"\n--clean: removing existing {BUNDLE}")
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True, exist_ok=True)

    selected = select_plants(args.state, args.n_plants, args.n_counties)
    selected_ids = set(selected["CWNS_ID"].astype(str))
    geoids = counties_needed(selected)

    collect_scripts()
    collect_cwns()
    collect_training()
    collect_parcels(args.state, geoids)
    keep_uuids = collect_nlcd_features(args.state, selected)
    collect_od(selected_ids)
    collect_features(selected_ids, keep_uuids)
    collect_models()
    collect_holdout()
    collect_diagnostics()
    collect_reference(args.state, geoids)
    if not args.no_raster:
        collect_nlcd_raster(geoids)
    collect_logs()
    collect_docs()
    write_manifest(args.state, selected, geoids, args)

    log("\n" + "=" * 60)
    log("BUNDLE COMPLETE")
    log("=" * 60)
    for sub in sorted(p for p in BUNDLE.rglob("*") if p.is_dir() and len(p.relative_to(BUNDLE).parts) <= 2):
        size = dir_size_mb(sub)
        if size > 0.5:
            log(f"  {str(sub.relative_to(BUNDLE)):<42} {size:>9.1f} MB")
    log(f"\n  TOTAL: {dir_size_mb(BUNDLE):.1f} MB")
    log(f"\n  {BUNDLE}")
    log(f"\n  To compress before download:")
    log(f"    cd {TESTING_ROOT} && tar -czf tpqa_test.tar.gz tpqa_test/")
    log(f"\n  Read tpqa_test/MANIFEST.md first -- it lists the four config.py")
    log(f"  paths to change, and the known issues carried over in this snapshot.")


if __name__ == "__main__":
    main()
