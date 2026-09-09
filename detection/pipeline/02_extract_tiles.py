"""
02_extract_tiles.py
===================
NAIP tile extractor for the labeling set. Reads whichever layers
01_sample_sites.py wrote, generates tiles, fetches 4-band NAIP imagery from
Planetary Computer, and writes RGB PNGs + NDWI GeoTIFFs plus an appended
metadata row per tile.

CHANGED 2026-09-03
    Imagery source: USDA ArcGIS -> Planetary Computer STAC. The USDA
    exportImage endpoint went dead 2026-08-28. Fetch logic lives in
    naip_fetch.py, shared with convert_tiles_to_500m.py rather than copied,
    since it carries dated bug fixes that are easy to lose in a rewrite.

    Tiling: overlapping grid over parcel bounds -> ONE-OR-NINE. Every site
    gets a tile centered on its parcel centroid. If the parcel extends past
    that tile's footprint, the surrounding ring of 8 is added, for 9 total.
    Never more. A parcel 10x the tile size gets the same 9 tiles as one
    that overruns by a metre -- deliberate, this set is for human review and
    the old grid could emit dozens of tiles for one site.

    Tile size: 200m -> 500m (config.py TILE_SIZE_M, OVERLAP_PCT now 0).

    Sites: plants + TRI -> plants + review_fp + TRI.

    NOTE: this script feeds the LABELING loop only. The HPC correction/
    pipeline has its own config and tiling and is untouched by any of this.

Filename contract (the pipeline's index -- do not change):
    plants/review: {CWNS_ID}_{ll_uuid}_r{row:02d}_c{col:02d}_rgb.png
    TRI:           TRI_{TRI_FACILITY_ID}_r01_c01_rgb.png

    In the 1-or-9 scheme rows/cols are always numbered as a 3x3, so the
    center tile is r02_c02 whether or not the ring exists. A lone tile is
    therefore r02_c02, not r01_c01 -- that way "is this the centroid tile"
    is answerable from the filename alone, without checking for siblings.

Safety:
    - SKIP-EXISTING: a tile whose RGB PNG already exists is never re-fetched.
    - ADDITIVE METADATA: new rows appended; tiles already in the CSV skipped.
    - --dry-run reports the tile count without fetching anything.
"""

import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.transform
from shapely import from_wkb
from shapely.geometry import box, Point
from PIL import Image

# config.py is in THIS directory. See the note in 01_sample_sites.py about
# why the old parents[1] was wrong.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import naip_fetch as NF

SAMPLE_LAYER_REVIEW = "review_fp"


# ===========================================================================
# Parcel loading
# ===========================================================================
def parcel_path(state: str, geoid: str) -> Path:
    return C.PARCEL_BASE / f"state={state}" / f"{geoid}.parquet"


def load_parcels(state: str, geoid: str) -> gpd.GeoDataFrame | None:
    path = parcel_path(state, geoid)
    if not path.exists():
        print(f"    parcel file not found: {path}")
        return None
    try:
        df = pd.read_parquet(path, columns=[C.PARCEL_ID_FIELD, C.PARCEL_WKB_FIELD])
    except Exception:
        df = pd.read_parquet(path)
    if C.PARCEL_WKB_FIELD not in df.columns:
        print(f"    no '{C.PARCEL_WKB_FIELD}' column in {path.name}")
        return None
    geom = from_wkb(df[C.PARCEL_WKB_FIELD].to_numpy())   # handles EWKB
    df = df.drop(columns=[C.PARCEL_WKB_FIELD])
    return gpd.GeoDataFrame(df, geometry=geom, crs=C.EXPORT_CRS)


# ===========================================================================
# Tile grid: one or nine
# ===========================================================================
def generate_tile_grid(geom_or_bounds, force_single: bool = False) -> list[dict]:
    """One tile on the centroid; the ring of 8 too if the parcel overruns it.

    Accepts a geometry or a (xmin, ymin, xmax, ymax) tuple, in PROJECTED_CRS
    (meters). The centroid is the geometry's true centroid when available --
    not the bounds midpoint, which for an L-shaped or crescent parcel can
    land off the parcel entirely.
    """
    size = C.TILE_SIZE_M
    half = size / 2

    if hasattr(geom_or_bounds, "centroid"):
        ctr = geom_or_bounds.centroid
        cx, cy = ctr.x, ctr.y
        xmin, ymin, xmax, ymax = geom_or_bounds.bounds
    else:
        xmin, ymin, xmax, ymax = geom_or_bounds
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2

    # Does the parcel fit inside the single centered tile?
    fits = (xmin >= cx - half and xmax <= cx + half
            and ymin >= cy - half and ymax <= cy + half)

    offsets = [(0, 0)] if (fits or force_single) else [
        (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]

    tiles = []
    for dr, dc in offsets:
        tx, ty = cx + dc * size, cy - dr * size   # row 1 is NORTH, so -dr
        tiles.append(dict(
            x_min=tx - half, y_min=ty - half,
            x_max=tx + half, y_max=ty + half,
            row=2 + dr, col=2 + dc, ctr_x=tx, ctr_y=ty))
    return tiles


def tiles_to_wgs84(tiles: list[dict]):
    """Batch-reproject bboxes and centers PROJECTED_CRS -> EXPORT_CRS."""
    boxes = gpd.GeoSeries(
        [box(t["x_min"], t["y_min"], t["x_max"], t["y_max"]) for t in tiles],
        crs=C.PROJECTED_CRS).to_crs(C.EXPORT_CRS)
    b = boxes.bounds.to_numpy()
    ctr = gpd.GeoSeries(
        [Point(t["ctr_x"], t["ctr_y"]) for t in tiles],
        crs=C.PROJECTED_CRS).to_crs(C.EXPORT_CRS)
    return ([tuple(map(float, row)) for row in b],
            [(float(p.x), float(p.y)) for p in ctr])


# ===========================================================================
# Saving
# ===========================================================================
def save_rgb_png(arr4: np.ndarray, path: Path):
    Image.fromarray(np.transpose(arr4[0:3], (1, 2, 0)), mode="RGB").save(path)


def save_ndwi_tif(arr4: np.ndarray, bbox_wgs84: tuple, path: Path):
    """NDWI = (Green - NIR)/(Green + NIR); green=idx1, nir=idx3."""
    green, nir = arr4[1].astype(np.float32), arr4[3].astype(np.float32)
    denom = green + nir
    ndwi = np.where(denom == 0, 0.0, (green - nir) / denom).astype(np.float32)
    h, w = ndwi.shape
    transform = rasterio.transform.from_bounds(*bbox_wgs84, w, h)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="float32", crs=f"EPSG:{C.EXPORT_CRS}",
                       transform=transform) as dst:
        dst.write(ndwi, 1)


# ===========================================================================
# Metadata
# ===========================================================================
def load_existing_tile_ids() -> set:
    if C.METADATA_CSV.exists():
        try:
            return set(pd.read_csv(C.METADATA_CSV, usecols=["tile_id"])["tile_id"])
        except Exception:
            return set()
    return set()


def existing_rgb(tile_id: str) -> bool:
    return (C.RGB_DIR / f"{tile_id}_rgb.png").exists()


def append_metadata(rows: list):
    if not rows:
        return
    df = pd.DataFrame(rows).reindex(columns=C.METADATA_COLUMNS)
    df.to_csv(C.METADATA_CSV, mode="a",
              header=not C.METADATA_CSV.exists(), index=False)


def make_meta_row(**kw) -> dict:
    row = {col: kw.get(col, "") for col in C.METADATA_COLUMNS}
    row["target_res_m"] = C.TARGET_RES_M
    row["image_px"] = C.IMAGE_PX
    row.update(kw)
    row["label"] = ""
    return row


# ===========================================================================
# Worker
# ===========================================================================
def fetch_and_save(task: dict) -> dict | None:
    bbox = task["bbox"]
    try:
        arr4 = NF.timed_fetch(bbox, task["item_url"], C.IMAGE_PX,
                              catalog=task["catalog"])
    except NF.TileOutsideItemCoverage:
        print(f"    {task['tile_id']}: SKIPPED -- not covered by any NAIP quad "
              f"(cross-quad boundary)")
        return None
    except Exception as e:
        print(f"    {task['tile_id']}: FAILED -- {e}")
        return None

    tid = task["tile_id"]
    rgb_path = C.RGB_DIR / f"{tid}_rgb.png"
    ndwi_path = C.NDWI_DIR / f"{tid}_ndwi.tif"
    save_rgb_png(arr4, rgb_path)
    save_ndwi_tif(arr4, bbox, ndwi_path)
    return make_meta_row(
        tile_id=tid, source=task["source"], CWNS_ID=task["cwns_id"],
        TRI_FACILITY_ID=task["tri_id"], ll_uuid_primary=task["ll_uuid"],
        ll_uuid_alternates=task["alt"], st=task["st"], geoid=task["geoid"],
        tile_row=task["row"], tile_col=task["col"],
        ctr_lon=task["ctr"][0], ctr_lat=task["ctr"][1],
        bbox_xmin=bbox[0], bbox_ymin=bbox[1], bbox_xmax=bbox[2], bbox_ymax=bbox[3],
        source_res_m=C.TARGET_RES_M, acq_date=task["acq_date"],
        rgb_path=str(rgb_path), ndwi_path=str(ndwi_path))


def run_tasks(tasks: list, executor) -> list:
    if not tasks:
        return []
    rows = []
    for fut in as_completed([executor.submit(fetch_and_save, t) for t in tasks]):
        row = fut.result()
        if row is not None:
            rows.append(row)
    return rows


def resolve_item(catalog, bbox):
    """Resolve the NAIP item once per SITE, from the centroid tile. Also
    yields acq_date for free -- the STAC item carries it, so unlike the old
    USDA /identify call this costs no extra request."""
    lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    item = NF.with_retry(NF.find_naip_item, catalog, lon, lat)
    if item is None:
        return None, ""
    return item.assets["image"].href, str(item.datetime.date())


# ===========================================================================
# Parcel-driven sites (plants and review false positives share this path)
# ===========================================================================
def build_parcel_tasks(gdf, source: str, existing_ids: set, catalog,
                       dry_run: bool) -> tuple[list, dict]:
    gdf = gdf.copy()
    if "CWNS_ID" in gdf.columns:
        gdf["CWNS_ID"] = gdf["CWNS_ID"].astype(str)
    gdf = gdf.to_crs(C.PROJECTED_CRS)

    counties = gdf[["st", "geoid"]].drop_duplicates()
    print(f"{source}: {len(gdf)} site(s) across {len(counties)} county/counties")

    tasks = []
    stats = {"single": 0, "ring": 0, "no_parcel": 0, "no_item": 0}

    for i, (_, cc) in enumerate(counties.iterrows(), start=1):
        state, geoid = cc["st"], cc["geoid"]
        parcels = load_parcels(state, geoid)
        if parcels is None or len(parcels) == 0:
            print(f"[{i}/{len(counties)}] {state}/{geoid}: no parcels")
            continue
        parcels = parcels.to_crs(C.PROJECTED_CRS)[[C.PARCEL_ID_FIELD, "geometry"]]

        county_sites = gdf[(gdf["st"] == state) & (gdf["geoid"] == geoid)]

        # Review rows already carry the parcel uuid (rank1_ll_uuid), so look
        # the geometry up directly rather than spatial-joining their point.
        # Two reasons: (a) their lat/lon is a vertex mean, not a centroid --
        # find_false_top_picks.py's _flatten_coords says so itself -- and can
        # land off the parcel, and (b) the site column and PARCEL_ID_FIELD are
        # both literally "ll_uuid", so sjoin would suffix them to
        # ll_uuid_left/ll_uuid_right and this lookup would KeyError.
        has_uuid = ("ll_uuid" in county_sites.columns
                    and county_sites["ll_uuid"].notna().any())
        if has_uuid:
            sites = [(r.get("CWNS_ID", idx), r["ll_uuid"], "")
                     for idx, r in county_sites.iterrows()]
        else:
            joined = gpd.sjoin(county_sites, parcels, how="left",
                               predicate="intersects")
            uuid_col = (C.PARCEL_ID_FIELD if C.PARCEL_ID_FIELD in joined.columns
                        else f"{C.PARCEL_ID_FIELD}_right")
            group_key = "CWNS_ID" if "CWNS_ID" in joined.columns else joined.index
            sites = []
            for site_id, grp in joined.groupby(group_key):
                uu = list(grp[uuid_col].dropna().unique())
                if not uu:
                    stats["no_parcel"] += 1
                    continue
                sites.append((site_id, uu[0], "|".join(map(str, uu[1:]))))

        parcel_geoms = parcels.set_index(C.PARCEL_ID_FIELD)["geometry"]
        for site_id, primary, alts in sites:
            if primary not in parcel_geoms.index:
                print(f"    {site_id}: ll_uuid {primary} not in "
                      f"{state}/{geoid} parcels -- skipping")
                stats["no_parcel"] += 1
                continue
            pgeom = parcel_geoms.loc[primary]
            if hasattr(pgeom, "iloc"):     # duplicate uuid in the parquet
                pgeom = pgeom.iloc[0]

            grid = generate_tile_grid(pgeom)
            stats["ring" if len(grid) > 1 else "single"] += 1
            bboxes, centers = tiles_to_wgs84(grid)

            # Centroid tile is index 0 when single, else the (0,0) offset.
            ctr_idx = next(k for k, t in enumerate(grid)
                           if t["row"] == 2 and t["col"] == 2)
            if dry_run:
                item_url, acq = "", ""
            else:
                item_url, acq = resolve_item(catalog, bboxes[ctr_idx])
                if item_url is None:
                    print(f"    {site_id}: no NAIP item -- skipping site")
                    stats["no_item"] += 1
                    continue

            for t, bbox, ctr in zip(grid, bboxes, centers):
                tid = f"{site_id}_{primary}_r{t['row']:02d}_c{t['col']:02d}"
                if tid in existing_ids or existing_rgb(tid):
                    continue
                tasks.append(dict(
                    tile_id=tid, bbox=bbox, ctr=ctr, item_url=item_url,
                    catalog=catalog, acq_date=acq, source=source,
                    cwns_id=str(site_id), tri_id="", ll_uuid=str(primary),
                    alt=alts, st=state, geoid=geoid,
                    row=t["row"], col=t["col"]))
                existing_ids.add(tid)

    return tasks, stats


# ===========================================================================
# TRI (single centered tile per facility -- points, no parcel)
# ===========================================================================
def build_tri_tasks(existing_ids: set, catalog, dry_run: bool) -> list:
    tri = gpd.read_file(C.SAMPLE_GPKG,
                        layer=C.SAMPLE_LAYER_TRI).to_crs(C.PROJECTED_CRS)
    print(f"\ntri: {len(tri)} facilities")

    tasks = []
    for _, f in tri.iterrows():
        tri_id = str(f[C.TRI_ID_FIELD])
        tid = f"TRI_{tri_id}_r01_c01"
        if tid in existing_ids or existing_rgb(tid):
            continue
        grid = generate_tile_grid(f.geometry.centroid.buffer(0.01),
                                  force_single=True)
        grid[0]["row"] = grid[0]["col"] = 1   # TRI keeps the r01_c01 contract
        (bbox,), (ctr,) = tiles_to_wgs84(grid)

        if dry_run:
            item_url, acq = "", ""
        else:
            item_url, acq = resolve_item(catalog, bbox)
            if item_url is None:
                print(f"    TRI_{tri_id}: no NAIP item -- skipping")
                continue

        tasks.append(dict(
            tile_id=tid, bbox=bbox, ctr=ctr, item_url=item_url,
            catalog=catalog, acq_date=acq, source="tri", cwns_id="",
            tri_id=tri_id, ll_uuid="", alt="", st="", geoid="", row=1, col=1))
        existing_ids.add(tid)
    return tasks


def list_gpkg_layers(path: Path) -> set:
    """Layer names from a GeoPackage, whichever engine is installed.

    GeoPandas >=1.0 defaults to pyogrio and does not pull in fiona, so
    importing fiona directly fails on a normal modern install. Try the
    engines in order of how likely they are to be present.
    """
    try:                                    # GeoPandas >= 1.0
        return set(gpd.list_layers(str(path))["name"])
    except AttributeError:
        pass
    try:
        import pyogrio
        return {row[0] for row in pyogrio.list_layers(str(path))}
    except ImportError:
        pass
    import fiona                            # last resort
    return set(fiona.listlayers(str(path)))


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["plants", "review", "tri", "all"],
                    default="all",
                    help="which sample layer(s) to extract (default: all present)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the tile count without fetching anything")
    ap.add_argument("--limit", type=int,
                    help="cap total tiles fetched (applied after counting)")
    args = ap.parse_args()

    print("=== 02_extract_tiles.py ===")
    print(f"tile size: {C.TILE_SIZE_M}m -> {C.IMAGE_PX}px "
          f"at {C.TARGET_RES_M}m/px\n")
    C.ensure_dirs()
    if not C.SAMPLE_GPKG.exists():
        print(f"ERROR: sample not found: {C.SAMPLE_GPKG}\nRun 01_sample_sites.py first.")
        return

    layers = list_gpkg_layers(C.SAMPLE_GPKG)
    existing = load_existing_tile_ids()
    print(f"{len(existing)} tiles already in metadata (will skip)")
    print(f"layers present: {sorted(layers)}\n")

    catalog = None if args.dry_run else NF.open_catalog()
    all_tasks, all_stats = [], {}

    def wants(name):
        return args.source in (name, "all")

    if wants("plants") and C.SAMPLE_LAYER_PLANTS in layers:
        gdf = gpd.read_file(C.SAMPLE_GPKG, layer=C.SAMPLE_LAYER_PLANTS)
        t, s = build_parcel_tasks(gdf, "plant", existing, catalog, args.dry_run)
        all_tasks += t
        all_stats["plant"] = s

    if wants("review") and SAMPLE_LAYER_REVIEW in layers:
        gdf = gpd.read_file(C.SAMPLE_GPKG, layer=SAMPLE_LAYER_REVIEW)
        t, s = build_parcel_tasks(gdf, "review_fp", existing, catalog, args.dry_run)
        all_tasks += t
        all_stats["review_fp"] = s

    if wants("tri") and C.SAMPLE_LAYER_TRI in layers:
        all_tasks += build_tri_tasks(existing, catalog, args.dry_run)

    # ---- report before doing anything expensive --------------------------
    print("\n--- tile plan ---")
    for src, s in all_stats.items():
        print(f"  {src}: {s['single']} single-tile site(s), "
              f"{s['ring']} nine-tile site(s)"
              f"  -> {s['single'] + s['ring'] * 9} tile(s)")
        if s["no_parcel"]:
            print(f"    {s['no_parcel']} site(s) matched no parcel -- skipped")
        if s["no_item"]:
            print(f"    {s['no_item']} site(s) had no NAIP item -- skipped")
    print(f"  TOTAL NEW TILES: {len(all_tasks)}")

    if args.limit and len(all_tasks) > args.limit:
        print(f"  --limit {args.limit}: dropping "
              f"{len(all_tasks) - args.limit} tile(s)")
        all_tasks = all_tasks[:args.limit]

    if args.dry_run:
        print("\n--dry-run: nothing fetched. Re-run without --dry-run to "
              "download, or adjust --n in 01_sample_sites.py first.")
        return

    print(f"\nFetching with {C.MAX_WORKERS} workers...\n")
    with ThreadPoolExecutor(max_workers=C.MAX_WORKERS) as executor:
        rows = run_tasks(all_tasks, executor)
        for _ in range(C.MAX_WORKERS):
            executor.submit(NF.close_thread_datasets)
    append_metadata(rows)

    failed = len(all_tasks) - len(rows)
    print(f"\nWrote {len(rows)} tile(s)" + (f", {failed} failed" if failed else ""))
    print(f"Metadata now indexes {len(load_existing_tile_ids())} tiles: "
          f"{C.METADATA_CSV}")
    print("\nNext: label the new tiles in label_app.R, then 03_prepare_dataset.py")


if __name__ == "__main__":
    main()