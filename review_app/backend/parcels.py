"""
parcels.py
==========
Live parcel geometry + attribute lookup against the LOCAL Regrid parquet
mirror -- same query shape 01a_extract_parcels.py / 02_feature_engineering.py
use on HPC against the real store, just pointed at your local copy instead.
No HPC connection needed at review time.

Geometry and attributes come from ONE query (get_parcel_context), not two --
this is what supplies context (owner, LBCS codes, acreage) for BOTH the
reported location's parcel AND every candidate, uniformly. 2026-08-26: this
replaced an earlier design that threaded engineered LBCS/owner features
through 05_run_inference.py -> 10_build_review_queue.py -> the app database,
which only ever reached candidates (the reported parcel was never in that
data path at all, since 05's plant_summary.parquet was kept deliberately
narrow). Raw attributes straight from the source parquet are simpler, reach
both, and match what was actually asked for -- context, not the full
engineered feature set.

Column names (owner, ll_gisacre, ll_bldg_count, lbcs_*_desc, zoning_type)
confirmed directly against 02_feature_engineering.py's fetch_parcel_attrs(),
which queries this exact store on HPC.

A single connection is held open for the app's lifetime (module-level) rather
than reconnecting per request.
"""
import json

import duckdb

import config as C

_con = None

ATTR_COLS = ["owner", "ll_gisacre", "ll_bldg_count", "lbcs_activity_desc",
             "lbcs_function_desc", "lbcs_structure_desc", "lbcs_site_desc",
             "lbcs_ownership_desc", "zoning_type", "zoning_subtype"]


def get_connection():
    global _con
    if _con is None:
        _con = duckdb.connect()
        # enable_geoparquet_conversion=false is REQUIRED, not optional -- without
        # it, DuckDB tries to interpret Regrid's embedded GeoParquet metadata and
        # fails with "Geoparquet metadata does not have a version" (confirmed
        # 2026-08-26 against the real local mirror). Same setting the HPC
        # pipeline already sets in every script that touches parcel parquet.
        _con.execute("INSTALL spatial; LOAD spatial; "
                     "SET enable_geoparquet_conversion = false;")
    return _con


def get_parcel_context(state: str, ll_uuids: list[str]) -> dict[str, dict]:
    """Returns {ll_uuid: {geometry, owner, ll_gisacre, ll_bldg_count,
    lbcs_activity_desc, lbcs_function_desc, lbcs_structure_desc,
    lbcs_site_desc, lbcs_ownership_desc, zoning_type, zoning_subtype}}.

    Missing ll_uuids (not found in the local store, bad state code) are
    simply absent from the result -- callers handle a missing key, don't
    assume every requested uuid comes back."""
    if not ll_uuids:
        return {}

    con = get_connection()
    glob_pattern = str(C.REGRID_ROOT / C.REGRID_STATE_GLOB.format(state=state))
    uuid_list = ", ".join(f"'{u}'" for u in ll_uuids)
    attr_select = ", ".join(ATTR_COLS)

    try:
        df = con.execute(f"""
            SELECT {C.REGRID_UUID_COL} AS ll_uuid,
                   ST_AsGeoJSON(ST_GeomFromWKB({C.REGRID_GEOM_COL})) AS geojson,
                   {attr_select}
            FROM read_parquet('{glob_pattern}')
            WHERE {C.REGRID_UUID_COL} IN ({uuid_list})
        """).df()
    except Exception as e:
        print(f"  Parcel lookup failed for state={state}: {e}")
        print(f"  Glob pattern tried: {glob_pattern}")
        print(f"  If this is a 'no files found' error, check config.py's "
              f"REGRID_STATE_GLOB against your actual local folder layout.")
        return {}

    out = {}
    for _, row in df.iterrows():
        try:
            geometry = json.loads(row["geojson"])
        except (TypeError, ValueError):
            geometry = None
        entry = {"geometry": geometry}
        for col in ATTR_COLS:
            val = row.get(col)
            entry[col] = None if (val is None or (isinstance(val, float) and val != val)) else val
        out[row["ll_uuid"]] = entry
    return out


def get_single_parcel_context(state: str, ll_uuid: str) -> dict | None:
    if not ll_uuid:
        return None
    result = get_parcel_context(state, [ll_uuid])
    return result.get(ll_uuid)


# Kept for anything still calling the geometry-only interface -- new code
# should use get_parcel_context() instead.
def get_parcel_geometries(state: str, ll_uuids: list[str]) -> dict[str, dict]:
    ctx = get_parcel_context(state, ll_uuids)
    return {uuid: entry["geometry"] for uuid, entry in ctx.items() if entry["geometry"]}
