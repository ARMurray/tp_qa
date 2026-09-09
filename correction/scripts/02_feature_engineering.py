"""
02_feature_engineering.py
==========================
Port of build_features_part1.R + part2.R + part3.R, combined into one script
(the R version split these across three files partly for HPC job-array
reasons that don't apply to a single local machine).

Stages, matching the R structure:
  PART 1  Plant-level features: discharge, population, census geography.
  PART 2  Parcel-level features: LBCS reclassification, owner regex flags,
          has_ww_keyword, OSM wastewater tag, county data quality.
  PART 3  Assemble training tables:
            - Stage 1: reported-parcel lookup for the 'classes' label layer,
              joined to plant + parcel features, PLUS OD features from 01b
              (NEW vs. the R pipeline -- Step 1 of the OD integration plan).
            - Stage 2: k=18 candidate parcels for the 'corrections' label
              layer, using 01a's NLCD output as the candidate universe
              directly rather than re-querying the parcel store.

Scope: like 01a/01b, defaults to the training-labeled plant universe
('classes' + 'corrections' layers of training_locations.gpkg), not the full
national plant list. Pass --full-universe once a trained Stage 1 model
exists to decide who needs this at inference time.

NOTE on OD features: only joined onto STAGE 1 training. 01b only computes
detection features for the SINGLE parcel a reported point falls on, not for
every Stage 2 candidate parcel -- so there's currently no OD signal to give
Stage 2. Extending OD to Stage 2 candidates would mean running detection
across every candidate parcel in the k=18 ring, a much bigger undertaking,
not attempted here.

Requires: duckdb, geopandas, pandas, numpy, pyarrow, shapely, h3

Usage:
    python 02_feature_engineering.py [--states OH,PA] [--full-universe]
"""
import argparse
import re
import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely import from_wkb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

# ===========================================================================
# LBCS / ownership reclassification (exact port of the R case_when blocks)
# ===========================================================================
ACTIVITY_MAP = {
    'Activities associated with utilities (water, sewer, power, etc.)': "Water Utility",
    'Health care, medical, or treatment': "Water Utility",
    'Industrial, manufacturing, and waste-related': "Water Utility",
    'Sewer-related control, monitor, or distribution': "Water Utility",
    'Sewer treatment and processing': "Water Utility",
    'Social, institutional, or infrastructure-related': "Water Utility",
    'Solid waste management': "Water Utility",
    'Water-supply-related': "Water Utility",
    'Water purification and filtration': "Water Utility",
    'Water storing, pumping, or piping': "Water Utility",
    'Power generation, control, monitor, or distribution': "Utility Other",
    'Power generation, storage, or processing': "Utility Other",
    'Power transmission lines or control': "Utility Other",
    'Storage of natural gas, fuels, etc.': "Utility Other",
    'Telecommunications-related control, monitor, or distribution': "Utility Other",
    'Farming, tilling, plowing, harvesting, or related': "Agriculture",
    'Livestock related': "Agriculture",
    'Logging': "Agriculture",
    'Mining including surface and subsurface strip mining': "Agriculture",
    'Pasturing, grazing, etc.': "Agriculture",
    'Agricultural vacant land': "Vacant",
    'Vacant parcel': "Vacant",
    'Vacant parcel, undevelopable': "Vacant",
    'Commercial vacant land': "Vacant",
    'Residential vacant land': "Vacant",
    'Industrial vacant land': "Vacant",
    'Leisure': "Leisure",
    'Boating, sailing, etc.': "Leisure",
    'Camping': "Leisure",
    'Golf': "Leisure",
    'Sailing, boating, and other port, marine and water-based': "Leisure",
    'Promenading and other activities in parks': "Leisure",
    'Passive leisure activity': "Leisure",
    'Gatherings at fairs and exhibitions': "Leisure",
    'Hockey, ice skating, etc.': "Leisure",
    'Skiing, snowboarding, etc.': "Leisure",
    'Movies, concerts, or entertainment shows': "Commercial",
    'Goods-oriented shopping': "Commercial",
    'Service-oriented shopping': "Commercial",
    'Shopping': "Commercial",
    'Shopping, business, or trade': "Commercial",
    'Restaurant-type activity': "Commercial",
    'Restaurant-type activity with drive-through': "Commercial",
    'Household': "Household",
    'Residential': "Household",
    'Unknown': "Unknown",
}
FUNCTION_UTILITY = {
    'Waste treatment and disposal',
    'Utilities and utility services',
    'Transportation, communication, information, and utilities',
    'Sewer, solid waste, and related services',
}
OWNERSHIP_GOVERNMENT = {
    'City, Village, Township, etc.', 'State government',
    'Federal government', 'County, Parish, Province, etc.',
}
OWNERSHIP_PRIVATE = {'Private persons and private joint ownership', 'Private trusts'}
OWNERSHIP_COMMERCIAL = {'Businesses and commercial entities'}

DOMINANT_CLASS_MAP = {
    11: "Open Water",
    21: "Developed", 22: "Developed", 23: "Developed", 24: "Developed",
    41: "Forest", 42: "Forest", 43: "Forest",
    81: "Agriculture", 82: "Agriculture",
    90: "Wetland", 95: "Wetland",
    31: "Barren/Shrub/Grassland", 52: "Barren/Shrub/Grassland", 71: "Barren/Shrub/Grassland",
}


def reclass_activity(x):
    if pd.isna(x):
        return "Unknown"
    return ACTIVITY_MAP.get(x, "Other")


def reclass_function(x):
    if pd.isna(x):
        return "Unknown"
    return "Utility" if x in FUNCTION_UTILITY else "Other"


def reclass_ownership(x):
    if pd.isna(x):
        return "Other / Unknown"
    if x in OWNERSHIP_GOVERNMENT:
        return "Government"
    if x in OWNERSHIP_PRIVATE:
        return "Private"
    if x in OWNERSHIP_COMMERCIAL:
        return "Commercial"
    return "Other / Unknown"


def reclass_dominant_class(x):
    if pd.isna(x):
        return "Unknown"
    return DOMINANT_CLASS_MAP.get(int(x), "Other")


PARCEL_WW_PATTERN = re.compile(
    r"\b(" + "|".join(C.PARCEL_WW_KEYWORDS) + r")\b", re.IGNORECASE
)


# ===========================================================================
# Shared: training plant universe (same pattern as 01a/01b)
# ===========================================================================
def load_training_plant_ids() -> set[str]:
    if not C.TRAINING_GPKG.exists():
        raise FileNotFoundError(
            f"{C.TRAINING_GPKG} not found. Copy training_locations.gpkg in from "
            f"the Location_Correction repo first."
        )
    classes = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CLASSES)
    corrections = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CORRECTIONS)
    return set(classes["CWNS_ID"].astype(str)) | set(corrections["CWNS_ID"].astype(str))


def load_treatment_plants(states, training_only=True) -> pd.DataFrame:
    facility_types = pd.read_csv(C.CWNS_DIR / "FACILITY_TYPES.txt", dtype=str, encoding="latin1")
    treatment_ids = set(
        facility_types.loc[facility_types["FACILITY_TYPE"] == "Treatment Plant", "CWNS_ID"]
    )
    loc = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt", dtype={"CWNS_ID": str}, encoding="latin1")
    loc = loc[loc["CWNS_ID"].isin(treatment_ids)]
    loc["LATITUDE"] = pd.to_numeric(loc["LATITUDE"], errors="coerce")
    loc["LONGITUDE"] = pd.to_numeric(loc["LONGITUDE"], errors="coerce")
    loc = loc.dropna(subset=["LATITUDE", "LONGITUDE"]).drop_duplicates(subset="CWNS_ID")
    loc = loc[["CWNS_ID", "STATE_CODE", "LATITUDE", "LONGITUDE"]]
    loc["h3_res9"] = loc.apply(
        lambda r: h3.latlng_to_cell(r["LATITUDE"], r["LONGITUDE"], 9), axis=1)

    if training_only:
        loc = loc[loc["CWNS_ID"].isin(load_training_plant_ids())]
    if states:
        loc = loc[loc["STATE_CODE"].isin(states)]
    return loc.reset_index(drop=True), treatment_ids


# ===========================================================================
# PART 1: Plant-level features
# ===========================================================================
def build_discharge_features(treatment_ids: set) -> pd.DataFrame:
    df = pd.read_csv(
        C.CWNS_DIR / "DISCHARGES.csv",
        names=["CWNS_ID", "FACILITY_ID", "STATE_CODE", "DISCHARGE_TYPE",
               "PRESENT_DISCHARGE_PERCENTAGE", "PROJECTED_DISCHARGE_PERCENTAGE",
               "DISCHARGES_TO"],
        header=0, dtype={"CWNS_ID": str}, encoding="latin1",
    )
    df = df[df["CWNS_ID"].isin(treatment_ids)]

    def agg(g):
        surf = g["DISCHARGE_TYPE"] == "Outfall To Surface Waters"
        return pd.Series(dict(
            surface_water_discharge=bool(surf.any()),
            surface_water_pct=float(g.loc[surf, "PRESENT_DISCHARGE_PERCENTAGE"].sum()),
            ocean_discharge=bool((g["DISCHARGE_TYPE"] == "Ocean Discharge").any()),
            cso_discharge=bool((g["DISCHARGE_TYPE"] == "CSO Discharge").any()),
            requires_npdes=bool(g["DISCHARGE_TYPE"].isin([
                "Outfall To Surface Waters", "Ocean Discharge",
                "CSO Discharge", "Overland Flow With Discharge"]).any()),
            any_reuse=bool(g["DISCHARGE_TYPE"].str.startswith("Reuse:", na=False).any()),
            n_discharge_types=g["DISCHARGE_TYPE"].nunique(),
        ))

    return df.groupby("CWNS_ID").apply(agg, include_groups=False).reset_index()


def build_population_features(treatment_ids: set) -> pd.DataFrame:
    df = pd.read_csv(C.CWNS_DIR / "POPULATION_WASTEWATER.txt", dtype={"CWNS_ID": str}, encoding="latin1")
    df = df[df["CWNS_ID"].isin(treatment_ids)]
    df = df[["CWNS_ID", "TOTAL_RES_POPULATION_2022"]].rename(
        columns={"TOTAL_RES_POPULATION_2022": "pop_served"})
    return df.drop_duplicates(subset="CWNS_ID")


def _clean_geo(series: pd.Series, remove_pattern: str) -> pd.Series:
    return (series.str.lower()
            .str.replace(remove_pattern, "", regex=True)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True))


def build_census_features(plants: pd.DataFrame) -> pd.DataFrame:
    """Spatial join to county subdivision / place / county, matching the R
    version's three separate st_intersection calls + name-cleaning regex."""
    if not C.CENSUS_GDB.exists():
        print(f"  WARNING: {C.CENSUS_GDB} not found -- skipping census features "
              f"(subdivision/place/county/is_rural will be NaN)")
        out = plants[["CWNS_ID"]].copy()
        for col in ["subdivision", "place", "county", "county_geoid"]:
            out[col] = None
        out["is_rural"] = None
        return out

    plants_sf = gpd.GeoDataFrame(
        plants[["CWNS_ID"]],
        geometry=gpd.points_from_xy(plants["LONGITUDE"], plants["LATITUDE"]),
        crs=4269,
    ).to_crs(5070)

    sd_remove = r"cdp|municipio|municipality|parish|city|st\.|borough|district|census area|town|village|ccd|township|barrio|precinct|census subarea"
    place_remove = r"cdp|municipio|municipality|parish|city|st\.|borough|district|census area|town|village"
    county_remove = r"county|municipio|municipality|parish|city|st\.|borough|district|census area"

    print("  Loading census geographies...")
    subdivisions = gpd.read_file(C.CENSUS_GDB, layer="County_Subdivision")[["NAMELSAD", "geometry"]].to_crs(5070)
    subdivisions["geometry"] = subdivisions.buffer(0)   # st_make_valid equivalent
    places_inc = gpd.read_file(C.CENSUS_GDB, layer="Incorporated_Place")[["NAMELSAD", "geometry"]].to_crs(5070)
    places_cdp = gpd.read_file(C.CENSUS_GDB, layer="Census_Designated_Place")[["NAMELSAD", "geometry"]].to_crs(5070)
    places = pd.concat([places_inc, places_cdp], ignore_index=True)
    places = gpd.GeoDataFrame(places, geometry=places.buffer(0), crs=5070)
    counties = gpd.read_file(C.CENSUS_GDB, layer="County")[["NAMELSAD", "GEOID", "geometry"]].to_crs(5070)
    counties["geometry"] = counties.buffer(0)

    print("  Running census spatial joins...")
    sd_join = gpd.sjoin(plants_sf, subdivisions, predicate="intersects", how="inner")[["CWNS_ID", "NAMELSAD"]]
    sd_join["subdivision"] = _clean_geo(sd_join["NAMELSAD"], sd_remove)
    sd_join = sd_join.drop_duplicates(subset="CWNS_ID")[["CWNS_ID", "subdivision"]]

    place_join = gpd.sjoin(plants_sf, places, predicate="intersects", how="inner")[["CWNS_ID", "NAMELSAD"]]
    place_join["place"] = _clean_geo(place_join["NAMELSAD"], place_remove)
    place_join = place_join.drop_duplicates(subset="CWNS_ID")[["CWNS_ID", "place"]]

    county_join = gpd.sjoin(plants_sf, counties, predicate="intersects", how="inner")[["CWNS_ID", "NAMELSAD", "GEOID"]]
    county_join["county"] = _clean_geo(county_join["NAMELSAD"], county_remove)
    county_join = county_join.drop_duplicates(subset="CWNS_ID")[["CWNS_ID", "county"]].rename(
        columns={}).assign(county_geoid=county_join["GEOID"].values)

    out = plants[["CWNS_ID"]].merge(sd_join, on="CWNS_ID", how="left") \
        .merge(place_join, on="CWNS_ID", how="left") \
        .merge(county_join, on="CWNS_ID", how="left")
    out["is_rural"] = out["place"].isna() | (out["place"].fillna("").str.len() == 0)
    return out


def build_plant_features(states, training_only) -> pd.DataFrame:
    print("PART 1: Building plant features")
    plants, treatment_ids = load_treatment_plants(states, training_only)
    print(f"  Plants loaded: {len(plants)}")

    discharge = build_discharge_features(treatment_ids)
    population = build_population_features(treatment_ids)
    census = build_census_features(plants)

    plant_features = plants.merge(discharge, on="CWNS_ID", how="left") \
        .merge(population, on="CWNS_ID", how="left") \
        .merge(census[["CWNS_ID", "subdivision", "place", "county", "county_geoid", "is_rural"]],
               on="CWNS_ID", how="left")

    # Defensive: STATE_CODE has been observed to drop out of this merge chain
    # on real data (root cause not yet isolated -- possibly dtype mismatch or
    # NaN-driven coercion in one of the intermediate frames). Re-attach it
    # explicitly from `plants` rather than trusting it survived the chain, so
    # downstream code (build_stage1_training, build_stage2_training, the
    # --states filter in main()) doesn't silently break on a KeyError.
    if "STATE_CODE" not in plant_features.columns:
        print("  WARNING: STATE_CODE dropped out of the plant_features merge chain -- "
              "re-attaching from source. Root cause not yet diagnosed, worth revisiting.")
        plant_features = plant_features.merge(
            plants[["CWNS_ID", "STATE_CODE"]], on="CWNS_ID", how="left")
    assert "STATE_CODE" in plant_features.columns, \
        "STATE_CODE still missing from plant_features after re-attach -- check plants df itself"

    print(f"  Plant features combined: {len(plant_features)} rows, {len(plant_features.columns)} cols")
    return plant_features


# ===========================================================================
# PART 2: Parcel-level features (LBCS, owner, OSM, county data quality)
# ===========================================================================
def fetch_parcel_attrs(con, state: str, uuids: list[str], chunk_size=5000) -> pd.DataFrame:
    chunks = [uuids[i:i + chunk_size] for i in range(0, len(uuids), chunk_size)]
    results = []
    for chunk in chunks:
        uuid_filter = ", ".join(f"'{u}'" for u in chunk)
        try:
            df = con.execute(f"""
                SELECT ll_uuid, owner, ll_gisacre, ll_bldg_count,
                       lbcs_activity_desc, lbcs_function_desc,
                       lbcs_structure_desc, lbcs_site_desc,
                       lbcs_ownership_desc, zoning_type, zoning_subtype, geoid
                FROM read_parquet('{C.PARCEL_BASE.as_posix()}/state={state}/*.parquet')
                WHERE ll_uuid IN ({uuid_filter})
            """).df()
            results.append(df)
        except Exception as e:
            print(f"    Chunk fetch failed: {e}")
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def engineer_parcel_attrs(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["lbcs_activity"] = df["lbcs_activity_desc"].apply(reclass_activity)
    df["lbcs_function"] = df["lbcs_function_desc"].apply(reclass_function)
    df["lbcs_structure"] = df["lbcs_structure_desc"].fillna("Unknown")
    df["lbcs_site"] = df["lbcs_site_desc"].fillna("Unknown")
    df["lbcs_ownership"] = df["lbcs_ownership_desc"].apply(reclass_ownership)
    df["zoning_type"] = df["zoning_type"].fillna("unknown")

    df["log_gisacre"] = np.log1p(df["ll_gisacre"].fillna(0))
    df["bldg_per_acre"] = np.where(
        df["ll_gisacre"].isna() | (df["ll_gisacre"] == 0), np.nan,
        df["ll_bldg_count"].fillna(0) / df["ll_gisacre"])

    combined_text = (
        df["zoning_type"].fillna("") + " " + df["zoning_subtype"].fillna("") + " " +
        df["owner"].fillna("") + " " + df["lbcs_ownership_desc"].fillna("") + " " +
        df["lbcs_activity_desc"].fillna("") + " " + df["lbcs_function_desc"].fillna("") + " " +
        df["lbcs_structure_desc"].fillna("") + " " + df["lbcs_site_desc"].fillna("")
    )
    df["has_ww_keyword"] = combined_text.str.contains(PARCEL_WW_PATTERN, na=False)

    owner_clean = df["owner"].fillna("").str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
    df["owner_clean"] = owner_clean
    df["owner_is_person"] = (
        owner_clean.str.contains(r"\b(etux|sfr|borrower|surv)\b", regex=True) |
        owner_clean.str.match(r"^[a-z]+ [a-z]+$")
    ).fillna(False)
    df["owner_is_govt"] = owner_clean.str.contains(
        r"\b(city|county|municipality|municipal|district|authority|auth|township|borough|"
        r"state|federal|dept|department|board|commission|muni|pub|wtr|commonw|commwlth)\b",
        regex=True).fillna(False)
    df["owner_is_utility"] = owner_clean.str.contains(
        r"\b(water|sewer|wastewater|utility|utilities|sanitary|sanitation|treatment|"
        r"wwtp|wsd|msd|puc|pwd|mwrd|wpcp|potw|sewerage|wtf|wwtf)\b",
        regex=True).fillna(False)
    df["owner_is_electric"] = owner_clean.str.contains(
        r"\b(electric|elec|power|energy|entergy|grid|verizon|telecom|telephone|tel|tele|"
        r"hydropower|centerpoint|oncor|niagara|nyseg|xcel)\b",
        regex=True).fillna(False)
    df["owner_has_llc_corp"] = owner_clean.str.contains(
        r"\b(llc|inc|corp|ltd|lp|company|association|trust|partners|partnership|holdings|"
        r"ventures|properties|realty|development|developer|builders|homes|lennar|pulte|"
        r"horton|forestar)\b",
        regex=True).fillna(False)

    return df[["ll_uuid", "geoid", "owner", "owner_clean",
               "ll_gisacre", "log_gisacre", "ll_bldg_count", "bldg_per_acre",
               "lbcs_activity", "lbcs_function", "lbcs_structure",
               "lbcs_site", "lbcs_ownership", "zoning_type",
               "owner_is_person", "owner_is_govt", "owner_is_utility",
               "owner_is_electric", "owner_has_llc_corp", "has_ww_keyword"]]


def build_osm_features(con, state: str, target_uuids: pd.Series) -> pd.DataFrame:
    if not C.OSM_PATH.exists():
        print(f"  WARNING: {C.OSM_PATH} not found -- osm_ww will be False for all parcels")
        return pd.DataFrame({"ll_uuid": target_uuids, "osm_ww": False})

    osm_pts = gpd.read_file(C.OSM_PATH, layer="Points").to_crs(4326)
    osm_pts["h3_index"] = osm_pts.geometry.apply(
        lambda g: h3.latlng_to_cell(g.y, g.x, 9))
    osm_upload = pd.DataFrame({
        "h3_index": osm_pts["h3_index"],
        "geom_wkb": osm_pts.geometry.apply(lambda g: g.wkb),
    })
    con.register("osm_points", osm_upload)
    try:
        match = con.execute(f"""
            SELECT DISTINCT p.ll_uuid
            FROM read_parquet('{C.PARCEL_BASE.as_posix()}/state={state}/*.parquet') p
            INNER JOIN osm_points o ON p.h3_index_9 = o.h3_index
            WHERE ST_Intersects(ST_GeomFromWKB(p.wkb_geometry), ST_GeomFromWKB(o.geom_wkb))
        """).df()
    except Exception as e:
        print(f"  OSM join failed: {e}")
        match = pd.DataFrame(columns=["ll_uuid"])
    finally:
        con.unregister("osm_points")

    matched = set(match["ll_uuid"])
    return pd.DataFrame({"ll_uuid": target_uuids, "osm_ww": target_uuids.isin(matched)})


def build_parcel_features_for_state(con, state: str) -> pd.DataFrame:
    nlcd_path = C.NLCD_OUTPUT_DIR / f"nlcd_{state}_k{C.K_RINGS}.parquet"
    if not nlcd_path.exists():
        print(f"  No NLCD output for {state} (run 01a first) -- skipping")
        return pd.DataFrame()

    nlcd_raw = pd.read_parquet(nlcd_path)
    nlcd = nlcd_raw.copy()
    nlcd["water_pixel_pct"] = np.where(
        nlcd["total_pixels"] > 0, nlcd["water_pixels"] / nlcd["total_pixels"], 0.0)
    nlcd["dominant_class_group"] = nlcd["dominant_class"].apply(reclass_dominant_class)
    nlcd = nlcd[["ll_uuid", "h3_index_9", "state", "total_pixels", "water_pixels",
                 "water_pixel_pct", "dominant_class_group", "dominant_count", "has_water"]]
    nlcd = nlcd.drop_duplicates(subset="ll_uuid")

    target_uuids = nlcd["ll_uuid"].tolist()
    print(f"  Fetching parcel attributes for {len(target_uuids)} parcels...")
    raw_attrs = fetch_parcel_attrs(con, state, target_uuids)
    parcel_attrs = engineer_parcel_attrs(raw_attrs) if len(raw_attrs) else pd.DataFrame()
    print(f"  has_ww_keyword TRUE: {parcel_attrs['has_ww_keyword'].sum() if len(parcel_attrs) else 0}")

    print("  Computing OSM wastewater features...")
    osm_features = build_osm_features(con, state, nlcd["ll_uuid"])
    print(f"  Parcels with OSM wastewater tag: {osm_features['osm_ww'].sum()}")

    if len(parcel_attrs):
        county_dq = parcel_attrs.groupby("geoid").apply(
            lambda g: pd.Series(dict(
                n_parcels=len(g),
                pct_lbcs_activity_known=(g["lbcs_activity"] != "Unknown").mean(),
                pct_lbcs_owner_known=(g["lbcs_ownership"] != "Other / Unknown").mean(),
                pct_owner_known=(g["owner_clean"].str.len() > 0).mean(),
                pct_zoning_known=(g["zoning_type"] != "unknown").mean(),
            )), include_groups=False,
        ).reset_index()
        county_dq["data_quality_score"] = (
            county_dq["pct_lbcs_activity_known"] + county_dq["pct_lbcs_owner_known"] +
            county_dq["pct_owner_known"] + county_dq["pct_zoning_known"]) / 4
    else:
        county_dq = pd.DataFrame(columns=["geoid", "n_parcels", "data_quality_score",
                                          "pct_lbcs_activity_known", "pct_owner_known"])

    out = nlcd.merge(
        parcel_attrs.drop(columns=["owner_clean"]) if len(parcel_attrs) else parcel_attrs,
        on="ll_uuid", how="left"
    ).merge(osm_features, on="ll_uuid", how="left") \
     .merge(county_dq[["geoid", "n_parcels", "data_quality_score",
                       "pct_lbcs_activity_known", "pct_owner_known"]],
            on="geoid", how="left")

    cols = ["ll_uuid", "h3_index_9", "state", "geoid",
            "total_pixels", "water_pixels", "water_pixel_pct",
            "dominant_class_group", "dominant_count", "has_water",
            "ll_gisacre", "log_gisacre", "ll_bldg_count", "bldg_per_acre",
            "lbcs_activity", "lbcs_function", "lbcs_structure",
            "lbcs_site", "lbcs_ownership", "zoning_type",
            "owner", "owner_is_person", "owner_is_govt",
            "owner_is_utility", "owner_is_electric", "owner_has_llc_corp",
            "has_ww_keyword", "osm_ww",
            "data_quality_score", "n_parcels",
            "pct_lbcs_activity_known", "pct_owner_known"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]


# ===========================================================================
# PART 3: Stage 1 / Stage 2 training table assembly
# ===========================================================================
def _is_missing(v) -> bool:
    """True for None, NaN, NaT, pd.NA.

    Written out rather than relying on falsiness because NaN is TRUTHY --
    `not float('nan')` is False -- which is exactly how a NaN reached
    re.escape() and crashed 05_run_inference.py on 2026-09-06.
    """
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        # pd.isna returns an array for list-likes; those aren't missing.
        return False


def _geo_term(v) -> str | None:
    """A geo term as a non-empty stripped string, or None.

    Coerces non-strings (a numeric county code, say) via str() so callers can
    hand the result straight to re.escape without another type check.
    """
    if _is_missing(v):
        return None
    s = str(v).strip()
    return s or None


def add_name_matching(df: pd.DataFrame) -> pd.DataFrame:
    owner_clean = df["owner"].fillna("").str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
    geo_terms = (df["subdivision"].fillna("") + "|" + df["place"].fillna("") + "|" + df["county"].fillna(""))
    geo_terms = geo_terms.str.replace(r"\|{2,}", "|", regex=True).str.strip("|")

    owner_base = r"\b(" + "|".join(C.OWNER_KEYWORDS) + r")\b"
    ww_base = r"\b(" + "|".join(C.WW_KEYWORDS) + r")\b"

    def build_pattern(base_keywords, geo):
        if len(geo) > 0:
            return r"\b(" + "|".join(base_keywords) + "|" + re.escape(geo) + r")\b"
        return None

    is_municipal, owner_water, sd_match, place_match, county_match = [], [], [], [], []
    for i, row in df.iterrows():
        oc = owner_clean.loc[i]
        gt = geo_terms.loc[i]
        op = build_pattern(C.OWNER_KEYWORDS, gt) or owner_base
        wp = build_pattern(C.WW_KEYWORDS, gt) or ww_base
        try:
            is_municipal.append(bool(re.search(op, oc, re.IGNORECASE)))
        except re.error:
            is_municipal.append(bool(re.search(owner_base, oc, re.IGNORECASE)))
        try:
            owner_water.append(bool(re.search(wp, oc, re.IGNORECASE)))
        except re.error:
            owner_water.append(bool(re.search(ww_base, oc, re.IGNORECASE)))

        # A missing geo term must become None, not reach re.escape().
        #
        # The previous guard was `None if not sd else ...`, which is wrong for
        # NaN: bool(float('nan')) is True, so `not sd` is False and the NaN
        # went straight into re.escape(), which then tried str(nan, 'latin1')
        # and raised TypeError. It only ever worked because a column with SOME
        # string values is object dtype and its gaps are None (correctly
        # falsy). A column that is entirely empty gets float64 dtype and every
        # gap is np.float64('nan') -- truthy. 02 never tripped it; 05 did, on
        # 2026-09-06, running 48 states where at least one geo column came
        # back wholly empty.
        sd = _geo_term(row.get("subdivision"))
        sd_match.append(None if sd is None
                        else bool(re.search(re.escape(sd), oc, re.IGNORECASE)))
        pl = _geo_term(row.get("place"))
        # is_rural NaN means unknown; treated as rural, matching the
        # .fillna(True) used when any_geo_match is assembled below.
        rural = row.get("is_rural")
        rural = True if _is_missing(rural) else bool(rural)
        place_match.append(None if (rural or pl is None)
                           else bool(re.search(re.escape(pl), oc, re.IGNORECASE)))
        co = _geo_term(row.get("county"))
        county_match.append(False if co is None
                            else bool(re.search(re.escape(co), oc, re.IGNORECASE)))

    df = df.copy()
    df["is_municipal"] = is_municipal
    df["owner_water"] = owner_water
    df["sd_match"] = sd_match
    df["place_match"] = place_match
    df["county_match"] = county_match
    df["any_geo_match"] = (
        df["county_match"].fillna(False) |
        df["sd_match"].apply(lambda v: bool(v) if v is not None else False) |
        ((~df["is_rural"].fillna(True)) & df["place_match"].apply(lambda v: bool(v) if v is not None else False))
    )
    return df.drop(columns=["owner"])


def point_in_parcel_lookup(con, state: str, pts: pd.DataFrame, extra_select="") -> pd.DataFrame:
    """pts must have: geom_wkb, plus whatever extra_select references.

    NOTE (2026-08-21): this used to prefilter with
    `INNER JOIN pts ON p.h3_index_9 = pts.h3_res9` before the ST_Intersects
    check -- an unsound shortcut. A parcel's stored h3_index_9 is a SINGLE
    cell (~0.1 km2), so any parcel larger than that has real area sitting in
    OTHER H3 cells too; a point genuinely inside such a parcel but landing in
    a different cell than the parcel's one indexed cell would be silently
    rejected before ST_Intersects was ever evaluated. Confirmed on real data:
    of 227 Stage 1 training rows dropped from the training data, 92 (41%)
    turned out to genuinely intersect a real parcel once checked directly --
    they were false negatives from this prefilter, not truly unmatched.
    01b_run_object_detection.py's find_containing_parcels never had this
    issue (direct ST_Intersects join, no H3 shortcut) -- matching that here."""
    con.register("pts", pts)
    try:
        return con.execute(f"""
            SELECT pts.*, p.ll_uuid {extra_select}
            FROM read_parquet('{C.PARCEL_BASE.as_posix()}/state={state}/*.parquet') p
            JOIN pts ON ST_Intersects(ST_GeomFromWKB(p.wkb_geometry), ST_GeomFromWKB(pts.geom_wkb))
        """).df()
    except Exception as e:
        print(f"    Point-in-parcel lookup failed for {state}: {e}")
        return pd.DataFrame()
    finally:
        con.unregister("pts")


def build_stage1_training(con, plant_features: pd.DataFrame, parcel_features: pd.DataFrame,
                           od_features: pd.DataFrame | None) -> pd.DataFrame:
    print("\nSTEP 14: Building Stage 1 training data")
    classes = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CLASSES)
    classes["CWNS_ID"] = classes["CWNS_ID"].astype(str)
    classes = classes.to_crs(4326)
    classes["h3_res9"] = classes.geometry.apply(lambda g: h3.latlng_to_cell(g.y, g.x, 9))
    classes = classes.merge(plant_features[["CWNS_ID", "STATE_CODE"]], on="CWNS_ID", how="left")
    print(f"  Classes: {len(classes)}")

    reported = []
    for state, grp in classes.dropna(subset=["STATE_CODE"]).groupby("STATE_CODE"):
        pts = pd.DataFrame({
            "CWNS_ID": grp["CWNS_ID"], "class": grp["class"], "h3_res9": grp["h3_res9"],
            "geom_wkb": grp.geometry.apply(lambda g: g.wkb),
        })
        matched = point_in_parcel_lookup(con, state, pts)
        if len(matched):
            reported.append(matched)
    reported_parcels = pd.concat(reported, ignore_index=True) if reported else pd.DataFrame()
    reported_parcels = reported_parcels.drop_duplicates(subset="CWNS_ID")
    print(f"  Reported parcels found: {len(reported_parcels)}")

    stage1 = reported_parcels.merge(
        # 03_train_stage1.py re-merges LATITUDE/LONGITUDE itself, fresh from
        # PHYSICAL_LOCATION.txt (see its own "Coordinates for spatial CV"
        # section) -- it assumes this parquet does NOT already carry those
        # columns. plant_features legitimately needs them upstream (Part 1),
        # but they must not survive into this output, or 03's later merge
        # collides on the name and pandas silently renames both sides to
        # LATITUDE_x/LATITUDE_y, leaving no plain "LATITUDE" column at all --
        # confirmed 2026-08-21, this crashed 03 with a KeyError.
        plant_features.drop(columns=["STATE_CODE", "LATITUDE", "LONGITUDE"], errors="ignore"),
        on="CWNS_ID", how="left"
    ).merge(
        parcel_features.drop(columns=["state"], errors="ignore"), on="ll_uuid", how="left"
    )
    stage1 = add_name_matching(stage1)

    if od_features is not None and len(od_features):
        n_before = len(stage1)
        stage1 = stage1.merge(od_features, on="CWNS_ID", how="left")
        stage1["od_ran"] = stage1["od_ran"].fillna(False)
        stage1["od_has_detection"] = stage1["od_has_detection"].fillna(False)
        for c in [c for c in stage1.columns if c.startswith("od_has_")]:
            stage1[c] = stage1[c].fillna(False)
        for c in [c for c in stage1.columns if c.startswith("od_n_")]:
            stage1[c] = stage1[c].fillna(0)
        assert len(stage1) == n_before, "OD join changed row count -- check for duplicate CWNS_IDs in od_features"
        print(f"  OD features joined -- od_ran True for {stage1['od_ran'].sum()} / {len(stage1)}")
    else:
        print("  No OD features available (01b hasn't produced output yet) -- skipping OD join")

    print(f"  Stage 1 training rows: {len(stage1)}")
    print(f"  Correct: {(stage1['class'] == 'Correct').sum()}")
    print(f"  Incorrect: {(stage1['class'] == 'Incorrect').sum()}")
    return stage1


def build_stage2_training(con, plant_features: pd.DataFrame, parcel_features: pd.DataFrame,
                           states: list[str] | None) -> pd.DataFrame:
    print("\nSTEP 15: Building Stage 2 training data")
    corrections = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CORRECTIONS).to_crs(4326)
    corrections["CWNS_ID"] = corrections["CWNS_ID"].astype(str)

    corrected_sf = gpd.GeoDataFrame(
        corrections[["CWNS_ID"]],
        geometry=gpd.points_from_xy(corrections["Corrected_X"], corrections["Corrected_Y"]),
        crs=4269,
    ).to_crs(4326)
    corrected_sf["h3_res9"] = corrected_sf.geometry.apply(lambda g: h3.latlng_to_cell(g.y, g.x, 9))

    corrected_matches = []
    for state, grp in plant_features.groupby("STATE_CODE"):
        cwns_in_state = set(grp["CWNS_ID"])
        pts_df = corrected_sf[corrected_sf["CWNS_ID"].isin(cwns_in_state)]
        if len(pts_df) == 0:
            continue
        pts = pd.DataFrame({
            "CWNS_ID": pts_df["CWNS_ID"], "h3_res9": pts_df["h3_res9"],
            "geom_wkb": pts_df.geometry.apply(lambda g: g.wkb),
        })
        matched = point_in_parcel_lookup(con, state, pts)
        if len(matched):
            matched = matched.rename(columns={"ll_uuid": "corrected_ll_uuid"})
            corrected_matches.append(matched)
    corrected_parcels = pd.concat(corrected_matches, ignore_index=True) if corrected_matches \
        else pd.DataFrame(columns=["CWNS_ID", "corrected_ll_uuid"])
    corrected_parcels = corrected_parcels.drop_duplicates(subset="CWNS_ID")
    print(f"  Corrected parcels found: {len(corrected_parcels)}")

    corrected_cwns = corrected_parcels["CWNS_ID"].tolist()
    stage2_plants = plant_features[plant_features["CWNS_ID"].isin(corrected_cwns)]
    print(f"  Plants for Stage 2: {len(stage2_plants)}")

    corrected_coords = corrections[["CWNS_ID", "Original_X", "Original_Y"]]

    rows = []
    for state in stage2_plants["STATE_CODE"].dropna().unique():
        nlcd_path = C.NLCD_OUTPUT_DIR / f"nlcd_{state}_k{C.K_RINGS}.parquet"
        if not nlcd_path.exists():
            print(f"  No 01a output for {state} -- Stage 2 candidates unavailable for its plants")
            continue
        state_candidates = pd.read_parquet(nlcd_path, columns=["ll_uuid", "h3_index_9"])
        state_candidates = state_candidates.merge(
            parcel_features.drop(columns=["state", "h3_index_9"], errors="ignore"),
            on="ll_uuid", how="left")

        for _, plant in stage2_plants[stage2_plants["STATE_CODE"] == state].iterrows():
            cwns_id = plant["CWNS_ID"]
            correct_matches = corrected_parcels.loc[corrected_parcels["CWNS_ID"] == cwns_id, "corrected_ll_uuid"]
            if len(correct_matches) == 0:
                continue
            correct_id = correct_matches.iloc[0]
            coords = corrected_coords[corrected_coords["CWNS_ID"] == cwns_id]
            if len(coords) == 0:
                continue

            plant_h3_cells = set(h3.grid_disk(plant["h3_res9"], C.K_RINGS))
            candidates = state_candidates[state_candidates["h3_index_9"].isin(plant_h3_cells)].copy()
            if len(candidates) == 0:
                continue

            centers = candidates["h3_index_9"].apply(h3.cell_to_latlng)
            candidates["centroid_lat"] = centers.apply(lambda c: c[0])
            candidates["centroid_lng"] = centers.apply(lambda c: c[1])

            cand_sf = gpd.GeoDataFrame(
                candidates, geometry=gpd.points_from_xy(candidates["centroid_lng"], candidates["centroid_lat"]),
                crs=4326).to_crs(5070)
            reported_pt = gpd.GeoSeries(
                [gpd.points_from_xy([coords["Original_X"].iloc[0]], [coords["Original_Y"].iloc[0]])[0]],
                crs=4269).to_crs(5070).iloc[0]
            distances = cand_sf.geometry.distance(reported_pt).to_numpy()

            candidates["CWNS_ID"] = cwns_id
            candidates["label"] = (candidates["ll_uuid"] == correct_id).astype(int)
            candidates["is_reported"] = False   # reported-parcel flag joined in below, if present
            candidates["distance_m"] = distances
            candidates["log_distance"] = np.log1p(distances)
            candidates["distance_ring"] = np.floor(distances / 330).astype(int)
            candidates["within_1km"] = distances <= 1000
            candidates["within_5km"] = distances <= 5000
            rows.append(candidates)

    if rows:
        stage2 = pd.concat(rows, ignore_index=True)
    else:
        # Empty-but-correctly-shaped fallback for a state/run with zero
        # Stage 2 candidates (e.g. AK: 0 corrections at all). The per-state
        # loop above is what normally brings in parcel-level columns like
        # 'owner'/'subdivision'/'place'/'county' (via state_candidates,
        # which already has parcel_features merged in) plus the explicitly
        # -added label/distance columns -- none of that runs when there are
        # no candidates, so build the same column set here directly from
        # parcel_features's own schema (already a parameter to this
        # function) rather than an ad-hoc guess. Missing 'owner' specifically
        # crashed add_name_matching() below on AK's real zero-correction case.
        explicit_cols = ["CWNS_ID", "label", "is_reported", "distance_m",
                          "log_distance", "distance_ring", "within_1km", "within_5km"]
        pf_cols = [c for c in parcel_features.columns if c not in ("state", "h3_index_9")]
        cols = pf_cols + [c for c in explicit_cols if c not in pf_cols]
        stage2 = pd.DataFrame(columns=cols)
    print(f"  Stage 2 raw candidates labeled: {len(stage2)}")

    # 04_train_stage2.py re-merges LATITUDE/LONGITUDE itself, fresh from
    # PHYSICAL_LOCATION.txt -- same reasoning as build_stage1_training above.
    # Must not let plant_features' copies survive into this output either,
    # or 04 hits the identical merge-collision crash.
    stage2 = stage2.merge(
        stage2_plants.drop(columns=["STATE_CODE", "LATITUDE", "LONGITUDE"], errors="ignore"),
        on="CWNS_ID", how="left")
    stage2 = add_name_matching(stage2)

    print(f"  Stage 2 training rows: {len(stage2)}")
    if len(stage2):
        print(f"  Positive labels: {(stage2['label'] == 1).sum()}")
        print(f"  Negative labels: {(stage2['label'] == 0).sum()}")
    return stage2


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=str, default=None)
    ap.add_argument("--full-universe", action="store_true")
    args = ap.parse_args()
    states = [s.strip() for s in args.states.split(",")] if args.states else None

    C.ensure_dirs()
    print("=== 02_feature_engineering.py ===\n")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; SET enable_geoparquet_conversion = false;")

    # ---- PART 1 ----
    plant_features = build_plant_features(states, training_only=not args.full_universe)
    plant_features.to_parquet(C.FEATURES_OUTPUT_DIR / "05_plant_features.parquet", index=False)

    # ---- PART 2 ----
    print("\nPART 2: Building parcel features")
    state_list = states or sorted(plant_features["STATE_CODE"].dropna().unique())

    # Write each state's parcel features to its own file immediately, rather
    # than accumulating a growing list of full per-state DataFrames in memory
    # (parcel_dfs.append(pf) for every state, only concatenated at the very
    # end -- for a nationwide run across 52 states/2382 plants, extrapolating
    # from Ohio's 935k rows for 84 plants implies ~20-25M+ rows held
    # SIMULTANEOUSLY before the final concat, which roughly doubles peak
    # memory again on top of that. Same per-state-file pattern already used
    # by 01a/01b -- write and release each state before moving to the next,
    # union them back with a single-file-at-a-time read afterward (which also
    # sidesteps the pyarrow whole-directory schema-merge crash class seen
    # elsewhere in this pipeline).
    #
    # IMPORTANT (2026-08-25): this directory is a PERSISTENT CACHE, not a
    # scratch dir for this run alone. mkdir(..., exist_ok=True) never clears
    # it, and it accumulates one file per state ever processed across every
    # invocation of this script (52 files from the 2026-08-21 nationwide
    # training run were still here, untouched, when the OH/MS/DE pilot ran).
    # That's deliberate and useful -- it means re-running a state you've
    # already built is cheap, and a later nationwide run can reuse work from
    # today's pilot. But it means the union below MUST filter to state_list
    # explicitly. Globbing every "*.parquet" in the directory (the original
    # code) silently pulls in every state ever cached, not just the ones this
    # run asked for -- confirmed 2026-08-25: a 3-state full-universe run
    # produced a 26.65M-row combined table against 4.87M candidate parcels
    # from 01a, a 5.47x inflation, because all 51 leftover training-run files
    # were sitting in this directory and got unioned in alongside the 3 new
    # ones. No duplicate ll_uuid values were involved anywhere -- every one
    # of those extra rows was a real, distinct parcel from a state nobody
    # asked for in this run. Do NOT try to fix this by deleting old files
    # instead of filtering the glob: those files are legitimate cached work
    # another run may still want.
    parcel_by_state_dir = C.FEATURES_OUTPUT_DIR / "10_parcel_features_by_state"
    parcel_by_state_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for state in state_list:
        print(f"\n--- State {state} ---")
        pf = build_parcel_features_for_state(con, state)
        if len(pf):
            pf.to_parquet(parcel_by_state_dir / f"state={state}.parquet", index=False)
            n_written += 1
        del pf   # encourage the state's parcel-attribute frame to be released
                 # before the next iteration builds another one

    print(f"\nUnioning parcel feature files for this run's {len(state_list)} state(s)...")
    part_files = sorted(parcel_by_state_dir / f"state={st}.parquet" for st in state_list)
    missing = [f for f in part_files if not f.exists()]
    part_files = [f for f in part_files if f.exists()]
    if missing:
        print(f"  WARNING: {len(missing)} requested state(s) have no cached parcel-features "
              f"file and will be absent from the combined table (likely no NLCD/01a output "
              f"for them, or zero candidate parcels): "
              f"{[f.stem.replace('state=', '') for f in missing]}")
    n_cache_total = len(list(parcel_by_state_dir.glob("*.parquet")))
    if n_cache_total > len(part_files):
        print(f"  (Cache directory holds {n_cache_total} state file(s) total, from earlier "
              f"runs -- using only this run's {len(part_files)} requested state(s), as it should.)")
    parcel_frames = [pd.read_parquet(f) for f in part_files]
    parcel_features = pd.concat(parcel_frames, ignore_index=True).drop_duplicates(subset="ll_uuid") \
        if parcel_frames else pd.DataFrame()
    del parcel_frames
    parcel_features.to_parquet(C.FEATURES_OUTPUT_DIR / "10_parcel_features.parquet", index=False)
    print(f"\nParcel features combined: {len(parcel_features)} rows")

    # ---- OD features from 01b (Stage 1 only -- see module docstring) ----
    od_plants_dir = C.OD_OUTPUT_DIR / "plants"
    od_features = None
    if od_plants_dir.exists() and any(od_plants_dir.rglob("*.parquet")):
        # NOTE: this is a flush_parquet-style append-only output dir. Reading
        # the whole tree with pd.read_parquet() lets pyarrow try to infer ONE
        # unified schema across every part-file at once, which blows up when
        # a column that's all-None in one part-file (inferred as pyarrow's
        # 'null' type) meets a real string/dictionary-encoded version of the
        # same column in another (e.g. od_dominant_class is None for every
        # plant in a detection-free flush, but a real string once any flush
        # has a detection in it).
        #
        # IMPORTANT (2026-08-20): this dir is NOT safe to reduce to "newest
        # part-file per partition" -- an earlier version of this fix did
        # that, which was correct for output that gets fully REPLACED on
        # rerun, but 01b's actual design flushes every ~200 plants and
        # RESUMES across separate runs, so a state's partition can hold many
        # part-files that each contain DIFFERENT plants. Keeping only the
        # newest would silently drop every plant from every flush except the
        # last one. Every part-file must be read and UNIONED.
        part_files_by_partition = {}
        for f in od_plants_dir.rglob("part-*.parquet"):
            part_files_by_partition.setdefault(f.parent, []).append(f)
        n_total_files = sum(len(v) for v in part_files_by_partition.values())
        print(f"  OD plants: {n_total_files} part-file(s) found across "
              f"{len(part_files_by_partition)} partition(s); reading and unioning all of them")
        od_frames = []
        for files in part_files_by_partition.values():
            # Sort by mtime so that IF a duplicate CWNS_ID exists across
            # part-files (e.g. a plant processed once during a --limit test
            # and again in a later full run), the keep="last" dedup below
            # reliably keeps the most recently written row, not whichever
            # happened to come last in directory-listing order.
            for f in sorted(files, key=lambda f: f.stat().st_mtime):
                # Read ONE file at a time -- this is what avoids the
                # whole-directory schema-merge crash, since pyarrow only
                # needs to resolve a single file's own schema here, not
                # reconcile it against every other file up front.
                df_part = pd.read_parquet(f)
                # normalize dtypes defensively -- dictionary-encoded columns from
                # older/newer pyarrow writes have caused schema-merge failures before
                for col in df_part.select_dtypes(include=["category"]).columns:
                    df_part[col] = df_part[col].astype(str)
                od_frames.append(df_part)
        od_features = pd.concat(od_frames, ignore_index=True) if od_frames else pd.DataFrame()
        od_features["CWNS_ID"] = od_features["CWNS_ID"].astype(str)
        n_before_dedup = len(od_features)
        od_features = od_features.drop_duplicates(subset="CWNS_ID", keep="last")
        if len(od_features) < n_before_dedup:
            print(f"  {n_before_dedup - len(od_features)} duplicate CWNS_ID rows across "
                  f"part-files (same plant processed more than once, e.g. a --limit test "
                  f"followed by a full run) -- kept the most recently written row for each")
        od_features = od_features.drop(columns=["state", "orig_lon", "orig_lat",
                                                  "n_objects_total", "n_objects_in_parcel",
                                                  "processed_at"], errors="ignore")

    # ---- PART 3 ----
    stage1 = build_stage1_training(con, plant_features, parcel_features, od_features)
    stage1.to_parquet(C.FEATURES_OUTPUT_DIR / "14_stage1_training.parquet", index=False)

    stage2 = build_stage2_training(con, plant_features, parcel_features, states)
    stage2.to_parquet(C.FEATURES_OUTPUT_DIR / "15_stage2_training.parquet", index=False)

    con.close()
    print(f"\n=== 02_feature_engineering.py complete ===")
    print(f"Outputs in: {C.FEATURES_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
