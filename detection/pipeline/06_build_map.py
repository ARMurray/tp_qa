"""
06_build_map.py
================
Builds a self-contained MapLibre GL JS web map from the STATE PIPELINE's
Parquet outputs (07_run_state_pipeline_hpc.py): detections/state={fips}/*.parquet
and plants/state={fips}/*.parquet.

Aggregates EVERY part-*.parquet file found under --inference-dir/detections
and --inference-dir/plants. Workflow: download the inference_state folder
(or just the states you have so far) into one local folder, run this script,
and the map reflects whatever's there -- rerun after downloading more states
and it picks up the new ones automatically, no manual merging needed.

Facility locations come from the same plants.gpkg (CWNS_ID + point geometry)
used to center tiles in the pipeline, filtered down to whichever CWNS_IDs
have a row in the `plants` table currently on disk.

Requires: geopandas, pandas, pyarrow
Usage:
    python 06_build_map.py
    python 06_build_map.py --inference-dir data/inference_state --plants-gpkg data/plants.gpkg
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

import pandas as pd
import geopandas as gpd

# Matches the Label Studio labeling interface colors (see docs).
CLASS_COLORS = {
    "clarifier":         "#FF0000",
    "aeration_basin":    "#0000FF",
    "digester":          "#FF8800",
    "oxidation_pond":    "#00AA00",
    "drying_bed":        "#FFA39E",
    "chlorine_contact":  "#1cf2d9",
}

# Vendored locally (pipeline/vendor/) and inlined directly into the generated
# HTML — this network blocks unpkg.com, so a <script src="https://..."> tag
# never loads. Inlining means the output page needs ZERO outbound requests.
VENDOR_DIR      = Path(__file__).resolve().parent / "vendor"
MAPLIBRE_JS     = VENDOR_DIR / "maplibre-gl.js"
MAPLIBRE_CSS    = VENDOR_DIR / "maplibre-gl.css"

# Two basemaps, BOTH raster tiles from the same Esri host (no API key). They
# live together in ONE style and are toggled by visibility — the map style is
# never swapped, so the data layers are added once and never disappear.
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services"
STREETS_TILES = f"{ESRI}/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}"
IMAGERY_TILES = f"{ESRI}/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}"
ESRI_ATTRIBUTION = "Esri, Maxar, Earthstar Geographics, and the GIS community"

BASE_STYLE = {
    "version": 8,
    "sources": {
        "streets": {"type": "raster", "tiles": [STREETS_TILES],
                    "tileSize": 256, "attribution": ESRI_ATTRIBUTION},
        "imagery": {"type": "raster", "tiles": [IMAGERY_TILES],
                    "tileSize": 256, "attribution": ESRI_ATTRIBUTION},
    },
    "layers": [
        {"id": "streets-layer", "type": "raster", "source": "streets",
         "layout": {"visibility": "visible"}},
        {"id": "imagery-layer", "type": "raster", "source": "imagery",
         "layout": {"visibility": "none"}},
    ],
}

OFFLINE_STYLE = {
    "version": 8,
    "sources": {},
    "layers": [
        {"id": "background", "type": "background",
         "paint": {"background-color": "#eef2f5"}}
    ],
}

# ===========================================================================
# Data loading -- reads the tiles/detections/objects/plants Parquet tables
# written by 07_run_state_pipeline_hpc.py (data/inference_state/{table}/state={fips}/*.parquet)
# ===========================================================================
def load_parquet_table(inference_dir: Path, table_name: str) -> pd.DataFrame:
    """Reads every part-*.parquet file under inference_dir/{table_name}/state=*/,
    regardless of how many states or how many flushes each contains --
    aggregates automatically as more states/parts are added, same idea as
    the old CSV glob just against the new partitioned layout."""
    table_dir = inference_dir / table_name
    if not table_dir.exists():
        return pd.DataFrame()
    files = sorted(table_dir.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f, engine="pyarrow"))
        except Exception as e:
            print(f"  WARNING: couldn't read {f}: {e}")
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "CWNS_ID" in combined.columns:
        combined["CWNS_ID"] = combined["CWNS_ID"].astype(str)
    return combined


def load_facilities(plants_gpkg: Path, plants_layer: str, keep_ids: set) -> list[dict]:
    """Reported/best-available plant locations (same geometry the pipeline
    tiled), filtered to CWNS_IDs that have a row in the `plants` table."""
    if not plants_gpkg.exists():
        raise FileNotFoundError(f"{plants_gpkg} not found.")

    plants = gpd.read_file(plants_gpkg, layer=plants_layer).to_crs(C.EXPORT_CRS)
    plants["CWNS_ID"] = plants["CWNS_ID"].astype(str)
    plants = plants[plants["CWNS_ID"].isin(keep_ids)]

    features = []
    for _, r in plants.iterrows():
        if r.geometry is None or r.geometry.is_empty:
            continue
        c = r.geometry.centroid
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [c.x, c.y]},
            "properties": {"CWNS_ID": r["CWNS_ID"]},
        })
    return features


def load_detections(dets: pd.DataFrame) -> list[dict]:
    """Raw per-box detections (pre-dedup) from the `detections` table,
    already aggregated across whichever states were found."""
    if dets.empty:
        return []
    class_col = "class_name" if "class_name" in dets.columns else "class"
    features = []
    for _, r in dets.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
            "properties": {
                "CWNS_ID":    r["CWNS_ID"],
                "class":      r[class_col],
                "confidence": round(float(r["confidence"]), 3),
            },
        })
    return features


def _bbox_ring(xmin, ymin, xmax, ymax):
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]


def load_objects(inference_dir: Path) -> list[dict]:
    """Merged post-dedup objects from the `objects` table, as polygons using
    the union bbox across all raw detections that fed each object. Filterable
    by the confidence slider via max_confidence."""
    objs = load_parquet_table(inference_dir, "objects")
    if objs.empty:
        return []
    features = []
    for _, r in objs.iterrows():
        if pd.isna(r.get("geo_xmin")):
            continue
        ring = _bbox_ring(r["geo_xmin"], r["geo_ymin"], r["geo_xmax"], r["geo_ymax"])
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "object_id":      r.get("object_id", ""),
                "CWNS_ID":        r["CWNS_ID"],
                "class":          r["class_name"],
                "max_confidence": round(float(r["max_confidence"]), 3),
                "n_merged":       int(r["n_merged"]),
            },
        })
    return features


def load_tiles(inference_dir: Path) -> list[dict]:
    """Every tile ATTEMPTED (kept/deleted/fetch_failed) from the `tiles`
    table, as polygons of the tile's exact NAIP footprint -- lets you check
    whether an object sits awkwardly against a tile edge. No confidence
    field (tiles aren't detections), so not affected by the slider."""
    tiles = load_parquet_table(inference_dir, "tiles")
    if tiles.empty:
        return []
    features = []
    for _, r in tiles.iterrows():
        ring = _bbox_ring(r["bbox_xmin"], r["bbox_ymin"], r["bbox_xmax"], r["bbox_ymax"])
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "tile_id":           r["tile_id"],
                "CWNS_ID":           r["CWNS_ID"],
                "outcome":           r["outcome"],
                "is_nodata":         bool(r.get("is_nodata", False)),
                "n_raw_detections":  int(r.get("n_raw_detections", 0) or 0),
            },
        })
    return features


# ===========================================================================
# HTML build
# ===========================================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>WWTP Infrastructure Detection Map</title>
<style>__MAPLIBRE_CSS__</style>
<style>
  body { margin: 0; padding: 0; font-family: -apple-system, Segoe UI, Arial, sans-serif; }
  #map { position: absolute; top: 0; bottom: 0; width: 100%; }
  #legend {
    position: absolute; top: 12px; right: 12px; z-index: 1;
    background: white; padding: 10px 14px; border-radius: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-size: 13px; line-height: 1.7;
    max-width: 220px;
  }
  #legend h4 { margin: 0 0 4px 0; font-size: 12px; text-transform: uppercase;
               letter-spacing: 0.03em; color: #555; }
  .swatch { display: inline-block; width: 11px; height: 11px; border-radius: 50%;
            margin-right: 6px; border: 1px solid rgba(0,0,0,0.3); vertical-align: middle; }
  .facility-swatch { background: #333333; }
  #stats { position: absolute; bottom: 12px; left: 12px; z-index: 1;
           background: white; padding: 6px 10px; border-radius: 6px;
           box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-size: 12px; color: #444;
           max-width: 320px; }
  #basemap-toggle {
    position: absolute; top: 12px; left: 56px; z-index: 1;
    background: white; border-radius: 6px; overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3); display: flex;
  }
  #basemap-toggle button {
    border: none; background: white; padding: 6px 12px; cursor: pointer;
    font-size: 13px; font-family: inherit;
  }
  #basemap-toggle button:not(:last-child) { border-right: 1px solid #ddd; }
  #basemap-toggle button.active { background: #333333; color: white; }
  .maplibregl-popup-content { font-size: 13px; }
  #controls {
    position: absolute; top: 12px; right: 244px; z-index: 1;
    background: white; padding: 10px 14px; border-radius: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-size: 13px; color: #333;
    max-width: 200px;
  }
  #controls label { display: block; margin-bottom: 6px; cursor: pointer; }
  #controls input[type="checkbox"] { margin-right: 6px; vertical-align: middle; }
  #controls input[type="range"] { width: 100%; margin-top: 2px; }
  #controls .slider-row { margin-top: 10px; }
  #controls .slider-row label { cursor: default; margin-bottom: 2px; }
</style>
</head>
<body>
<div id="map"></div>
<div id="basemap-toggle">
  <button id="btn-streets">Streets</button>
  <button id="btn-imagery">Imagery</button>
</div>
<div id="controls">
  <label><input type="checkbox" id="chk-objects"> Object boxes</label>
  <label><input type="checkbox" id="chk-tiles"> Tile boxes</label>
  <div class="slider-row">
    <label for="conf-slider">Min confidence: <span id="conf-slider-label">0.00</span></label>
    <input type="range" id="conf-slider" min="0" max="1" step="0.01" value="0">
  </div>
</div>
<div id="legend">
  <h4>Facilities</h4>
  <div><span class="swatch facility-swatch"></span>Reported location</div>
  <h4 style="margin-top:10px;">Detected infrastructure</h4>
  __LEGEND_ITEMS__
  <h4 style="margin-top:10px;">Tile outcome</h4>
  <div><span class="swatch" style="background:#2ca02c; border-radius:2px;"></span>Kept (detection found)</div>
  <div><span class="swatch" style="background:#888888; border-radius:2px;"></span>Deleted (no detection)</div>
  <div><span class="swatch" style="background:#d62728; border-radius:2px;"></span>Fetch failed</div>
</div>
<div id="stats">__N_FACILITIES__ facilities &nbsp;|&nbsp; __N_DETECTIONS__ detections &nbsp;|&nbsp; states loaded: __STATES_LOADED__</div>
<script>__MAPLIBRE_JS__</script>
<script>
const facilities = __FACILITY_GEOJSON__;
const detections = __DETECTION_GEOJSON__;
const objects = __OBJECT_GEOJSON__;
const tiles = __TILE_GEOJSON__;
const classColors = __CLASS_COLORS_JSON__;

const map = new maplibregl.Map({
  container: 'map',
  style: __BASE_STYLE_JSON__,
  center: __CENTER__,
  zoom: 4
});

map.addControl(new maplibregl.NavigationControl(), 'top-left');

map.on('load', () => {
  map.addSource('facilities', { type: 'geojson', data: facilities });
  map.addLayer({
    id: 'facilities-layer',
    type: 'circle',
    source: 'facilities',
    paint: {
      'circle-radius': 6,
      'circle-color': '#333333',
      'circle-stroke-width': 2,
      'circle-stroke-color': '#ffffff'
    }
  });

  map.addSource('detections', { type: 'geojson', data: detections });
  const colorExpr = ['match', ['get', 'class']];
  for (const [cls, color] of Object.entries(classColors)) {
    colorExpr.push(cls, color);
  }
  colorExpr.push('#999999'); // fallback for any unmapped class

  map.addLayer({
    id: 'detections-layer',
    type: 'circle',
    source: 'detections',
    paint: {
      'circle-radius': ['+', 3, ['*', 5, ['get', 'confidence']]],
      'circle-color': colorExpr,
      'circle-stroke-width': 1,
      'circle-stroke-color': '#ffffff',
      'circle-opacity': 0.85
    }
  });

  // Merged object bounding boxes (post-dedup) -- hidden by default, same
  // per-class colors as the detection points. Filterable by the confidence
  // slider via max_confidence.
  map.addSource('objects', { type: 'geojson', data: objects });
  map.addLayer({
    id: 'objects-fill',
    type: 'fill',
    source: 'objects',
    layout: { visibility: 'none' },
    paint: { 'fill-color': colorExpr, 'fill-opacity': 0.15 }
  });
  map.addLayer({
    id: 'objects-outline',
    type: 'line',
    source: 'objects',
    layout: { visibility: 'none' },
    paint: { 'line-color': colorExpr, 'line-width': 2 }
  });

  // Tile footprints -- hidden by default (33% overlap means a lot of
  // stacked rectangles). Colored by outcome, not confidence (tiles aren't
  // detections). A transparent fill layer sits under the outline purely so
  // clicking anywhere INSIDE a tile (not just exactly on its edge) triggers
  // the popup.
  map.addSource('tiles', { type: 'geojson', data: tiles });
  const tileColorExpr = ['match', ['get', 'outcome'],
    'kept', '#2ca02c',
    'deleted', '#888888',
    'fetch_failed', '#d62728',
    '#888888'];
  map.addLayer({
    id: 'tiles-fill',
    type: 'fill',
    source: 'tiles',
    layout: { visibility: 'none' },
    paint: { 'fill-color': tileColorExpr, 'fill-opacity': 0.01 }
  });
  map.addLayer({
    id: 'tiles-outline',
    type: 'line',
    source: 'tiles',
    layout: { visibility: 'none' },
    paint: { 'line-color': tileColorExpr, 'line-width': 1, 'line-opacity': 0.7 }
  });

  for (const layerId of ['facilities-layer', 'detections-layer', 'objects-fill', 'tiles-fill']) {
    map.on('click', layerId, (e) => {
      const p = e.features[0].properties;
      const rows = Object.entries(p)
        .map(([k, v]) => `<b>${k}</b>: ${v}`)
        .join('<br>');
      new maplibregl.Popup()
        .setLngLat(e.lngLat)
        .setHTML(rows)
        .addTo(map);
    });
    map.on('mouseenter', layerId, () => { map.getCanvas().style.cursor = 'pointer'; });
    map.on('mouseleave', layerId, () => { map.getCanvas().style.cursor = ''; });
  }

  // Confidence slider -- filters detection points AND object boxes together.
  // Tile boxes are untouched (no confidence field to filter on).
  function applyConfidenceFilter(minConf) {
    map.setFilter('detections-layer', ['>=', ['get', 'confidence'], minConf]);
    map.setFilter('objects-fill', ['>=', ['get', 'max_confidence'], minConf]);
    map.setFilter('objects-outline', ['>=', ['get', 'max_confidence'], minConf]);
  }
  const confSlider = document.getElementById('conf-slider');
  const confLabel = document.getElementById('conf-slider-label');
  confSlider.addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    confLabel.textContent = v.toFixed(2);
    applyConfidenceFilter(v);
  });
  applyConfidenceFilter(parseFloat(confSlider.value));

  document.getElementById('chk-objects').addEventListener('change', (e) => {
    const vis = e.target.checked ? 'visible' : 'none';
    map.setLayoutProperty('objects-fill', 'visibility', vis);
    map.setLayoutProperty('objects-outline', 'visibility', vis);
  });
  document.getElementById('chk-tiles').addEventListener('change', (e) => {
    const vis = e.target.checked ? 'visible' : 'none';
    map.setLayoutProperty('tiles-fill', 'visibility', vis);
    map.setLayoutProperty('tiles-outline', 'visibility', vis);
  });
});

let currentBasemap = 'streets';

function switchBasemap(key) {
  if (key === currentBasemap) return;
  currentBasemap = key;
  map.setLayoutProperty('streets-layer', 'visibility', key === 'streets' ? 'visible' : 'none');
  map.setLayoutProperty('imagery-layer', 'visibility', key === 'imagery' ? 'visible' : 'none');
  updateToggleUI();
}

function updateToggleUI() {
  document.getElementById('btn-streets').classList.toggle('active', currentBasemap === 'streets');
  document.getElementById('btn-imagery').classList.toggle('active', currentBasemap === 'imagery');
}

document.getElementById('btn-streets').addEventListener('click', () => switchBasemap('streets'));
document.getElementById('btn-imagery').addEventListener('click', () => switchBasemap('imagery'));
updateToggleUI();
</script>
</body>
</html>
"""


def load_vendor_asset(path: Path, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found at {path}.\n"
            f"Expected the vendored MapLibre files under pipeline/vendor/ "
            f"(maplibre-gl.js, maplibre-gl.css) — see docs for how they were fetched."
        )
    return path.read_text(encoding="utf-8")


def build_html(facility_features: list[dict], detection_features: list[dict],
                object_features: list[dict], tile_features: list[dict],
                states_loaded: list[str]) -> str:
    if facility_features:
        lons = [f["geometry"]["coordinates"][0] for f in facility_features]
        lats = [f["geometry"]["coordinates"][1] for f in facility_features]
        center = [sum(lons) / len(lons), sum(lats) / len(lats)]
    else:
        center = [-98.5, 39.8]

    legend_items = "\n  ".join(
        f'<div><span class="swatch" style="background:{color}"></span>{cls}</div>'
        for cls, color in CLASS_COLORS.items()
    )

    maplibre_js  = load_vendor_asset(MAPLIBRE_JS, "MapLibre JS")
    maplibre_css = load_vendor_asset(MAPLIBRE_CSS, "MapLibre CSS")

    html = HTML_TEMPLATE
    html = html.replace("__LEGEND_ITEMS__", legend_items)
    html = html.replace("__N_FACILITIES__", str(len(facility_features)))
    html = html.replace("__N_DETECTIONS__", str(len(detection_features)))
    html = html.replace("__STATES_LOADED__", ", ".join(states_loaded) if states_loaded else "none")
    html = html.replace(
        "__FACILITY_GEOJSON__",
        json.dumps({"type": "FeatureCollection", "features": facility_features}),
    )
    html = html.replace(
        "__DETECTION_GEOJSON__",
        json.dumps({"type": "FeatureCollection", "features": detection_features}),
    )
    html = html.replace(
        "__OBJECT_GEOJSON__",
        json.dumps({"type": "FeatureCollection", "features": object_features}),
    )
    html = html.replace(
        "__TILE_GEOJSON__",
        json.dumps({"type": "FeatureCollection", "features": tile_features}),
    )
    html = html.replace("__CLASS_COLORS_JSON__", json.dumps(CLASS_COLORS))
    html = html.replace("__BASE_STYLE_JSON__", json.dumps(BASE_STYLE))
    html = html.replace("__CENTER__", json.dumps(center))
    html = html.replace("__MAPLIBRE_CSS__", maplibre_css)
    html = html.replace("__MAPLIBRE_JS__", maplibre_js)
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference-dir", type=str,
                     default=str(getattr(C, "STATE_INFERENCE_ROOT", Path("data/inference_state"))),
                     help="folder containing detections/state=XX/*.parquet and plants/state=XX/*.parquet")
    ap.add_argument("--plants-gpkg", type=str,
                     default=str(getattr(C, "PLANTS_RAW_GPKG", Path("data/plants.gpkg"))))
    ap.add_argument("--plants-layer", type=str,
                     default=str(getattr(C, "PLANTS_RAW_LAYER", "treatment")))
    ap.add_argument("--output", type=str, default=None,
                     help="output HTML path (default: <inference-dir>/facility_map.html)")
    args = ap.parse_args()

    inference_dir = Path(args.inference_dir)
    inference_dir.mkdir(parents=True, exist_ok=True)
    output_html = Path(args.output) if args.output else inference_dir / "facility_map.html"

    print("=== 06_build_map.py ===\n")
    print(f"Scanning {inference_dir}/detections and {inference_dir}/plants for Parquet files...")

    dets = load_parquet_table(inference_dir, "detections")
    plants_table = load_parquet_table(inference_dir, "plants")

    det_states = set(dets["state_fips"]) if "state_fips" in dets.columns else set()
    plant_states = set(plants_table["state_fips"]) if "state_fips" in plants_table.columns else set()
    states_loaded = sorted(det_states | plant_states)

    if not states_loaded:
        det_dir, plants_dir = inference_dir / "detections", inference_dir / "plants"
        det_n = len(list(det_dir.rglob("*.parquet"))) if det_dir.exists() else 0
        plants_n = len(list(plants_dir.rglob("*.parquet"))) if plants_dir.exists() else 0
        raise SystemExit(
            f"No usable Parquet files found under {inference_dir}.\n"
            f"  {det_dir}: exists={det_dir.exists()}, .parquet files found={det_n}\n"
            f"  {plants_dir}: exists={plants_dir.exists()}, .parquet files found={plants_n}\n"
            f"If one of these is 0/missing, check how the folder was downloaded --\n"
            f"a common cause is an extra nesting level, e.g. "
            f"{inference_dir}/inference_state/detections/... instead of "
            f"{inference_dir}/detections/..."
        )
    print(f"  States found: {', '.join(states_loaded)}")
    print(f"  Detections rows (raw, pre-dedup): {len(dets)}")
    print(f"  Plant (corrected-coordinate) rows: {len(plants_table)}")

    keep_ids = set(plants_table["CWNS_ID"]) if not plants_table.empty else set()

    print("\nLoading facility locations...")
    facility_features = load_facilities(Path(args.plants_gpkg), args.plants_layer, keep_ids)
    print(f"  {len(facility_features)} facilities")

    print("Loading detected infrastructure...")
    detection_features = load_detections(dets)
    print(f"  {len(detection_features)} detections")

    print("Loading merged object boxes...")
    object_features = load_objects(inference_dir)
    print(f"  {len(object_features)} objects")

    print("Loading tile footprints...")
    tile_features = load_tiles(inference_dir)
    print(f"  {len(tile_features)} tiles")

    html = build_html(facility_features, detection_features, object_features, tile_features, states_loaded)
    output_html.write_text(html, encoding="utf-8")

    print(f"\nWrote {output_html}")
    print("\nMapLibre uses a background Web Worker, which Chrome/Edge block on")
    print("file:// pages. If double-clicking the file shows a blank map, serve it locally:")
    print(f"    cd {output_html.parent}")
    print("    python -m http.server 8000")
    print(f"    (then open http://localhost:8000/{output_html.name})")
    print("\nRerun this script anytime after downloading more states' Parquet files into")
    print(f"{inference_dir} -- it picks up whatever's there, no manual merging needed.")


if __name__ == "__main__":
    main()
