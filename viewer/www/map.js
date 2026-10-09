// CWNS treatment-plant location viewer. Pattern follows sewershed_plus's
// map.js: deck.gl, raster basemap TileLayer + BitmapLayer, search bar with
// suggestions, legend that doubles as the filter panel, slide-in detail panel.
// No vector tiles: every plant arrives in one JSON (see app.py).

const { DeckGL, TileLayer, BitmapLayer, ScatterplotLayer, LineLayer, GeoJsonLayer, TextLayer, FlyToInterpolator } = deck;

const DATA_URL = "/data/plants.json";
const SITES_URL = "/data/sites.geojson";
const DETECTIONS_URL = "/data/detections.json";
const SITES_MIN_ZOOM = 11;       // parcel outlines of each plant's final site
const DETECTIONS_MIN_ZOOM = 13;  // detected objects
const PARCELS_MIN_ZOOM = 15;     // every parcel in view, live from the local Regrid mirror
const ALLOBJ_MIN_ZOOM = 14;      // every detected object in view (candidates, reported, corrected)
const RANK_COLORS = { 1: [249, 115, 22], top5: [250, 204, 21], rest: [229, 231, 235] };
const REPORTED_PARCEL_COLOR = [34, 211, 238];
const OTHER_CAND_COLOR = [167, 139, 250];

const CLASS_COLORS = {
  clarifier: [56, 189, 248],
  aeration_basin: [244, 114, 182],
  digester: [250, 204, 21],
  chlorine_contact: [52, 211, 153],
  drying_bed: [251, 146, 60],
  oxidation_pond: [129, 140, 248],
};
const CLASS_FALLBACK = [229, 231, 235];

// Order here = legend order (most important first). Colours chosen to stay
// distinct on both the street and imagery basemaps.
const STATUS = {
  moved:               { label: "Moved (model)",          color: [249, 115, 22],  hint: "Stage 1 flagged it and the re-ranker's #1 parcel cleared the cutoff" },
  flagged_not_moved:   { label: "Flagged, not moved",     color: [220, 38, 38],   hint: "Stage 1 doubts the reported location; no candidate cleared the cutoff" },
  kept_site:           { label: "Kept – split site",      color: [234, 179, 8],   hint: "The re-ranker's #1 is a neighbouring parcel of the same plant site as the reported parcel (split parcels), so it was not moved" },
  verified_corrected:  { label: "Verified – corrected",   color: [37, 99, 235],   hint: "A reviewer supplied the corrected location" },
  verified_correct:    { label: "Verified – correct",     color: [22, 163, 74],   hint: "A reviewer confirmed the reported location" },
  reviewed_unresolved: { label: "Reviewed – unresolved",  color: [147, 51, 234],  hint: "Reviewed as wrong, no corrected location yet" },
  kept_osm:            { label: "Kept – OSM tagged",      color: [20, 184, 166],  hint: "Reported parcel carries an OSM wastewater tag" },
  kept_model:          { label: "Kept – model",           color: [163, 230, 53],  hint: "Stage 1 scored the reported location as correct" },
  pending:             { label: "Pending (no model run)", color: [156, 163, 175], hint: "Preview mode: model output not loaded yet" },
  not_assessed:        { label: "Not assessed",           color: [107, 114, 128], hint: "Outside model scope (no NAIP, population ≤ 100 or unknown, …)" },
};
const UNKNOWN_COLOR = [0, 0, 0];
const SELECTED_COLOR = [255, 255, 0];
const REPORTED_COLOR = [80, 80, 80];
const MOVE_STATUSES = new Set(["moved", "verified_corrected"]);
const REPORTED_MIN_ZOOM = 9;   // ghost "was here" rings only when zoomed in

const BASEMAPS = {
  street: {
    url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: '<a href="https://www.openstreetmap.org/copyright" target="_blank">© OpenStreetMap contributors</a>',
  },
  imagery: {
    url: "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "Imagery © Esri, Maxar, Earthstar Geographics",
  },
};

function basemapLayer(key) {
  return new TileLayer({
    id: `basemap-${key}`,
    data: BASEMAPS[key].url,
    minZoom: 0,
    maxZoom: 19,
    tileSize: 256,
    renderSubLayers: (props) => {
      const { west, south, east, north } = props.tile.bbox;
      return new BitmapLayer(props, { data: null, image: props.data, bounds: [west, south, east, north] });
    },
  });
}

// ---- state ----------------------------------------------------------------
let ALL = [];
let BY_ID = new Map();
let filtered = [];
let moves = [];
let selected = null;
let basemap = "street";
const enabled = new Set(Object.keys(STATUS));
let stateFilter = "";
let showMoves = true;
let sizeBase = 4;
let SITES = { type: "FeatureCollection", features: [] };
let DETS = [];
let sitesShown = { type: "FeatureCollection", features: [] };
let detsShown = [];
let showSites = true;
let showDets = true;
let viewState = { longitude: -96.5, latitude: 38.5, zoom: 3.8, pitch: 0, bearing: 0 };
let CUTOFF = null;
// Explore: the selected plant's candidates, and what is in the map window.
let cands = [];                 // candidate rows of the selected plant (ranked)
let candParcels = { type: "FeatureCollection", features: [] };  // their outlines (+ reported parcel)
let activeParcel = null;        // ll_uuid highlighted from the table or a click
let showParcels = false;
let showAllObj = false;
let viewParcels = { type: "FeatureCollection", features: [] };
let viewObjects = [];
let viewSeq = 0;

const $ = (id) => document.getElementById(id);
const tooltip = $("tooltip");
const panel = $("detail-panel");
const statusEl = $("status");

function colorOf(p) {
  return (STATUS[p.status] || {}).color || UNKNOWN_COLOR;
}
function labelOf(s) {
  return (STATUS[s] || {}).label || s;
}
function hex(c) {
  return "#" + c.slice(0, 3).map((v) => v.toString(16).padStart(2, "0")).join("");
}
function fmt(v, nd = 3) {
  return v === null || v === undefined ? "—" : typeof v === "number" ? v.toFixed(nd) : String(v);
}
function fmtDist(m) {
  if (m === null || m === undefined) return "—";
  return m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${Math.round(m)} m`;
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function radiusPx() {
  const z = viewState.zoom;
  const k = z < 5 ? 0.6 : z < 8 ? 0.85 : z < 12 ? 1.2 : 1.6;
  return Math.max(1, sizeBase * k);
}

// ---- filtering ------------------------------------------------------------
function applyFilters() {
  filtered = ALL.filter((p) => enabled.has(p.status) && (!stateFilter || p.state === stateFilter));
  moves = filtered.filter(
    (p) => MOVE_STATUSES.has(p.status) && p.rlat !== null && p.rlon !== null && (p.moved_m || 0) > 1
  );
  const keep = (id) => {
    const p = BY_ID.get(id);
    return p && enabled.has(p.status) && (!stateFilter || p.state === stateFilter);
  };
  sitesShown = { type: "FeatureCollection", features: SITES.features.filter((f) => keep(f.properties.CWNS_ID)) };
  detsShown = DETS.filter((d) => keep(d.id));
  $("visible-count").textContent = `${filtered.length.toLocaleString()} of ${ALL.length.toLocaleString()} plants shown`;
  render();
}

// ---- layers ---------------------------------------------------------------
function layers() {
  const r = radiusPx();
  const zoomBucket = Math.round(viewState.zoom * 2);
  const out = [basemapLayer(basemap)];

  if (showSites && viewState.zoom >= SITES_MIN_ZOOM && sitesShown.features.length) {
    out.push(
      new GeoJsonLayer({
        id: "sites",
        data: sitesShown,
        stroked: true,
        filled: true,
        getFillColor: (f) => [...((STATUS[f.properties.status] || {}).color || UNKNOWN_COLOR), 45],
        getLineColor: (f) => [...((STATUS[f.properties.status] || {}).color || UNKNOWN_COLOR), 230],
        lineWidthUnits: "pixels",
        getLineWidth: 2,
        pickable: true,
      })
    );
  }

  if (showParcels && viewState.zoom >= PARCELS_MIN_ZOOM && viewParcels.features.length) {
    out.push(
      new GeoJsonLayer({
        id: "view-parcels",
        data: viewParcels,
        stroked: true,
        filled: true,
        getFillColor: (f) => (f.properties.cand && f.properties.cand.length ? [...OTHER_CAND_COLOR, 50] : [255, 255, 255, 8]),
        getLineColor: (f) => (f.properties.cand && f.properties.cand.length ? [...OTHER_CAND_COLOR, 230] : [255, 255, 255, 170]),
        lineWidthUnits: "pixels",
        getLineWidth: (f) => (f.properties.ll_uuid === activeParcel ? 3 : 1),
        pickable: true,
        updateTriggers: { getLineWidth: activeParcel },
      })
    );
  }

  if (selected && candParcels.features.length) {
    const rankOf = new Map(cands.map((c) => [c.parcel, c.rank]));
    const colorFor = (f) => {
      const u = f.properties.ll_uuid;
      if (rankOf.has(u)) {
        const r = rankOf.get(u);
        return r === 1 ? RANK_COLORS[1] : r !== null && r <= 5 ? RANK_COLORS.top5 : RANK_COLORS.rest;
      }
      return REPORTED_PARCEL_COLOR;
    };
    out.push(
      new GeoJsonLayer({
        id: "cand-parcels",
        data: candParcels,
        stroked: true,
        filled: true,
        getFillColor: (f) => [...colorFor(f), f.properties.ll_uuid === activeParcel ? 90 : 35],
        getLineColor: (f) => [...colorFor(f), 240],
        lineWidthUnits: "pixels",
        getLineWidth: (f) => (f.properties.ll_uuid === activeParcel ? 4 : 2),
        pickable: true,
        updateTriggers: { getFillColor: [activeParcel, selected.id], getLineWidth: activeParcel },
      })
    );
    const labelled = cands.filter((c) => c.lat !== null && c.lon !== null && c.rank !== null && c.rank <= 20);
    if (labelled.length && viewState.zoom >= 12) {
      out.push(
        new TextLayer({
          id: "cand-ranks",
          data: labelled,
          getPosition: (c) => [c.lon, c.lat],
          getText: (c) => String(c.rank),
          getSize: 14,
          getColor: [17, 24, 39, 255],
          background: true,
          getBackgroundColor: (c) => [...(c.rank === 1 ? RANK_COLORS[1] : c.rank <= 5 ? RANK_COLORS.top5 : RANK_COLORS.rest), 230],
          backgroundPadding: [4, 1],
          fontWeight: 700,
          pickable: false,
        })
      );
    }
  }

  if (showMoves && moves.length) {
    out.push(
      new LineLayer({
        id: "move-lines",
        data: moves,
        getSourcePosition: (d) => [d.rlon, d.rlat],
        getTargetPosition: (d) => [d.lon, d.lat],
        getColor: (d) => [...colorOf(d), 200],
        getWidth: 2,
        widthUnits: "pixels",
        pickable: false,
      })
    );
    if (viewState.zoom >= REPORTED_MIN_ZOOM) {
      out.push(
        new ScatterplotLayer({
          id: "reported-ghosts",
          data: moves,
          getPosition: (d) => [d.rlon, d.rlat],
          radiusUnits: "pixels",
          getRadius: r,
          filled: true,
          getFillColor: [255, 255, 255, 120],
          stroked: true,
          getLineColor: REPORTED_COLOR,
          lineWidthUnits: "pixels",
          getLineWidth: 1.5,
          pickable: true,
          updateTriggers: { getRadius: [zoomBucket, sizeBase] },
        })
      );
    }
  }

  out.push(
    new ScatterplotLayer({
      id: "plants",
      data: filtered,
      getPosition: (d) => [d.lon, d.lat],
      radiusUnits: "pixels",
      getRadius: r,
      getFillColor: (d) => [...colorOf(d), 235],
      stroked: true,
      getLineColor: [255, 255, 255, 200],
      lineWidthUnits: "pixels",
      getLineWidth: viewState.zoom >= 7 ? 1 : 0.5,
      pickable: true,
      updateTriggers: { getRadius: [zoomBucket, sizeBase], getLineWidth: zoomBucket },
    })
  );

  if (showDets && viewState.zoom >= DETECTIONS_MIN_ZOOM && detsShown.length) {
    out.push(
      new ScatterplotLayer({
        id: "detections",
        data: detsShown,
        getPosition: (d) => [d.lon, d.lat],
        radiusUnits: "pixels",
        getRadius: (d) => (d.used ? 5 : 4),
        getFillColor: (d) => [...(CLASS_COLORS[d.cls] || CLASS_FALLBACK), d.conf >= 0.4 ? 230 : 110],
        stroked: true,
        getLineColor: (d) => (d.used ? [0, 0, 0, 255] : [255, 255, 255, 200]),
        lineWidthUnits: "pixels",
        getLineWidth: (d) => (d.used ? 2 : 1),
        pickable: true,
      })
    );
  }

  if (showAllObj && viewState.zoom >= ALLOBJ_MIN_ZOOM && viewObjects.length) {
    out.push(
      new ScatterplotLayer({
        id: "all-objects",
        data: viewObjects,
        getPosition: (d) => [d.lon, d.lat],
        radiusUnits: "pixels",
        getRadius: 4,
        getFillColor: (d) => [...(CLASS_COLORS[d.cls] || CLASS_FALLBACK), d.conf >= 0.4 ? 230 : 110],
        stroked: true,
        getLineColor: [17, 24, 39, 220],
        lineWidthUnits: "pixels",
        getLineWidth: 1,
        pickable: true,
      })
    );
  }

  if (selected) {
    const sel = [{ pos: [selected.lon, selected.lat] }];
    if (selected.rlat !== null && MOVE_STATUSES.has(selected.status) && (selected.moved_m || 0) > 1) {
      sel.push({ pos: [selected.rlon, selected.rlat], ghost: true });
    }
    out.push(
      new ScatterplotLayer({
        id: "selected",
        data: sel,
        getPosition: (d) => d.pos,
        radiusUnits: "pixels",
        getRadius: r + 6,
        filled: false,
        stroked: true,
        getLineColor: (d) => (d.ghost ? REPORTED_COLOR : SELECTED_COLOR),
        lineWidthUnits: "pixels",
        getLineWidth: 3,
        pickable: false,
        updateTriggers: { getRadius: [zoomBucket, sizeBase, selected.id] },
      })
    );
  }
  return out;
}

let deckgl = null;
function render() {
  if (deckgl) deckgl.setProps({ layers: layers(), viewState });
}

// ---- tooltip / click ------------------------------------------------------
function onHover({ object, x, y, layer }) {
  if (!object) {
    tooltip.style.display = "none";
    return;
  }
  if (layer && (layer.id === "view-parcels" || layer.id === "cand-parcels")) {
    const pr = object.properties;
    const c = cands.find((x) => x.parcel === pr.ll_uuid);
    const others = (pr.cand || []).filter((x) => !selected || x.id !== selected.id);
    tooltip.innerHTML =
      `<div><b>${esc(pr.owner || "(no owner)")}</b></div>` +
      (pr.ll_gisacre !== null && pr.ll_gisacre !== undefined ? `<div>${fmt(pr.ll_gisacre, 2)} acres</div>` : "") +
      (c ? `<div>candidate #${c.rank ?? "—"} · re-rank ${fmt(c.rr)}</div>` :
        selected && pr.ll_uuid === selected.rparcel ? "<div>reported parcel</div>" : "") +
      (others.length ? `<div style="color:#7c3aed">candidate of ${others.length} other plant(s)</div>` : "") +
      '<div style="color:#888">click for details</div>';
    tooltip.style.left = `${x + 14}px`;
    tooltip.style.top = `${y + 14}px`;
    tooltip.style.display = "block";
    return;
  }
  if (layer && layer.id === "all-objects") {
    tooltip.innerHTML =
      `<div><b>${esc(object.cls.replace(/_/g, " "))}</b> · ${(object.conf * 100).toFixed(0)}%</div>` +
      `<div style="color:#888">${esc(object.src)} run for plant ${esc(object.id)}</div>`;
    tooltip.style.left = `${x + 14}px`;
    tooltip.style.top = `${y + 14}px`;
    tooltip.style.display = "block";
    return;
  }
  if (layer && layer.id === "detections") {
    tooltip.innerHTML =
      `<div><b>${esc(object.cls.replace(/_/g, " "))}</b> · ${(object.conf * 100).toFixed(0)}%</div>` +
      `<div style="color:#888">${esc(object.src)} detection · plant ${esc(object.id)}</div>` +
      (object.used ? "<div>used for the moved point</div>" : "") +
      (object.conf < 0.4 ? '<div style="color:#888">below 0.4, not used</div>' : "");
    tooltip.style.left = `${x + 14}px`;
    tooltip.style.top = `${y + 14}px`;
    tooltip.style.display = "block";
    return;
  }
  if (layer && layer.id === "sites") {
    object = BY_ID.get(object.properties.CWNS_ID);
    if (!object) {
      tooltip.style.display = "none";
      return;
    }
  }
  const ghost = layer && layer.id === "reported-ghosts";
  tooltip.innerHTML =
    `<div><b>${esc(object.name || object.id)}</b></div>` +
    `<div class="tt-status" style="color:${hex(colorOf(object))}">${esc(labelOf(object.status))}</div>` +
    (ghost ? `<div>reported location (before move)</div>` : "") +
    `<div style="color:#888">${esc(object.id)} · ${esc(object.state || "")}</div>`;
  tooltip.style.left = `${x + 14}px`;
  tooltip.style.top = `${y + 14}px`;
  tooltip.style.display = "block";
}

function onClick({ object, layer }) {
  if (!object) return;
  if (layer && (layer.id === "view-parcels" || layer.id === "cand-parcels")) {
    showParcelCard(object.properties);
    return;
  }
  if (layer && layer.id === "all-objects") {
    const p = BY_ID.get(object.id);
    if (p) select(p, false);
    return;
  }
  if (layer && layer.id === "sites") object = BY_ID.get(object.properties.CWNS_ID);
  else if (layer && layer.id === "detections") object = BY_ID.get(object.id);
  if (object) select(object, false);
}

function select(p, fly) {
  const changed = !selected || selected.id !== p.id;
  selected = p;
  if (changed) {
    cands = [];
    candParcels = { type: "FeatureCollection", features: [] };
    activeParcel = null;
    $("parcel-card").classList.remove("open");   // it described the parcel relative to the old plant
  }
  showPanel(p);
  if (changed) loadCandidates(p);
  history.replaceState(null, "", `#id=${encodeURIComponent(p.id)}`);
  if (fly) {
    // Centre the plant in the part of the map the detail panel leaves visible.
    const zoom = Math.max(viewState.zoom, 15);
    const w = window.innerWidth, h = window.innerHeight;
    const panelW = Math.min(400, 0.92 * w);
    const vp = new deck.WebMercatorViewport({ width: w, height: h, longitude: p.lon, latitude: p.lat, zoom });
    const [lon, lat] = vp.unproject([w / 2 + panelW / 2, h / 2]);
    viewState = {
      ...viewState,
      longitude: lon,
      latitude: lat,
      zoom,
      transitionDuration: 1200,
      transitionInterpolator: new FlyToInterpolator(),
    };
  }
  render();
}

function closePanel() {
  panel.classList.remove("open");
  document.body.classList.remove("panel-open");
  selected = null;
  cands = [];
  candParcels = { type: "FeatureCollection", features: [] };
  activeParcel = null;
  history.replaceState(null, "", location.pathname);
  render();
}

const TIER_LABELS = {
  verified: "verified by a reviewer",
  model: "model decision",
  model_small_plant: "model decision, small plant (less certain)",
};

const COORD_LABELS = {
  detections: (p) => `mean of ${p.nobj} detection(s)`,
  centroid: () => "parcel centroid",
  polylabel: () => "parcel interior point",
  reviewer: () => "reviewer",
  reported: () => "reported location",
};

function gmaps(lat, lon) {
  return `https://www.google.com/maps/search/?api=1&query=${lat},${lon}`;
}

function showPanel(p) {
  const c = colorOf(p);
  const fact = (k, v) => `<div class="fact"><dt>${k}</dt><dd>${v}</dd></div>`;
  const moved = MOVE_STATUSES.has(p.status) && (p.moved_m || 0) > 1;
  const reviews = (p.reviews || [])
    .map(
      (r) =>
        `<div class="review-item">Round ${r.round}: <b>${esc(r.verdict)}</b> <span style="color:#888">(${esc(r.task)})</span>` +
        (r.notes ? `<div class="notes">${esc(r.notes)}</div>` : "") +
        `</div>`
    )
    .join("");

  panel.innerHTML = `
    <button class="detail-close" id="detail-close">×</button>
    <div class="detail-badge" style="background:${hex(c)}">${esc(labelOf(p.status))}</div>
    <div class="detail-title">${esc(p.name || "(no name)")}</div>
    <div class="detail-sub">${esc([p.city, p.county, p.state].filter(Boolean).join(", "))}</div>
    <p class="detail-note">${esc((STATUS[p.status] || {}).hint || "")}${p.reason ? ` — <i>${esc(p.reason)}</i>` : ""}</p>

    <div class="detail-section-title">Location</div>
    <dl class="detail-facts">
      ${fact("CWNS ID", esc(p.id))}
      ${fact("Shown at", `${fmt(p.lat, 6)}, ${fmt(p.lon, 6)}`)}
      ${moved ? fact("Reported at", `${fmt(p.rlat, 6)}, ${fmt(p.rlon, 6)}`) : ""}
      ${moved ? fact("Moved by", fmtDist(p.moved_m)) : ""}
      ${p.coord ? fact("Point from", esc(COORD_LABELS[p.coord] ? COORD_LABELS[p.coord](p) : p.coord)) : ""}
      ${p.insite === false ? fact("Point inside site", '<span style="color:#b45309">no, between parcels</span>') : ""}
      ${p.nsite ? fact("Site parcels", p.nsite) : ""}
      ${p.parcel ? fact("Moved to parcel", `<span style="font-weight:500;font-size:12px">${esc(p.parcel)}</span>`) : ""}
      ${fact("Owner type", esc(p.owner || "—"))}
      ${p.pop !== null && p.pop !== undefined ? fact("Population served", Math.round(p.pop).toLocaleString()) : ""}
      ${p.tier && p.tier !== "none" ? fact("Confidence tier", esc(TIER_LABELS[p.tier] || p.tier)) : ""}
      ${fact("OSM wastewater tag", p.osm === null ? "—" : p.osm ? "yes" : "no")}
    </dl>
    <div class="detail-links">
      <a href="${gmaps(p.lat, p.lon)}" target="_blank">Google Maps (shown)</a>
      ${moved ? `<a href="${gmaps(p.rlat, p.rlon)}" target="_blank">Google Maps (reported)</a>` : ""}
    </div>

    <div class="detail-section-title">Model</div>
    <dl class="detail-facts">
      ${fact("Stage 1 route", esc(p.route || "—"))}
      ${fact("Stage 1 P(correct)", fmt(p.s1))}
      ${fact("Re-rank #1 score", fmt(p.rr))}
      ${fact("Nothing fired (Stage 2a order)", p.fb === null ? "—" : p.fb ? "yes" : "no")}
    </dl>

    <div class="detail-section-title">Candidates</div>
    <div id="cand-box"><div class="cand-note">loading…</div></div>

    <div class="detail-section-title">Review history</div>
    ${reviews || '<div class="detail-note">Not reviewed in any round.</div>'}
  `;
  $("detail-close").addEventListener("click", closePanel);
  panel.classList.add("open");
  document.body.classList.add("panel-open");
}

// ---- explore: candidates, parcels, objects ---------------------------------
async function loadCandidates(p) {
  const box = () => $("cand-box");
  let data;
  try {
    const r = await fetch(`/api/plant/${encodeURIComponent(p.id)}/candidates`);
    data = await r.json();
  } catch (err) {
    if (box()) box().innerHTML = `<div class="cand-note">could not load candidates: ${esc(err.message)}</div>`;
    return;
  }
  if (!selected || selected.id !== p.id) return;
  cands = data.candidates || [];
  renderCandTable(p, data.loaded);
  // Outlines: every candidate plus the reported parcel, from the local mirror.
  const ids = cands.map((c) => c.parcel);
  if (p.rparcel && !ids.includes(p.rparcel)) ids.push(p.rparcel);
  if (!ids.length || !p.state) {
    render();
    return;
  }
  statusEl.textContent = `loading ${ids.length} parcel outline(s)…`;
  try {
    const r = await fetch(`/api/parcels?state=${encodeURIComponent(p.state)}&ids=${ids.map(encodeURIComponent).join(",")}`);
    const fc = await r.json();
    if (!selected || selected.id !== p.id) return;
    if (fc.error) throw new Error(fc.error);
    candParcels = fc;
    const nOk = fc.features.length;
    statusEl.textContent = `${nOk} of ${ids.length} parcel outline(s) found in the local Regrid mirror`;
    const owners = new Map(fc.features.map((f) => [f.properties.ll_uuid, f.properties.owner]));
    document.querySelectorAll(".cand-owner").forEach((el) => {
      const o = owners.get(el.dataset.parcel);
      el.textContent = o || "—";
    });
  } catch (err) {
    statusEl.textContent = `parcel outlines unavailable: ${err.message}`;
  }
  render();
}

function renderCandTable(p, loaded) {
  const box = $("cand-box");
  if (!box) return;
  if (!loaded) {
    box.innerHTML = '<div class="cand-note">No viewer_candidates.parquet yet -- it is written on the HPC by 13\'s job (export_viewer_data.py). Commit correction/diagnostics/output/ and pull.</div>';
    return;
  }
  if (!cands.length) {
    const why = p.route === "osm_confirmed" ? "the reported parcel has an OSM wastewater tag, so it passed by rule"
      : p.s1 !== null && p.s1 !== undefined && (p.status === "kept_model" || p.status === "kept_site") ? `Stage 1 scored the reported location as correct (P = ${fmt(p.s1)})`
      : p.status === "not_assessed" ? "the plant is outside model scope"
      : p.status && p.status.startsWith("verified") ? "a reviewer settled this plant"
      : "the plant was not flagged, or no parcel survived the candidate search";
    box.innerHTML = `<div class="cand-note">No candidates were scored: ${esc(why)}. Turn on <b>All parcels</b> to inspect the area.</div>`;
    return;
  }
  const cut = CUTOFF;
  const rows = [];
  let cutDrawn = false;
  for (const c of cands) {
    if (cut !== null && !cutDrawn && (c.rr === null || c.rr < cut)) {
      rows.push(`<tr class="cutoff"><td colspan="6"></td></tr>`);
      cutDrawn = true;
    }
    const col = c.rank === 1 ? RANK_COLORS[1] : c.rank !== null && c.rank <= 5 ? RANK_COLORS.top5 : RANK_COLORS.rest;
    const od = c.od_ran === false || c.od_ran === null ? '<span class="od-no">not run</span>'
      : c.od ? `<span class="od-yes">${c.nobj ?? "?"} obj</span><div class="sub">${esc((c.odcls || "").replace(/_/g, " "))} ${c.odconf !== null ? (c.odconf * 100).toFixed(0) + "%" : ""}</div>`
      : '<span class="od-no">none</span>';
    const flags = [c.osm ? "OSM" : "", c.kw ? "ww keyword" : "", c.util ? "utility" : ""].filter(Boolean).join(", ");
    rows.push(
      `<tr class="cand-row" data-parcel="${esc(c.parcel)}">` +
        `<td><span class="rank-dot" style="background:${hex(col)}">${c.rank ?? "—"}</span><div class="sub">2a #${c.rank2a ?? "—"}</div></td>` +
        `<td class="num">${fmt(c.rr)}<div class="sub">2a ${fmt(c.s2a)}</div></td>` +
        `<td class="num">${fmtDist(c.dist)}</td>` +
        `<td>${od}</td>` +
        `<td><span class="cand-owner" data-parcel="${esc(c.parcel)}">…</span>` +
          `<div class="sub">${c.acres !== null ? fmt(c.acres, 1) + " ac" : ""}${flags ? " · " + esc(flags) : ""}${c.name ? " · name " + fmt(c.name, 2) : ""}</div></td>` +
      `</tr>`
    );
  }
  const fb = cands.some((c) => c.fb) ? " Nothing fired in this pool, so the order is Stage 2a's." : "";
  box.innerHTML =
    `<div class="cand-note">${cands.length} scored candidate(s), best first.` +
    (cut !== null ? ` Dashed line = move cutoff ${cut}.` : "") + fb + ` Click a row to show the parcel.</div>` +
    `<table class="cand-table"><thead><tr><th>#</th><th class="num">re-rank</th><th class="num">dist</th><th>detection</th><th>owner</th></tr></thead>` +
    `<tbody>${rows.join("")}</tbody></table>`;
  box.querySelectorAll("tr.cand-row").forEach((tr) =>
    tr.addEventListener("click", () => focusParcel(tr.dataset.parcel))
  );
}

function focusParcel(uuid) {
  activeParcel = uuid;
  document.querySelectorAll("tr.cand-row").forEach((tr) => tr.classList.toggle("active", tr.dataset.parcel === uuid));
  const f = candParcels.features.find((x) => x.properties.ll_uuid === uuid) ||
    viewParcels.features.find((x) => x.properties.ll_uuid === uuid);
  const c = cands.find((x) => x.parcel === uuid);
  if (f) showParcelCard(f.properties);
  const lon = c && c.lon !== null ? c.lon : null, lat = c && c.lat !== null ? c.lat : null;
  if (lon !== null && lat !== null) {
    const zoom = Math.max(viewState.zoom, 16);
    const w = window.innerWidth, h = window.innerHeight;
    const panelW = Math.min(400, 0.92 * w);
    const vp = new deck.WebMercatorViewport({ width: w, height: h, longitude: lon, latitude: lat, zoom });
    const [clon, clat] = vp.unproject([w / 2 + panelW / 2, h / 2]);
    viewState = { ...viewState, longitude: clon, latitude: clat, zoom, transitionDuration: 700, transitionInterpolator: new FlyToInterpolator() };
    scheduleViewFetch();
  }
  render();
}

const ATTR_LABELS = [
  ["owner", "Owner"], ["address", "Address"], ["ll_gisacre", "Acres"], ["ll_bldg_count", "Buildings"],
  ["usedesc", "Use"], ["lbcs_activity_desc", "Activity"], ["lbcs_function_desc", "Function"],
  ["lbcs_structure_desc", "Structure"], ["lbcs_site_desc", "Site"], ["lbcs_ownership_desc", "Ownership"],
  ["zoning_type", "Zoning"], ["zoning_subtype", "Zoning subtype"],
];

function showParcelCard(pr) {
  activeParcel = pr.ll_uuid;
  const card = $("parcel-card");
  const fact = (k, v) => `<div class="fact"><dt>${k}</dt><dd>${v}</dd></div>`;
  const attrs = ATTR_LABELS.filter(([k]) => pr[k] !== null && pr[k] !== undefined && pr[k] !== "")
    .map(([k, l]) => fact(l, esc(typeof pr[k] === "number" ? fmt(pr[k], k === "ll_gisacre" ? 2 : 0) : pr[k]))).join("");
  const mine = cands.find((x) => x.parcel === pr.ll_uuid);
  let role = "";
  if (selected && mine) {
    role = `<div class="detail-section-title">Candidate for the selected plant</div><dl>` +
      fact("Rank (re-rank / Stage 2a)", `#${mine.rank ?? "—"} / #${mine.rank2a ?? "—"}`) +
      fact("Re-rank score", fmt(mine.rr) + (CUTOFF !== null ? (mine.rr !== null && mine.rr >= CUTOFF ? " (≥ cutoff)" : " (below cutoff)") : "")) +
      fact("Stage 2a score", fmt(mine.s2a)) +
      fact("Distance from reported point", fmtDist(mine.dist)) +
      fact("Detection", mine.od_ran ? (mine.od ? `${mine.nobj ?? "?"} object(s), max ${fmt(mine.odconf, 2)} ${esc(mine.odcls || "")}` : "ran, nothing found") : "not run") +
      fact("Name match", fmt(mine.name, 2)) +
      fact("OSM wastewater tag / keyword / utility owner", `${mine.osm ? "yes" : "no"} / ${mine.kw ? "yes" : "no"} / ${mine.util ? "yes" : "no"}`) +
      `</dl>`;
  } else if (selected && pr.ll_uuid === selected.rparcel) {
    role = '<div class="cand-note">This is the selected plant\'s <b>reported</b> parcel.</div>';
  }
  const others = (pr.cand || []).filter((x) => !selected || x.id !== selected.id);
  const otherHtml = others.length
    ? `<div class="detail-section-title">Candidate of other plant(s)</div>` +
      others.map((x) => {
        const p = BY_ID.get(x.id);
        return `<div class="pc-cand"><a data-id="${esc(x.id)}">${esc(p ? p.name || x.id : x.id)}</a>` +
          ` <span class="sub">${esc(x.id)}</span><br>rank #${x.rank ?? "—"} · re-rank ${fmt(x.rr)} · detection ${x.od === null ? "not run" : x.od ? "yes" : "no"}` +
          (p ? ` · <span style="color:${hex(colorOf(p))}">${esc(labelOf(p.status))}</span>` : "") + `</div>`;
      }).join("")
    : (pr.cand ? '<div class="cand-note">Not a scored candidate of any other plant.</div>' : "");
  card.innerHTML =
    `<button class="pc-close" id="pc-close">×</button>` +
    `<div class="pc-title">${esc(pr.owner || "(no owner)")}</div>` +
    `<div class="pc-sub">parcel ${esc(pr.ll_uuid)} · ${esc(pr.state || "")}</div>` +
    `<dl>${attrs}</dl>${role}${otherHtml}`;
  card.classList.add("open");
  $("pc-close").onclick = () => { card.classList.remove("open"); activeParcel = null; render(); };
  card.querySelectorAll("a[data-id]").forEach((a) =>
    a.addEventListener("click", () => { const p = BY_ID.get(a.dataset.id); if (p) select(p, true); })
  );
  render();
}

function viewBounds() {
  const vp = new deck.WebMercatorViewport({ ...viewState, width: window.innerWidth, height: window.innerHeight });
  const [w, s, e, n] = vp.getBounds();
  return { w, s, e, n };
}

let viewTimer = null;
function scheduleViewFetch() {
  clearTimeout(viewTimer);
  viewTimer = setTimeout(fetchViewData, 450);
}

async function fetchViewData() {
  const b = viewBounds();
  const q = `w=${b.w}&s=${b.s}&e=${b.e}&n=${b.n}`;
  const seq = ++viewSeq;
  const jobs = [];
  if (showParcels && viewState.zoom >= PARCELS_MIN_ZOOM) {
    statusEl.textContent = "loading parcels in view…";
    jobs.push(fetch(`/api/parcels_in_view?${q}`).then((r) => r.json()).then((fc) => {
      if (seq !== viewSeq) return;
      if (fc.error) { statusEl.textContent = fc.error; return; }
      viewParcels = fc;
      statusEl.textContent = fc.too_big ? "zoom in further to load parcels"
        : `${fc.features.length.toLocaleString()} parcel(s) in view (${(fc.states || []).join(", ")})` + (fc.truncated ? " -- truncated, zoom in" : "");
    }).catch((err) => { statusEl.textContent = `parcels unavailable: ${err.message}`; }));
  }
  if (showAllObj && viewState.zoom >= ALLOBJ_MIN_ZOOM) {
    jobs.push(fetch(`/api/objects?${q}`).then((r) => r.json()).then((d) => {
      if (seq === viewSeq && Array.isArray(d)) viewObjects = d;
    }).catch(() => {}));
  }
  await Promise.all(jobs);
  if (seq === viewSeq) render();
}

// ---- legend / filters -----------------------------------------------------
function buildLegend(counts) {
  const box = $("status-filters");
  box.innerHTML = "";
  for (const [key, s] of Object.entries(STATUS)) {
    const n = counts[key] || 0;
    if (!n && key === "pending") continue;
    const row = document.createElement("label");
    row.className = "row" + (n ? "" : " empty");
    row.title = s.hint;
    row.innerHTML =
      `<input type="checkbox" data-status="${key}" checked>` +
      `<span class="swatch" style="background:${hex(s.color)}"></span>${s.label}` +
      `<span class="n">${n.toLocaleString()}</span>`;
    box.appendChild(row);
  }
  const links = document.createElement("div");
  links.className = "legend-links";
  links.innerHTML = '<a id="all-on">all</a><a id="all-off">none</a><a id="only-model">model decisions only</a>';
  box.appendChild(links);

  box.addEventListener("change", (e) => {
    const k = e.target.dataset.status;
    if (!k) return;
    e.target.checked ? enabled.add(k) : enabled.delete(k);
    applyFilters();
  });
  const setAll = (keys) => {
    enabled.clear();
    keys.forEach((k) => enabled.add(k));
    box.querySelectorAll("input[data-status]").forEach((i) => (i.checked = enabled.has(i.dataset.status)));
    applyFilters();
  };
  $("all-on").onclick = () => setAll(Object.keys(STATUS));
  $("all-off").onclick = () => setAll([]);
  $("only-model").onclick = () => setAll(["moved", "flagged_not_moved", "kept_osm", "kept_model"]);
}

function buildStateFilter() {
  const states = [...new Set(ALL.map((p) => p.state).filter(Boolean))].sort();
  const sel = $("state-filter");
  for (const s of states) {
    const o = document.createElement("option");
    o.value = o.textContent = s;
    sel.appendChild(o);
  }
  sel.addEventListener("change", () => {
    stateFilter = sel.value;
    applyFilters();
    if (stateFilter) fitTo(filtered);
  });
}

function fitTo(points) {
  if (!points.length) return;
  let w = 180, s = 90, e = -180, n = -90;
  for (const p of points) {
    w = Math.min(w, p.lon); e = Math.max(e, p.lon);
    s = Math.min(s, p.lat); n = Math.max(n, p.lat);
  }
  const vp = new deck.WebMercatorViewport({ width: window.innerWidth, height: window.innerHeight });
  const { longitude, latitude, zoom } = vp.fitBounds([[w, s], [e, n]], { padding: 80 });
  viewState = {
    ...viewState, longitude, latitude, zoom: Math.min(zoom, 12),
    transitionDuration: 900, transitionInterpolator: new FlyToInterpolator(),
  };
  render();
}

// ---- search ---------------------------------------------------------------
function setupSearch() {
  const input = $("search-input");
  const sugg = $("search-suggestions");
  let results = [];
  let hi = -1;

  function search(q) {
    q = q.trim().toLowerCase();
    if (!q) return [];
    const out = [];
    for (const p of ALL) {
      if (p.id.startsWith(q) || (p.name && p.name.toLowerCase().includes(q))) {
        out.push(p);
        if (out.length >= 15) break;
      }
    }
    return out;
  }
  function draw() {
    if (!input.value.trim()) {
      sugg.style.display = "none";
      return;
    }
    sugg.innerHTML = results.length
      ? results
          .map(
            (p, i) =>
              `<div class="suggestion-item${i === hi ? " highlighted" : ""}" data-index="${i}">` +
              `<span class="swatch" style="width:10px;height:10px;border-radius:50%;background:${hex(colorOf(p))}"></span>` +
              `<span class="suggestion-name">${esc(p.name || p.id)}</span>` +
              `<span class="suggestion-id">${esc(p.state || "")} ${esc(p.id)}</span></div>`
          )
          .join("")
      : '<div class="suggestion-empty">No matches</div>';
    sugg.style.display = "block";
  }
  function choose(p) {
    if (!p) return;
    sugg.style.display = "none";
    input.value = p.name || p.id;
    select(p, true);
  }
  input.addEventListener("input", () => {
    results = search(input.value);
    hi = results.length ? 0 : -1;
    draw();
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { hi = Math.min(results.length - 1, hi + 1); draw(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { hi = Math.max(0, hi - 1); draw(); e.preventDefault(); }
    else if (e.key === "Enter") choose(results[hi] || results[0]);
    else if (e.key === "Escape") sugg.style.display = "none";
  });
  $("search-button").addEventListener("click", () => choose(results[hi] || results[0]));
  sugg.addEventListener("click", (e) => {
    const item = e.target.closest(".suggestion-item");
    if (item) choose(results[Number(item.dataset.index)]);
  });
  document.addEventListener("click", (e) => {
    if (!e.target.closest("#search-bar")) sugg.style.display = "none";
  });
}

// ---- basemap / controls ---------------------------------------------------
function setBasemap(key) {
  basemap = key;
  $("basemap-street").classList.toggle("active", key === "street");
  $("basemap-imagery").classList.toggle("active", key === "imagery");
  $("map-attribution").innerHTML = BASEMAPS[key].attribution;
  render();
}

// ---- boot -----------------------------------------------------------------
async function main() {
  deckgl = new DeckGL({
    container: "map",
    viewState,
    controller: true,
    layers: [basemapLayer(basemap)],
    getCursor: ({ isHovering, isDragging }) => (isDragging ? "grabbing" : isHovering ? "pointer" : "grab"),
    onViewStateChange: ({ viewState: vs, interactionState }) => {
      // Once a fly-to has finished, drop its transition props: otherwise every
      // later render() re-sends them, deck starts the transition again, and
      // the view-fetch -> render cycle never settles.
      if (!(interactionState && interactionState.inTransition)) {
        const { transitionDuration, transitionInterpolator, transitionEasing, transitionInterruption, ...rest } = vs;
        vs = rest;
      }
      const bucketChanged = Math.round(vs.zoom * 2) !== Math.round(viewState.zoom * 2) ||
        (vs.zoom >= REPORTED_MIN_ZOOM) !== (viewState.zoom >= REPORTED_MIN_ZOOM);
      viewState = vs;
      if (bucketChanged) render();
      else deckgl.setProps({ viewState });
      if (showParcels || showAllObj) scheduleViewFetch();
    },
    onHover,
    onClick,
  });

  $("basemap-street").onclick = () => setBasemap("street");
  $("basemap-imagery").onclick = () => setBasemap("imagery");
  $("show-moves").onchange = (e) => { showMoves = e.target.checked; render(); };
  $("show-sites").onchange = (e) => { showSites = e.target.checked; render(); };
  $("show-dets").onchange = (e) => { showDets = e.target.checked; render(); };
  $("show-parcels").onchange = (e) => { showParcels = e.target.checked; if (showParcels) fetchViewData(); else render(); };
  $("show-allobj").onchange = (e) => { showAllObj = e.target.checked; if (showAllObj) fetchViewData(); else render(); };
  $("size-slider").oninput = (e) => { sizeBase = Number(e.target.value); render(); };
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && panel.classList.contains("open")) closePanel();
  });
  setBasemap("street");

  statusEl.textContent = "loading plants…";
  const res = await fetch(DATA_URL);
  if (!res.ok) {
    statusEl.textContent = `failed to load ${DATA_URL}: HTTP ${res.status}`;
    return;
  }
  const data = await res.json();
  ALL = data.plants;
  CUTOFF = data.cutoff ?? null;
  BY_ID = new Map(ALL.map((p) => [p.id, p]));
  buildLegend(data.counts || {});
  buildStateFilter();
  setupSearch();
  statusEl.textContent =
    `${data.source}` + (data.cutoff !== null && data.cutoff !== undefined ? ` · move cutoff ${data.cutoff}` : "");
  applyFilters();

  // Sites and detections are secondary: load them after the plants are up.
  Promise.all([
    fetch(SITES_URL).then((r) => (r.ok ? r.json() : SITES)),
    fetch(DETECTIONS_URL).then((r) => (r.ok ? r.json() : [])),
  ])
    .then(([s, d]) => {
      SITES = s;
      DETS = d;
      $("layer-counts").textContent =
        `${SITES.features.length.toLocaleString()} sites · ${DETS.length.toLocaleString()} detections`;
      applyFilters();
    })
    .catch((err) => console.warn("sites/detections not loaded", err));

  const fromHash = () => {
    const m = location.hash.match(/id=([^&]+)/);
    const id = m && decodeURIComponent(m[1]);
    if (id && BY_ID.has(id) && (!selected || selected.id !== id)) select(BY_ID.get(id), true);
  };
  window.addEventListener("hashchange", fromHash);
  fromHash();
}

main().catch((err) => {
  console.error(err);
  statusEl.textContent = `error: ${err.message}`;
});
