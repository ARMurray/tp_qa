// CWNS treatment-plant location viewer. Pattern follows sewershed_plus's
// map.js: deck.gl, raster basemap TileLayer + BitmapLayer, search bar with
// suggestions, legend that doubles as the filter panel, slide-in detail panel.
// No vector tiles: every plant arrives in one JSON (see app.py).

const { DeckGL, TileLayer, BitmapLayer, ScatterplotLayer, LineLayer, FlyToInterpolator } = deck;

const DATA_URL = "/data/plants.json";

// Order here = legend order (most important first). Colours chosen to stay
// distinct on both the street and imagery basemaps.
const STATUS = {
  moved:               { label: "Moved (model)",          color: [249, 115, 22],  hint: "Stage 1 flagged it and the re-ranker's #1 parcel cleared the cutoff" },
  flagged_not_moved:   { label: "Flagged, not moved",     color: [220, 38, 38],   hint: "Stage 1 doubts the reported location; no candidate cleared the cutoff" },
  verified_corrected:  { label: "Verified – corrected",   color: [37, 99, 235],   hint: "A reviewer supplied the corrected location" },
  verified_correct:    { label: "Verified – correct",     color: [22, 163, 74],   hint: "A reviewer confirmed the reported location" },
  reviewed_unresolved: { label: "Reviewed – unresolved",  color: [147, 51, 234],  hint: "Reviewed as wrong, no corrected location yet" },
  kept_osm:            { label: "Kept – OSM tagged",      color: [20, 184, 166],  hint: "Reported parcel carries an OSM wastewater tag" },
  kept_model:          { label: "Kept – model",           color: [163, 230, 53],  hint: "Stage 1 scored the reported location as correct" },
  pending:             { label: "Pending (no model run)", color: [156, 163, 175], hint: "Preview mode: model output not loaded yet" },
  not_assessed:        { label: "Not assessed",           color: [107, 114, 128], hint: "Outside model scope (no NAIP, population ≤ 1,000, …)" },
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
let viewState = { longitude: -96.5, latitude: 38.5, zoom: 3.8, pitch: 0, bearing: 0 };

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
  $("visible-count").textContent = `${filtered.length.toLocaleString()} of ${ALL.length.toLocaleString()} plants shown`;
  render();
}

// ---- layers ---------------------------------------------------------------
function layers() {
  const r = radiusPx();
  const zoomBucket = Math.round(viewState.zoom * 2);
  const out = [basemapLayer(basemap)];

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

function onClick({ object }) {
  if (object) select(object, false);
}

function select(p, fly) {
  selected = p;
  showPanel(p);
  history.replaceState(null, "", `#id=${encodeURIComponent(p.id)}`);
  if (fly) {
    viewState = {
      ...viewState,
      longitude: p.lon,
      latitude: p.lat,
      zoom: Math.max(viewState.zoom, 15),
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
  history.replaceState(null, "", location.pathname);
  render();
}

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
      ${p.parcel ? fact("Moved to parcel", `<span style="font-weight:500;font-size:12px">${esc(p.parcel)}</span>`) : ""}
      ${fact("Owner type", esc(p.owner || "—"))}
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

    <div class="detail-section-title">Review history</div>
    ${reviews || '<div class="detail-note">Not reviewed in any round.</div>'}
  `;
  $("detail-close").addEventListener("click", closePanel);
  panel.classList.add("open");
  document.body.classList.add("panel-open");
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
    onViewStateChange: ({ viewState: vs }) => {
      const bucketChanged = Math.round(vs.zoom * 2) !== Math.round(viewState.zoom * 2) ||
        (vs.zoom >= REPORTED_MIN_ZOOM) !== (viewState.zoom >= REPORTED_MIN_ZOOM);
      viewState = vs;
      if (bucketChanged) render();
      else deckgl.setProps({ viewState });
    },
    onHover,
    onClick,
  });

  $("basemap-street").onclick = () => setBasemap("street");
  $("basemap-imagery").onclick = () => setBasemap("imagery");
  $("show-moves").onchange = (e) => { showMoves = e.target.checked; render(); };
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
  BY_ID = new Map(ALL.map((p) => [p.id, p]));
  buildLegend(data.counts || {});
  buildStateFilter();
  setupSearch();
  statusEl.textContent =
    `${data.source}` + (data.cutoff !== null && data.cutoff !== undefined ? ` · move cutoff ${data.cutoff}` : "");
  applyFilters();

  const m = location.hash.match(/id=([^&]+)/);
  if (m && BY_ID.has(decodeURIComponent(m[1]))) select(BY_ID.get(decodeURIComponent(m[1])), true);
}

main().catch((err) => {
  console.error(err);
  statusEl.textContent = `error: ${err.message}`;
});
