###############################################################################
# label_app.R
# =============================================================================
# A bounding-box annotation app for NAIP tiles, written to produce output that
# is byte-compatible with what Label Studio's YOLO export gave you, so
# 03_prepare_dataset.py needs no changes at all.
#
# OUTPUT CONTRACT (this is the part that matters):
#   - One .txt per image, named {tile_stem}_rgb.txt  -- the same stem as the
#     PNG. 03_prepare_dataset.py's parse_tile_stem() strips the trailing "_rgb"
#     and matches against build_rgb_index(), so this drops straight in.
#   - Each line: "class_id cx cy w h", all normalized 0-1, YOLO convention.
#   - A tile reviewed with NO objects gets a ZERO-BYTE .txt file. That is not a
#     bug: is_negative() in 03 checks file size, and INCLUDE_NEGATIVES = TRUE
#     means those tiles train the model on "nothing here". Skipping a tile
#     entirely and marking it empty are different things, and the app keeps
#     them different.
#   - classes.txt is written in ALPHABETICAL order to match config.py's CLASSES
#     list. Do not reorder it to match the old Label Studio UI order (which had
#     clarifier first) -- class IDs are positional and everything downstream
#     assumes alphabetical.
#
# FIXES 2026-08-28, before this app was ever used (moved into
# tp_qa/detection/OWM_Imagery_Labeler/ at the same time):
#   1. Solo use (N_LABELERS=1) now writes straight to labels/, not a
#      labels_<user>/ shard requiring a manual merge_labels() run.
#   2. The Enter-key "Add box" shortcut was a non-functional stub.
#   3. Added guidance to pin here()'s root explicitly with a .here marker.
#
# REWRITE 2026-09-02 -- canvas-based editor:
#   The drawing surface is no longer a base R plot. Shiny's plotOutput only
#   exposes click, dblclick, hover and brush; it has no mousedown/mousemove/
#   mouseup. Dragging an existing box to move or resize it, and dragging the
#   background to pan, are therefore not expressible against plotOutput at all
#   -- not awkward, actually impossible. The image now renders into an HTML
#   <canvas> that owns pointer interaction directly.
#
#   Division of responsibility:
#     - JS owns box geometry WHILE THE MOUSE IS DOWN, and pushes the complete
#       box array to R on mouseup (and on any discrete edit). It never pushes
#       mid-drag, so R isn't re-rendering 60 times a second.
#     - R owns files, the image list, the box table, dirty state and saving.
#       R pushes boxes TO the canvas only on tile load and on explicit
#       server-side edits, never from an observer watching rv$boxes -- that
#       would echo every JS push straight back and loop.
#
# RUN:
#   install.packages(c("shiny", "png", "DT", "jsonlite", "here"))
#   shiny::runApp("label_app.R")
###############################################################################

library(shiny)
library(png)
library(here)
library(DT)

`%||%` <- function(a, b) if (is.null(a)) b else a

# =============================================================================
# CONFIG -- the only block you should need to edit
# =============================================================================

REPO_ROOT   <- here()
# here() finds its root via heuristics (nearest .here/.Rproj/.git file,
# walking UP from the working directory) -- it does NOT mean "the folder this
# script lives in." Create an empty file named exactly ".here" in this same
# directory (OWM_Imagery_Labeler/); the `here` package checks for that marker
# FIRST, before .Rproj or git root, so it settles this unambiguously.
IMAGE_DIR   <- file.path(REPO_ROOT, "data", "tiles", "rgb", "png")
OUTPUT_ROOT <- file.path(REPO_ROOT, "annotation", "ls_export")

# all_images <- sort(list.files(IMAGE_DIR, pattern = "_500m_rgb\\.png$"))

# Alphabetical -- matches config.py CLASSES. Index-1 here == YOLO class_id.
CLASSES <- c(
  "aeration_basin",
  "chlorine_contact",
  "clarifier",
  "digester",
  "drying_bed",
  "oxidation_pond"
)

# Carried over from your Label Studio labeling interface so boxes look familiar.
CLASS_COLORS <- c(
  aeration_basin   = "#0000FF",
  chlorine_contact = "#1cf2d9",
  clarifier        = "#FF0000",
  digester         = "#FF8800",
  drying_bed       = "#FFA39E",
  oxidation_pond   = "#00AA00"
)

# --- SHARDING ---------------------------------------------------------------
# Set N_LABELERS to the number of people working, and give each person a unique
# MY_SHARD from 1..N_LABELERS. Tiles are split by CWNS_ID (not by tile), so all
# tiles belonging to one plant go to the same person.
N_LABELERS <- 1
MY_SHARD   <- 1

LABELER_ID <- Sys.info()[["user"]]
LABEL_DIR  <- if (N_LABELERS == 1) {
  file.path(OUTPUT_ROOT, "labels")
} else {
  file.path(OUTPUT_ROOT, paste0("labels_", LABELER_ID))
}

# =============================================================================
# Setup
# =============================================================================

dir.create(LABEL_DIR, recursive = TRUE, showWarnings = FALSE)
writeLines(CLASSES, file.path(OUTPUT_ROOT, "classes.txt"))

# Lets the browser fetch tile PNGs directly, so the canvas can draw them
# without R round-tripping pixel data on every zoom.
addResourcePath("tiles", IMAGE_DIR)

shard_of <- function(filename) {
  cwns <- sub("_.*$", "", filename)
  (sum(utf8ToInt(cwns)) %% N_LABELERS) + 1L
}

all_images <- sort(list.files(IMAGE_DIR, pattern = "_rgb\\.png$"))
if (length(all_images) == 0) {
  stop("No *_rgb.png files found in ", IMAGE_DIR,
       " -- check IMAGE_DIR at the top of this script.")
}
if (N_LABELERS > 1) {
  all_images <- all_images[vapply(all_images, shard_of, integer(1)) == MY_SHARD]
}

label_path <- function(png_name) {
  file.path(LABEL_DIR, sub("\\.png$", ".txt", png_name))
}

is_done <- function(png_name) file.exists(label_path(png_name))

n_boxes_on_disk <- function(png_name) {
  p <- label_path(png_name)
  if (!file.exists(p)) return(NA_integer_)
  if (file.info(p)$size == 0) return(0L)
  length(readLines(p, warn = FALSE))
}

empty_boxes <- function() {
  data.frame(class_id = integer(), xmin = numeric(), ymin = numeric(),
             xmax = numeric(), ymax = numeric())
}

read_boxes <- function(png_name, W, H) {
  p <- label_path(png_name)
  if (!file.exists(p) || file.info(p)$size == 0) return(empty_boxes())
  lines <- readLines(p, warn = FALSE)
  lines <- lines[nzchar(trimws(lines))]
  if (length(lines) == 0) return(empty_boxes())
  parts <- do.call(rbind, lapply(strsplit(trimws(lines), "\\s+"), as.numeric))
  data.frame(
    class_id = as.integer(parts[, 1]),
    xmin = (parts[, 2] - parts[, 4] / 2) * W,
    ymin = (parts[, 3] - parts[, 5] / 2) * H,
    xmax = (parts[, 2] + parts[, 4] / 2) * W,
    ymax = (parts[, 3] + parts[, 5] / 2) * H
  )
}

write_boxes <- function(png_name, boxes, W, H) {
  p <- label_path(png_name)
  if (is.null(boxes) || nrow(boxes) == 0) {
    file.create(p)            # zero-byte file == confirmed negative
    return(invisible(NULL))
  }
  cx <- (boxes$xmin + boxes$xmax) / 2 / W
  cy <- (boxes$ymin + boxes$ymax) / 2 / H
  w  <- (boxes$xmax - boxes$xmin) / W
  h  <- (boxes$ymax - boxes$ymin) / H
  writeLines(sprintf("%d %.6f %.6f %.6f %.6f", boxes$class_id, cx, cy, w, h), p)
}

# Dimensions are needed to convert between YOLO's normalized coordinates and
# the canvas's pixel coordinates. Cached because it means decoding the PNG.
png_dims <- local({
  cache <- new.env(parent = emptyenv())
  function(png_name) {
    hit <- cache[[png_name]]
    if (!is.null(hit)) return(hit)
    a <- readPNG(file.path(IMAGE_DIR, png_name))
    d <- c(W = dim(a)[2], H = dim(a)[1])
    cache[[png_name]] <- d
    d
  }
})

boxes_from_json <- function(txt) {
  if (is.null(txt) || !nzchar(txt)) return(empty_boxes())
  b <- jsonlite::fromJSON(txt, simplifyDataFrame = TRUE)
  if (is.null(b) || !is.data.frame(b) || nrow(b) == 0) return(empty_boxes())
  data.frame(
    class_id = as.integer(b$class_id),
    xmin = as.numeric(b$xmin), ymin = as.numeric(b$ymin),
    xmax = as.numeric(b$xmax), ymax = as.numeric(b$ymax)
  )
}

# =============================================================================
# Canvas editor (client side)
# =============================================================================

CANVAS_JS <- "
(function () {
  var CLASSES = null, COLORS = null;
  var cv, ctx, img = new Image(), imgReady = false;
  var boxes = [], sel = -1, W = 1, H = 1;
  var view = { s: 1, ox: 0, oy: 0 };
  var drag = null;
  var HANDLE_PX = 7;   // corner grab radius, in SCREEN pixels
  var MIN_SIDE = 3;    // smallest box we'll create, in IMAGE pixels

  function px(e) {
    var r = cv.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }
  function toImg(p) { return { x: (p.x - view.ox) / view.s, y: (p.y - view.oy) / view.s }; }
  function toScr(x, y) { return { x: x * view.s + view.ox, y: y * view.s + view.oy }; }

  function fitView() {
    var cw = cv.clientWidth, ch = cv.clientHeight;
    view.s = Math.min(cw / W, ch / H) * 0.96;
    view.ox = (cw - W * view.s) / 2;
    view.oy = (ch - H * view.s) / 2;
  }

  function resize() {
    if (!cv) return;
    var dpr = window.devicePixelRatio || 1;
    cv.width  = cv.clientWidth  * dpr;
    cv.height = cv.clientHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function norm(b) {
    return { class_id: b.class_id,
             xmin: Math.min(b.xmin, b.xmax), xmax: Math.max(b.xmin, b.xmax),
             ymin: Math.min(b.ymin, b.ymax), ymax: Math.max(b.ymin, b.ymax) };
  }
  function clampBox(b) {
    b = norm(b);
    b.xmin = Math.max(0, Math.min(W, b.xmin));
    b.xmax = Math.max(0, Math.min(W, b.xmax));
    b.ymin = Math.max(0, Math.min(H, b.ymin));
    b.ymax = Math.max(0, Math.min(H, b.ymax));
    return b;
  }

  function push() {
    Shiny.setInputValue('boxes_json', JSON.stringify(boxes), { priority: 'event' });
    Shiny.setInputValue('sel_idx', sel + 1, { priority: 'event' });
  }

  function corners(b) {
    return [[b.xmin, b.ymin], [b.xmax, b.ymin], [b.xmax, b.ymax], [b.xmin, b.ymax]];
  }

  function hitHandle(p) {
    if (sel < 0 || sel >= boxes.length) return -1;
    var cs = corners(boxes[sel]);
    for (var i = 0; i < 4; i++) {
      var s = toScr(cs[i][0], cs[i][1]);
      if (Math.abs(s.x - p.x) <= HANDLE_PX && Math.abs(s.y - p.y) <= HANDLE_PX) return i;
    }
    return -1;
  }

  function hitBox(ip) {
    for (var i = boxes.length - 1; i >= 0; i--) {   // topmost wins
      var b = boxes[i];
      if (ip.x >= b.xmin && ip.x <= b.xmax && ip.y >= b.ymin && ip.y <= b.ymax) return i;
    }
    return -1;
  }

  function activeClassId() {
    var el = document.querySelector('input[name=\\'class\\']:checked');
    if (!el) return 0;
    var i = CLASSES.indexOf(el.value);
    return i < 0 ? 0 : i;
  }

  function draw() {
    if (!ctx) return;
    var cw = cv.clientWidth, ch = cv.clientHeight;
    ctx.clearRect(0, 0, cw, ch);
    ctx.fillStyle = '#1b1d21';
    ctx.fillRect(0, 0, cw, ch);
    if (!imgReady) return;

    ctx.imageSmoothingEnabled = view.s < 1;
    ctx.drawImage(img, view.ox, view.oy, W * view.s, H * view.s);

    for (var i = 0; i < boxes.length; i++) {
      var b = boxes[i];
      var a = toScr(b.xmin, b.ymin), c = toScr(b.xmax, b.ymax);
      var col = COLORS[CLASSES[b.class_id]] || '#ffffff';
      ctx.lineWidth = (i === sel) ? 3 : 2;
      ctx.strokeStyle = col;
      ctx.setLineDash(i === sel ? [6, 4] : []);
      ctx.strokeRect(a.x, a.y, c.x - a.x, c.y - a.y);
      ctx.setLineDash([]);

      ctx.font = '12px Segoe UI, system-ui, sans-serif';
      var label = CLASSES[b.class_id];
      var tw = ctx.measureText(label).width;
      ctx.fillStyle = col;
      ctx.fillRect(a.x, a.y - 15, tw + 8, 15);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, a.x + 4, a.y - 4);

      if (i === sel) {
        var cs = corners(b);
        for (var k = 0; k < 4; k++) {
          var s = toScr(cs[k][0], cs[k][1]);
          ctx.fillStyle = '#fff';
          ctx.strokeStyle = col;
          ctx.lineWidth = 2;
          ctx.fillRect(s.x - 4, s.y - 4, 8, 8);
          ctx.strokeRect(s.x - 4, s.y - 4, 8, 8);
        }
      }
    }

    if (drag && drag.mode === 'draw' && drag.preview) {
      var p0 = toScr(drag.preview.xmin, drag.preview.ymin);
      var p1 = toScr(drag.preview.xmax, drag.preview.ymax);
      ctx.setLineDash([5, 3]);
      ctx.strokeStyle = COLORS[CLASSES[activeClassId()]] || '#fff';
      ctx.lineWidth = 2;
      ctx.strokeRect(p0.x, p0.y, p1.x - p0.x, p1.y - p0.y);
      ctx.setLineDash([]);
    }

    ctx.fillStyle = 'rgba(255,255,255,0.75)';
    ctx.font = '11px Consolas, monospace';
    ctx.fillText(Math.round(view.s * 100) + '%', 8, ch - 8);
  }

  function onDown(e) {
    if (!imgReady) return;
    cv.focus();
    var p = px(e), ip = toImg(p);

    if (e.button === 2 || e.button === 1 || e.shiftKey) {
      drag = { mode: 'pan', from: p, ox: view.ox, oy: view.oy };
      e.preventDefault();
      return;
    }
    if (e.button !== 0) return;

    var h = hitHandle(p);
    if (h >= 0) {
      drag = { mode: 'resize', corner: h, orig: Object.assign({}, boxes[sel]) };
      return;
    }
    var hb = hitBox(ip);
    if (hb >= 0) {
      if (sel !== hb) { sel = hb; push(); }
      drag = { mode: 'move', startImg: ip, orig: Object.assign({}, boxes[hb]) };
      draw();
      return;
    }
    if (sel !== -1) { sel = -1; push(); }
    drag = { mode: 'draw', startImg: ip, preview: null };
    draw();
  }

  function onMove(e) {
    if (!imgReady || !cv) return;
    var p = px(e), ip = toImg(p);

    if (!drag) {
      cv.style.cursor = hitHandle(p) >= 0 ? 'nwse-resize'
                      : (hitBox(ip) >= 0 ? 'move' : 'crosshair');
      return;
    }

    if (drag.mode === 'pan') {
      view.ox = drag.ox + (p.x - drag.from.x);
      view.oy = drag.oy + (p.y - drag.from.y);
    } else if (drag.mode === 'draw') {
      drag.preview = clampBox({ class_id: activeClassId(),
                                xmin: drag.startImg.x, ymin: drag.startImg.y,
                                xmax: ip.x, ymax: ip.y });
    } else if (drag.mode === 'move') {
      var dx = ip.x - drag.startImg.x, dy = ip.y - drag.startImg.y;
      var o = drag.orig, bw = o.xmax - o.xmin, bh = o.ymax - o.ymin;
      var nx = Math.max(0, Math.min(W - bw, o.xmin + dx));
      var ny = Math.max(0, Math.min(H - bh, o.ymin + dy));
      boxes[sel] = { class_id: o.class_id, xmin: nx, ymin: ny,
                     xmax: nx + bw, ymax: ny + bh };
    } else if (drag.mode === 'resize') {
      var o2 = drag.orig, nb = Object.assign({}, o2);
      if (drag.corner === 0) { nb.xmin = ip.x; nb.ymin = ip.y; }
      if (drag.corner === 1) { nb.xmax = ip.x; nb.ymin = ip.y; }
      if (drag.corner === 2) { nb.xmax = ip.x; nb.ymax = ip.y; }
      if (drag.corner === 3) { nb.xmin = ip.x; nb.ymax = ip.y; }
      boxes[sel] = clampBox(nb);
    }
    draw();
  }

  function onUp() {
    if (!drag) return;
    var d = drag; drag = null;

    if (d.mode === 'draw') {
      var b = d.preview;
      if (b && (b.xmax - b.xmin) >= MIN_SIDE && (b.ymax - b.ymin) >= MIN_SIDE) {
        b.class_id = activeClassId();
        boxes.push(b);
        sel = boxes.length - 1;
        push();
      }
    } else if (d.mode === 'move' || d.mode === 'resize') {
      boxes[sel] = clampBox(boxes[sel]);
      push();
    }
    draw();
  }

  function onWheel(e) {
    if (!imgReady) return;
    e.preventDefault();
    var p = px(e);
    var before = toImg(p);
    var f = Math.pow(1.0015, -e.deltaY);
    view.s = Math.max(0.1, Math.min(40, view.s * f));
    view.ox = p.x - before.x * view.s;
    view.oy = p.y - before.y * view.s;
    draw();
  }

  function deleteSelected() {
    if (sel < 0) return;
    boxes.splice(sel, 1);
    sel = -1;
    push();
    draw();
  }

  function onKey(e) {
    var t = e.target.tagName;
    if (t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT') return;

    if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault(); deleteSelected(); return;
    }
    if (e.key === 'Escape') { sel = -1; push(); draw(); return; }
    if (e.key === 'f' || e.key === 'F') { fitView(); draw(); return; }
    if (CLASSES && e.key >= '1' && e.key <= String(CLASSES.length)) {
      var id = parseInt(e.key, 10) - 1;
      Shiny.setInputValue('set_class', CLASSES[id], { priority: 'event' });
      if (sel >= 0) { boxes[sel].class_id = id; push(); draw(); }
      return;
    }
    if (e.key === 'n' || e.key === 'N') {
      Shiny.setInputValue('key_next', Math.random(), { priority: 'event' }); return;
    }
    if (e.key === 'b' || e.key === 'B') {
      Shiny.setInputValue('key_prev', Math.random(), { priority: 'event' }); return;
    }
    if (e.key === 's' || e.key === 'S') {
      Shiny.setInputValue('key_save', Math.random(), { priority: 'event' }); return;
    }
  }

  function init() {
    cv = document.getElementById('editor');
    if (!cv || ctx) return;
    ctx = cv.getContext('2d');
    cv.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    cv.addEventListener('wheel', onWheel, { passive: false });
    cv.addEventListener('contextmenu', function (e) { e.preventDefault(); });
    document.addEventListener('keydown', onKey);
    window.addEventListener('resize', resize);
    img.onload = function () {
      W = img.naturalWidth; H = img.naturalHeight;
      imgReady = true;
      fitView();
      resize();
    };
    resize();
  }

  $(document).on('shiny:connected', init);

  Shiny.addCustomMessageHandler('init_editor', function (m) {
    CLASSES = m.classes; COLORS = m.colors;
    init();
  });

  Shiny.addCustomMessageHandler('load_tile', function (m) {
    init();
    boxes = m.boxes || [];
    sel = -1;
    imgReady = false;
    img.src = 'tiles/' + encodeURIComponent(m.name);
    draw();
  });

  // Server-side edits (delete button, clear, box-table selection) come back
  // through here. Deliberately NOT wired to an observer on the box data --
  // that would echo every JS push back to the client and loop.
  Shiny.addCustomMessageHandler('set_boxes', function (m) {
    boxes = m.boxes || [];
    sel = (typeof m.sel === 'number') ? m.sel - 1 : -1;
    draw();
  });
})();
"

# =============================================================================
# UI
# =============================================================================

ui <- fluidPage(
  tags$head(tags$style(HTML("
    body { font-family: 'Segoe UI', system-ui, sans-serif; }
    .tile-name { font-family: Consolas, monospace; font-size: 11px;
                 word-break: break-all; color: #444; margin-bottom: 4px; }
    .swatch { width: 13px; height: 13px; border: 1px solid #0006;
              display: inline-block; vertical-align: middle; }
    .counter { font-size: 20px; font-weight: 600; }
    .hint { color: #666; font-size: 12px; line-height: 1.6; }
    #editor { width: 100%; height: 660px; display: block;
              border: 1px solid #ccc; border-radius: 3px;
              background: #1b1d21; outline: none; touch-action: none; }
    .savebar { display: flex; align-items: center; gap: 12px; margin-top: 8px;
               flex-wrap: wrap; min-height: 30px; font-size: 13px; }
    .badge-dirty { background: #b3261e; color: #fff; padding: 3px 9px;
                   border-radius: 3px; font-weight: 600; }
    .badge-clean { background: #1e7d34; color: #fff; padding: 3px 9px;
                   border-radius: 3px; font-weight: 600; }
    .save-detail { color: #555; font-family: Consolas, monospace; font-size: 11px; }
    .panel-head { font-weight: 600; margin-bottom: 6px; }
    table.dataTable td { padding: 3px 6px !important; font-size: 12px; }
  "))),
  tags$script(HTML(CANVAS_JS)),

  titlePanel("NAIP tile labeler"),

  fluidRow(
    # ---- left: image browser -----------------------------------------------
    column(
      3,
      div(class = "panel-head", "Tiles"),
      selectInput("filter", NULL,
                  choices = c("All tiles" = "all",
                              "Not yet labeled" = "todo",
                              "Labeled" = "done"),
                  selected = "all", width = "100%"),
      div(class = "counter", textOutput("progress", inline = TRUE)),
      br(), br(),
      DTOutput("imglist"),
      br(),
      fluidRow(
        column(6, actionButton("prev", "< Back (b)", width = "100%")),
        column(6, actionButton("nxt", "Next (n) >", width = "100%"))
      )
    ),

    # ---- centre: canvas -----------------------------------------------------
    column(
      6,
      div(class = "tile-name", textOutput("tile_name")),
      tags$canvas(id = "editor", tabindex = "0"),
      div(
        class = "savebar",
        uiOutput("dirty_badge", inline = TRUE),
        actionButton("save", "Save annotations (s)", class = "btn-success"),
        actionButton("save_next", "Save & next", class = "btn-primary"),
        span(class = "save-detail", textOutput("save_detail", inline = TRUE))
      )
    ),

    # ---- right: classes + boxes --------------------------------------------
    column(
      3,
      radioButtons("class", "Class for new boxes (keys 1-6)",
                   choiceNames = unname(mapply(
                     function(nm, col) {
                       tagList(span(class = "swatch",
                                    style = paste0("background:", col, ";")),
                               " ", nm)
                     },
                     CLASSES, CLASS_COLORS[CLASSES],
                     SIMPLIFY = FALSE, USE.NAMES = FALSE
                   )),
                   choiceValues = unname(CLASSES)),
      tags$hr(),
      div(class = "panel-head", "Boxes on this tile"),
      DTOutput("boxtable"),
      br(),
      actionButton("del", "Delete selected box", width = "100%"),
      br(), br(),
      actionButton("clear", "Clear all boxes", width = "100%"),
      tags$hr(),
      div(class = "hint",
          tags$b("Draw"), " - drag on empty image.", br(),
          tags$b("Select"), " - click a box.", br(),
          tags$b("Move"), " - drag a selected box.", br(),
          tags$b("Resize"), " - drag a corner handle.", br(),
          tags$b("Zoom"), " - mouse wheel. ", tags$b("Pan"), " - right-drag or shift-drag.", br(),
          tags$b("f"), " fits the tile to the window.", br(),
          tags$b("1-6"), " sets the class, and retypes the selected box.", br(),
          tags$b("Delete"), " removes the selected box.", br(), br(),
          "Saving with zero boxes marks the tile as a confirmed negative, which",
          " is useful training signal. Move on without saving to leave a tile",
          " undecided.")
    )
  )
)

# =============================================================================
# Server
# =============================================================================

server <- function(input, output, session) {

  rv <- reactiveValues(
    idx = 1L, boxes = empty_boxes(), sel = 0L,
    W = 1, H = 1, dirty = FALSE,
    saved_msg = "", status_tick = 0L, pending = NA_integer_
  )

  current_png <- reactive(all_images[rv$idx])

  boxes_to_client <- function(b) {
    if (nrow(b) == 0) return(list())
    jsonlite::toJSON(b, dataframe = "rows", digits = 6)
  }

  load_tile <- function(i) {
    nm <- all_images[i]
    d  <- png_dims(nm)
    W  <- unname(d["W"]); H <- unname(d["H"])
    b  <- read_boxes(nm, W, H)          # local, not read back off rv

    rv$idx   <- i
    rv$W     <- W
    rv$H     <- H
    rv$boxes <- b
    rv$sel   <- 0L
    rv$dirty <- FALSE

    session$sendCustomMessage("load_tile", list(
      name  = nm,
      boxes = boxes_to_client(b)
    ))
  }

  session$onFlushed(function() {
    isolate({
      session$sendCustomMessage("init_editor", list(
        classes = CLASSES,
        colors  = as.list(CLASS_COLORS)
      ))
      load_tile(1L)
    })
  }, once = TRUE)

  push_boxes <- function() {
    session$sendCustomMessage("set_boxes", list(
      boxes = boxes_to_client(rv$boxes),
      sel   = rv$sel
    ))
  }

  # Hand the class list/colours to the canvas, then load the first tile once
  # the client has had a flush to register its message handlers.
  session$onFlushed(function() {
    session$sendCustomMessage("init_editor", list(
      classes = CLASSES,
      colors  = as.list(CLASS_COLORS)
    ))
    load_tile(1L)
  }, once = TRUE)

  # ---- incoming edits from the canvas ---------------------------------------
  observeEvent(input$boxes_json, {
    rv$boxes <- boxes_from_json(input$boxes_json)
    rv$dirty <- TRUE
  })

  observeEvent(input$sel_idx, {
    rv$sel <- as.integer(input$sel_idx)
  })

  observeEvent(input$set_class, {
    updateRadioButtons(session, "class", selected = input$set_class)
  })

  # ---- saving ---------------------------------------------------------------
  # Writes, then reads the file back and compares the line count against what's
  # in memory. A write that silently produced the wrong thing (permissions, a
  # sync client holding a lock, a full disk) reports as an error rather than a
  # green tick -- which matters, given the whole reason this app exists is that
  # a previous set of labels quietly disappeared.
  save_current <- function() {
    nm <- current_png()
    p  <- label_path(nm)
    err <- NULL
    tryCatch(
      write_boxes(nm, rv$boxes, rv$W, rv$H),
      error   = function(e) err <<- conditionMessage(e),
      warning = function(w) err <<- conditionMessage(w)
    )

    if (!is.null(err)) {
      showNotification(paste0("Save FAILED: ", err), type = "error", duration = NULL)
      rv$saved_msg <- "write failed"
      return(FALSE)
    }

    on_disk <- n_boxes_on_disk(nm)
    expect  <- nrow(rv$boxes)
    if (is.na(on_disk) || on_disk != expect) {
      showNotification(
        sprintf("Save VERIFICATION FAILED: expected %d box(es), file holds %s.",
                expect, ifelse(is.na(on_disk), "no file", as.character(on_disk))),
        type = "error", duration = NULL
      )
      rv$saved_msg <- "verification failed"
      return(FALSE)
    }

    rv$dirty <- FALSE
    rv$saved_msg <- sprintf("%s | %d box(es) -> %s",
                            format(Sys.time(), "%H:%M:%S"), expect, basename(p))
    rv$status_tick <- rv$status_tick + 1L
    showNotification(
      HTML(sprintf("Saved <b>%d</b> box(es) to<br><code>%s</code>", expect, p)),
      type = "message", duration = 4
    )
    TRUE
  }

  observeEvent(input$save,     save_current())
  observeEvent(input$key_save, save_current())

  # ---- navigation, with an unsaved-changes guard ----------------------------
  navigate_to <- function(i) {
    if (i < 1 || i > length(all_images) || i == rv$idx) return(invisible(NULL))
    if (isTRUE(rv$dirty)) {
      rv$pending <- i
      showModal(modalDialog(
        title = "Unsaved changes",
        sprintf("This tile has %d unsaved box(es).", nrow(rv$boxes)),
        footer = tagList(
          modalButton("Stay here"),
          actionButton("discard", "Discard and move on"),
          actionButton("save_go", "Save and move on", class = "btn-success")
        ),
        easyClose = TRUE
      ))
      return(invisible(NULL))
    }
    load_tile(i)
  }

  observeEvent(input$save_go, {
    removeModal()
    if (save_current() && !is.na(rv$pending)) load_tile(rv$pending)
    rv$pending <- NA_integer_
  })
  observeEvent(input$discard, {
    removeModal()
    if (!is.na(rv$pending)) load_tile(rv$pending)
    rv$pending <- NA_integer_
  })

  go_to <- function(step) navigate_to(rv$idx + step)

  observeEvent(input$nxt,      go_to(1L))
  observeEvent(input$prev,     go_to(-1L))
  observeEvent(input$key_next, go_to(1L))
  observeEvent(input$key_prev, go_to(-1L))
  observeEvent(input$save_next, { if (save_current()) load_tile(min(rv$idx + 1L, length(all_images))) })

  # ---- server-side box edits ------------------------------------------------
  observeEvent(input$del, {
    if (rv$sel < 1 || rv$sel > nrow(rv$boxes)) {
      showNotification("Select a box first (click it on the image).",
                       type = "warning", duration = 3)
      return()
    }
    rv$boxes <- rv$boxes[-rv$sel, , drop = FALSE]
    rv$sel <- 0L
    rv$dirty <- TRUE
    push_boxes()
  })

  observeEvent(input$clear, {
    if (nrow(rv$boxes) == 0) return()
    rv$boxes <- empty_boxes()
    rv$sel <- 0L
    rv$dirty <- TRUE
    push_boxes()
  })

  # Selecting a row in the box table selects it on the canvas too.
  observeEvent(input$boxtable_rows_selected, {
    s <- input$boxtable_rows_selected
    new_sel <- if (length(s) == 0) 0L else as.integer(s)
    if (identical(new_sel, rv$sel)) return()
    rv$sel <- new_sel
    push_boxes()
  }, ignoreNULL = FALSE)

  # ---- image browser --------------------------------------------------------
  list_df <- reactive({
    rv$status_tick                      # recompute after each save
    labeled <- vapply(all_images, is_done, logical(1))
    df <- data.frame(
      idx    = seq_along(all_images),
      Status = ifelse(labeled, "labeled", "-"),
      Tile   = sub("_rgb\\.png$", "", all_images),
      stringsAsFactors = FALSE
    )
    switch(input$filter %||% "all",
           todo = df[df$Status == "-", , drop = FALSE],
           done = df[df$Status == "labeled", , drop = FALSE],
           df)
  })

  output$imglist <- renderDT({
    df <- isolate(list_df())
    datatable(
      df[, c("Status", "Tile")],
      selection = "single", rownames = FALSE,
      options = list(
        pageLength = 15, lengthChange = FALSE, scrollX = TRUE,
        dom = "ftip",
        columnDefs = list(list(width = "70px", targets = 0))
      )
    ) |>
      formatStyle("Status",
                  color = styleEqual(c("labeled", "-"), c("#1e7d34", "#999")),
                  fontWeight = "bold")
  }, server = TRUE)

  img_proxy <- dataTableProxy("imglist")

  observeEvent(list_df(), {
    replaceData(img_proxy, list_df()[, c("Status", "Tile")],
                resetPaging = FALSE, rownames = FALSE)
  }, ignoreInit = TRUE)

  observeEvent(input$imglist_rows_selected, {
    r <- input$imglist_rows_selected
    if (length(r) == 0) return()
    target <- list_df()$idx[r]
    if (length(target) != 1 || is.na(target)) return()
    navigate_to(as.integer(target))
  })

  # ---- readouts -------------------------------------------------------------
  output$progress <- renderText({
    rv$status_tick
    done <- sum(vapply(all_images, is_done, logical(1)))
    sprintf("%d / %d labeled", done, length(all_images))
  })

  output$tile_name <- renderText({
    sprintf("[%d of %d]  %s", rv$idx, length(all_images), current_png())
  })

  output$dirty_badge <- renderUI({
    if (isTRUE(rv$dirty)) {
      span(class = "badge-dirty", "Unsaved changes")
    } else if (nzchar(rv$saved_msg)) {
      span(class = "badge-clean", "Saved")
    } else {
      span(style = "color:#888;", "No changes")
    }
  })

  output$save_detail <- renderText(rv$saved_msg)

  output$boxtable <- renderDT({
    b <- rv$boxes
    df <- if (nrow(b) == 0) {
      data.frame(class = character(), w_px = integer(), h_px = integer())
    } else {
      data.frame(
        class = CLASSES[b$class_id + 1L],
        w_px  = round(b$xmax - b$xmin),
        h_px  = round(b$ymax - b$ymin)
      )
    }
    datatable(df, selection = "single", rownames = TRUE,
              options = list(dom = "t", paging = FALSE, ordering = FALSE))
  }, server = FALSE)
}

# =============================================================================
# merge_labels() -- run from the R console after everyone has finished
# =============================================================================

merge_labels <- function(output_root = OUTPUT_ROOT) {
  src_dirs <- list.dirs(output_root, recursive = FALSE)
  src_dirs <- src_dirs[grepl("labels_", basename(src_dirs))]
  if (length(src_dirs) == 0) {
    message("No labels_<user>/ shard folders found -- nothing to merge. ",
            "If N_LABELERS is 1, this is expected: the app already writes ",
            "straight to labels/, so there's nothing for this function to do.")
    return(invisible(character(0)))
  }
  dest <- file.path(output_root, "labels")
  dir.create(dest, showWarnings = FALSE)

  seen <- list()
  for (d in src_dirs) {
    for (f in list.files(d, pattern = "\\.txt$")) {
      if (!is.null(seen[[f]])) {
        warning(sprintf("%s labeled by both %s and %s -- kept %s",
                        f, seen[[f]], basename(d), seen[[f]]))
        next
      }
      file.copy(file.path(d, f), file.path(dest, f), overwrite = TRUE)
      seen[[f]] <- basename(d)
    }
  }
  message(sprintf("Merged %d label files into %s", length(seen), dest))
  invisible(names(seen))
}

shinyApp(ui, server)