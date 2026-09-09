# Detection Pipeline Reference

Companion to `TPQA_MASTER_REFERENCE.md` (which covers `correction/`, the HPC
side). This covers `detection/` — the local, Windows, GPU-bound pipeline that
trains the object-detection model `correction/`'s `01b`/`01c` consume as
`best.pt`. Written 2026-08-28 to close out a long session that found and
fixed several real bugs, swapped imagery providers, and built the tooling to
harvest historical labels into the current tile size. Read this before
picking detection work back up — most of what's below was hard-won.

---

## 1. The Two-Pipeline Relationship (recap)

```
  detection/pipeline/  (local, Windows, GPU-bound)
        │
        │  produces
        ▼
     best.pt  ──────────────────────►  correction/  (HPC, Linux)
                  uploaded manually        consumed by 01b and 01c
```

Full detail in `TPQA_MASTER_REFERENCE.md` §1. This doc only covers the
`detection/` side.

---

## 2. Real Folder Structure (the thing that caused the most pain this session)

```
tp_qa/detection/                     <-- REPO_ROOT, confirmed 2026-08-28
├── pipeline/                        scripts only -- NOT the data root
│   ├── config.py                    REPO_ROOT = Path(__file__).resolve().parent.parent
│   ├── 01_sample_sites.py
│   ├── 02_extract_tiles.py
│   ├── 03_prepare_dataset.py
│   ├── 04_train_model.py
│   ├── 05_run_inference.py          DEPRECATED
│   ├── 06_build_map.py              DEPRECATED
│   ├── diagnose_label_matching.py   NEW 2026-08-28 -- see §5
│   ├── reconstruct_tile_metadata.py NEW 2026-08-28 -- see §6
│   ├── convert_tiles_to_500m.py     NEW 2026-08-28 -- see §7
│   └── fold_in_500m.py              NEW 2026-08-28 -- see §7
├── data/
│   ├── tile_metadata.csv            LIVE, current round's tiles (48 plants
│   │                                 as of 2026-08-28, growing)
│   ├── tile_metadata_500.csv        convert_tiles_to_500m.py's raw output --
│   │                                 folded into tile_metadata.csv by
│   │                                 fold_in_500m.py, not a permanent file
│   └── tiles/
│       ├── rgb/png/                 LIVE 200m RGB tiles (333px)
│       ├── ndwi/                    LIVE 200m NDWI tiles
│       ├── rgb_500/                 convert_tiles_to_500m.py's raw output,
│       │                             flat (NOT nested under a png/ subfolder,
│       │                             unlike rgb/) -- folded into rgb/png/
│       └── ndwi_500/                same, folded into ndwi/
├── annotation/
│   └── ls_export/
│       ├── classes.txt              alphabetical order -- see §4
│       ├── labels/                  LIVE labels, clean filenames going
│       │                             forward (label_app.R), some legacy
│       │                             mangled Label Studio filenames still
│       │                             present -- see §5
│       └── labels_500/              convert_tiles_to_500m.py's raw output,
│                                     folded into labels/
├── dataset/                         03_prepare_dataset.py's YOLO output
├── OWM_Imagery_Labeler/             label_app.R -- see §4
│   └── .here                        REQUIRED marker file, see §4
└── models/
    └── runs/                        04_train_model.py's output (outside
                                       the git-tracked path on purpose --
                                       see config.py's own comment on an EDR/
                                       write-permission issue that forced this)
```

**`REPO_ROOT` bug, fixed 2026-08-28:** `config.py` lives in `pipeline/`, but
every data/annotation/dataset/model folder is a sibling of `pipeline/`, one
level up. The config previously resolved `REPO_ROOT` to `pipeline/` itself
(`Path(__file__).resolve().parent`), silently pointing every path at empty
folders that were never populated. Fixed to `.parent.parent`. If you ever see
`02`/`03`/`04` or any of the new scripts below report suspiciously empty
input, check this first — it's the single most consequential bug found this
session and would have masked itself as "no data" rather than an error.

**A separate, standalone repo (`C:\Users\AMURRA02\wastewater_infrastructure_detection\`)
is dead** — an earlier, pre-consolidation location. Nothing should read from
it going forward; it was a red herring during this session's debugging, not
a real data source.

---

## 3. Imagery Source: Planetary Computer, Not USDA (changed 2026-08-28)

`02_extract_tiles.py` (200m tiles) and the original version of
`convert_tiles_to_500m.py` both used USDA's ArcGIS Image Server
(`gis.apfo.usda.gov`) — a simple `exportImage` REST call. **That service went
dead 2026-08-28** (SSL/connection failures, not a code bug).

`convert_tiles_to_500m.py` now fetches from Planetary Computer instead,
porting `correction/`'s `01b_run_object_detection.py` fetch machinery
directly rather than reimplementing it:

- STAC item search (`collections=["naip"]`, point-intersects), items sorted
  by date, most recent first.
- SAS-token signing via `pystac_client.Client.open(STAC_URL,
  modifier=planetary_computer.sign_inplace)` — every asset URL from a
  catalog opened this way is pre-signed, no separate signing call needed.
- Retry-with-backoff (`with_retry`) for transient/rate-limit errors, with
  two dated bug fixes worth knowing about if this is ever touched again:
  `TileOutsideItemCoverage` is checked by exception **type**, never retried
  (retrying against the same item fails identically every time); the
  429/503/504 check uses a **word-boundary regex**, not substring matching,
  because a coordinate like `11429` false-positive-matched a plain `"429"`
  substring check once (confirmed 2026-08-21, cost ~134s of wasted backoff).
- Cross-quad fallback: if a tile's window falls outside the originally
  resolved item's raster, re-resolve a fresh item from the tile's own
  center point before giving up. **This matters more for 500m tiles than it
  did for `01b`'s original tiles** — larger footprint, more likely to
  straddle a NAIP quad boundary. Confirmed 2026-08-28: 6/71 groups (~8.5%)
  hit this and it wasn't recoverable even with the fallback — same accepted
  tail-issue rate already documented for `correction/`'s `01b`/`01c` on HPC.
  Real cross-quad mosaicking would close it fully; not built, deliberately,
  as low-value for the current scope.

`STAC_URL` (`https://planetarycomputer.microsoft.com/api/stac/v1`) is
hardcoded in `convert_tiles_to_500m.py` rather than pulled from
`detection/pipeline/config.py`, since that config was built around the USDA
service and has no equivalent constant. Worth adding there if more of
`detection/` moves off USDA — `02_extract_tiles.py` itself still uses the
dead USDA endpoint and will need the same fix whenever it's next run.

**`pip install pystac-client planetary-computer`** required locally — not
part of `detection/`'s original dependency set.

---

## 4. `label_app.R` (replaced Label Studio, which was deleted from this machine)

Lives at `detection/OWM_Imagery_Labeler/label_app.R`. Two real bugs fixed
2026-08-28, before it had ever been used for real work:

1. **Solo use (`N_LABELERS=1`) now writes straight to `labels/`.** It
   previously always wrote to a per-user shard (`labels_<username>/`)
   regardless of labeler count, requiring a manual `merge_labels()` run
   even for one person — directly working against "save straight to the
   source folder." Multi-labeler mode (`N_LABELERS > 1`) still shards
   correctly; that part of the design was right.
2. **The "Add box (Enter)" keyboard shortcut was a non-functional stub** —
   `session$sendInputMessage("add", NULL)` doesn't trigger a server-side
   Shiny observer, and `shinyjs` was never loaded. Fixed by sharing the
   add-box logic between the button click and the keypress handler directly.

**Requires a `.here` marker file** in `OWM_Imagery_Labeler/` (empty file,
literally named `.here`) — `here::here()`'s root-detection heuristics could
otherwise latch onto a `.git` higher up `tp_qa/` instead of this folder.
Create this before first run if it isn't already there.

Classes are written in **alphabetical order**
(`aeration_basin, chlorine_contact, clarifier, digester, drying_bed,
oxidation_pond`) to match `config.py`'s `CLASSES` list — class IDs are
positional. Don't reorder to match any other convention.

**No in-place class change or box resize** — only delete-and-redraw. Known,
accepted limitation, not fixed.

---

## 5. Legacy Label Filename Formats

Two different filename conventions exist in `annotation/ls_export/labels/`,
both handled correctly by `03_prepare_dataset.py`'s `parse_tile_stem()`:

- **Clean** (what `label_app.R` writes going forward): `{tile_id}_rgb.txt`
- **Label Studio export-mangled** (legacy, ~264 files as of 2026-08-28):
  `{hash}__Users%5C...%5C{tile_stem}_rgb.txt` or
  `{hash}-{tile_stem}_rgb.txt`

`diagnose_label_matching.py` (new) checks which convention a given set of
label files actually uses and cross-matches against `tile_metadata.csv` —
run this first if labels ever seem to not be matching tiles, rather than
guessing. It reuses `parse_tile_stem()` from `03_prepare_dataset.py` via
`importlib` rather than reimplementing it, matching the reuse pattern
`correction/`'s `01c`/`05` already use for the same reason (avoid drift
between two copies of the same parsing logic).

**Important finding, 2026-08-28:** `tile_metadata.csv` only ever reflects
the *current* extraction round. The 264 legacy-format labels turned out to
belong to an earlier round (round 1) that round 2's `tile_metadata.csv` has
zero record of — only 1 CWNS_ID overlapped between the two. `SAMPLE_GPKG`
being named `training_sample_round2.gpkg` was the tell. If this happens
again (a 0-match result from `diagnose_label_matching.py` even after
confirming filename parsing works), suspect a round mismatch before
anything else — check for an older metadata file, or use
`reconstruct_tile_metadata.py` (§6) if none can be found.

---

## 6. Recovering Metadata for Labels That Predate It

`reconstruct_tile_metadata.py` (new). For labels whose original
`tile_metadata.csv` row can't be found, regenerates the geographic bounds
from first principles — fully deterministic, since tile geometry only
depends on parcel geometry + `02_extract_tiles.py`'s own grid algorithm,
both reusable directly:

1. Parses `(CWNS_ID, ll_uuid, row, col)` from the label filename via
   `03_prepare_dataset.py`'s `parse_tile_stem()`.
2. Derives state from the CWNS_ID's 2-digit FIPS prefix (standard table,
   confirmed against real data: CWNS `29001011003` has `st=MO`, FIPS 29 is
   Missouri).
3. Looks up that parcel's real geometry from the Regrid store — **must**
   query via `ST_AsText(ST_GeomFromWKB(...))` and parse WKT text, not pull
   the raw BLOB column into Python and decode with `shapely.from_wkb()`
   client-side. The latter failed on 100% of real parcels across multiple
   states with `TypeError: Expected bytes or string, got int` — too uniform
   to be a data anomaly, and it was the one place this script diverged from
   the pattern every other script in this project already uses for this
   exact lookup (`parcels.py`, `01a_extract_parcels.py`,
   `02_feature_engineering.py` all decode WKB inside DuckDB, never in
   Python). Switching to the proven pattern fixed it completely (264/264
   on the confirming re-run).
4. Regenerates `02_extract_tiles.py`'s tile grid for that parcel (reused via
   `importlib`, not reimplemented) and picks out the specific `(row, col)`
   tile the filename refers to.
5. Copies the label to a clean filename alongside the reconstructed metadata
   row, so downstream tools need zero awareness of the legacy naming.

**Caveat, stated plainly:** if Regrid's parcel data changed since the
original extraction (re-survey, boundary correction, split/merge), the
regenerated grid could differ from what was actually shown at labeling time.
Flagged automatically if a lookup fails outright; not detectable if the
parcel still exists but its boundary shifted slightly.

Requires `duckdb` pinned to a real release (`pip install duckdb==1.5.5`
confirmed working) — a fresh `pip install duckdb` grabbed a hash-versioned
pre-release build with no published `spatial` extension, a 404 on load.

---

## 7. Converting 200m Tiles to 500m

`convert_tiles_to_500m.py` (new). Groups existing labeled 200m tiles by
`(CWNS_ID, ll_uuid_primary)` and collapses each group into one 500m tile —
directly solves "too many overlapping images per site." Matches
`correction/`'s own tile size (`TILE_SIZE_M=500`, `IMAGE_PX=833`) at the
same 0.6m/px resolution.

**Geometry:** every box is remapped old-pixel-space → real-world WGS84 →
new-pixel-space using each old tile's stored `bbox_xmin/ymin/xmax/ymax`,
matching the same linear-affine convention `save_ndwi_tif()` already uses
(`rasterio.transform.from_bounds`). Boxes labeled redundantly across
multiple old overlapping tiles for the same real object are de-duplicated
via per-class IoU merge (threshold 0.5) after remapping.

**Confirmed 2026-08-28, first real run:** 264 old tiles → 65 new tiles, 519
boxes in → 395 actually written (438 after dedup, minus the boxes belonging
to 6 groups whose NAIP fetch ultimately failed — see §3's cross-quad note).
12 boxes genuinely fell outside their new tile's extent after remapping
(real, not a bug — worth spot-checking any group flagged with "wide spread"
in the output, since that's exactly when this is expected).

**Output lands in NEW sibling folders** (`rgb_500/`, `ndwi_500/`,
`labels_500/`, `tile_metadata_500.csv`) — reviewable before merging, not
auto-merged into the live dataset.

`fold_in_500m.py` (new) does the actual merge: copies into the same live
folders `label_app.R` and `03_prepare_dataset.py` already use, never
silently overwriting an existing destination file, deduplicating
`tile_metadata.csv` on `tile_id` (keep newest). After this runs, 500m tiles
are indistinguishable from any other tile to every existing tool.

**`--source-root` override** on `convert_tiles_to_500m.py`: defaults to
`config.py`'s own `DATA_DIR`/`ANNOTATION_DIR` (correct now that `REPO_ROOT`
is fixed), but can point at a different tree — e.g.
`reconstruct_tile_metadata.py`'s output folder, for converting legacy
labels that predate the live metadata file.

**Mixed tile sizes in one training set — informational, not fixed:** 200m
(333px) and 500m (833px) tiles are both captured at the same native
0.6m/px resolution, so a real object occupies the same pixel footprint in
either natively. If YOLO training resizes everything to one common `imgsz`,
that resize is a different-direction rescale depending on source tile size
(333→imgsz upscales, 833→imgsz downscales) — a plausible source of scale
variance the model has to learn around. Not necessarily a problem (YOLO is
reasonably scale-robust), but worth checking first if training results look
odd. `tile_metadata.csv`'s `source` column (`upscaled_500m` vs whatever the
original extraction used) already supports filtering by tile size if this
ever needs isolating.

---

## 8. Feeding Reviewer Findings Back into Annotation

`analysis/find_false_top_picks.py` (in the review app, `tp_qa/review_app/`,
not `detection/` — but the output feeds `detection/`'s next annotation
round). Pulls every reviewed plant where the model's rank-1 candidate was
**not** the correct site — the reviewer picked a lower-ranked candidate
instead, or confirmed the reported location was right despite candidates
existing. Outputs a CSV with each false-positive site's location, owner,
and LBCS description.

**Direct motivation:** repeated review findings of baseball fields and
cul-de-sacs scoring above real treatment facilities — a visual-confusion
pattern the current annotation set has no explicit hard-negative category
for (existing hard negatives are drawn from TRI facilities only, in
`01_sample_sites.py`). The CSV's `rank1_lat`/`rank1_lon` columns are
candidate sites for the next hard-negative annotation batch. Not wired into
`01_sample_sites.py` automatically — a manual step for now.

---

## 9. Recommended Order If Picking This Back Up

1. Confirm `fold_in_500m.py` ran cleanly and `label_app.R` can open/edit a
   folded-in 500m tile.
2. Run `find_false_top_picks.py` again against whatever review rounds have
   accumulated since 2026-08-28; pull `rank1_lat`/`rank1_lon` sites into the
   next `01_sample_sites.py` batch as explicit hard negatives.
3. Retrain (`03_prepare_dataset.py` → `04_train_model.py`) once enough new
   labels + hard negatives exist to matter — watch for the mixed-tile-size
   consideration in §7 if results look off.
4. Re-deploy `best.pt` to HPC (`correction/`'s model directory) — still a
   manual step, per `TPQA_MASTER_REFERENCE.md` §1.
5. If `02_extract_tiles.py` is ever run again, it still points at the dead
   USDA service — port the same Planetary Computer fetch from
   `convert_tiles_to_500m.py` before relying on it.
