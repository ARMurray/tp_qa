# Current Status & Next Steps

## Status Summary
**Phase:** Full pipeline complete and working end-to-end in Python on the local
EPA Windows laptop. **Next phase:** port the compute-heavy stages (~scripts 02–05)
to the HPC cluster, then expand annotations and rerun the full loop.

---

## Completed

### ✅ Repo consolidation
- Single git repo at `C:\Users\AMURRA02\wastewater_infrastructure_detection`
  (NON-OneDrive path, so OneDrive doesn't sync tiles/venv)
- Venvs kept OUTSIDE the repo (they hardcode absolute paths); represented by
  `requirements.txt` + `requirements-labelstudio.txt`
- Round-1 tiles (3,561) and 182 annotations preserved through the reorg —
  filenames are the index, so moving files kept everything intact

### ✅ Full R → Python port + new stages
All six scripts written, wired to a central `config.py`, and run successfully:
- `01_sample_sites.py` — correct-only plants (190) + TRI (50); now actually
  writes the sample that `02` consumes (the R version left this disconnected)
- `02_extract_tiles.py` — threaded NAIP extraction; same endpoint/CRS/filename
  contract as R; skip-existing + additive metadata
- `03_prepare_dataset.py` — LS export → YOLO dataset, plant-level split
- `04_train_model.py` — YOLOv8s training (config-wired, `RUN_NAME="wwtp_v2"`)
- `05_run_inference.py` — inference → deduped detections → corrected coordinates
- `06_build_map.py` — MapLibre QA web map (facilities + detections, basemap toggle)

### ✅ End-to-end run confirmed
Ran the pipeline through training, inference, and the map. `05` produced its two
outputs (`detections` and `corrected_coordinates`); `06` renders points on
streets/imagery basemaps.

---

## Environment / access issues resolved this phase
(Full detail in PROJECT_OVERVIEW.md → Known Issues. Summary here so the next
session has the landmines mapped.)

- **Training checkpoint write failure** (`PermissionError` on `last.pt`): ACLs
  clean, Controlled Folder Access off, no named EDR, raw + 80 MB looped writes
  succeeded — but `torch.save` (zip-of-pickles) failed everywhere. Suspected
  endpoint-security behavioral rule. Worked around by (1) moving `RUNS_DIR` to
  `C:\Users\AMURRA02\ml_artifacts\runs` (outside repo) and (2) forcing legacy
  non-zip torch serialization. **Clean fix = EPA IT exclusion (ticket). Should
  not recur on Linux HPC.**
- **pandas `groupby().apply()`** dropped grouping columns in `01` → rewrote to
  explicit iteration + `pd.concat`.
- **Population path**: files are under `Github/Sewersheds/Data/`, not
  `Location_Correction` — `config.py` now has separate `SEWERSHEDS` root.
- **torch nightly** (cu128, for Blackwell sm_120) must install BEFORE
  `requirements.txt`.
- **Web map (06):** `unpkg.com` blocked → MapLibre vendored + inlined; `file://`
  Web Worker block → must serve via `python -m http.server`; `setStyle()` toggle
  destroyed data layers → single-style visibility toggle with Esri raster
  basemaps.
- **Can't write `C:\` root** (non-admin) — use paths under the user profile.

---

## Next Steps

### ⏳ Phase A — HPC port (~scripts 02–05)  ← immediate next work
Detailed plan in IMPLEMENTATION_PLAN.md → "HPC Porting Plan". Key items:
1. **Env + paths on HPC** — cluster module/conda env from `requirements.txt`;
   cluster-appropriate CUDA/torch (drop the laptop's Blackwell nightly); add an
   HPC path profile in `config.py`.
2. **Confirm compute-node network policy** — `02` (NAIP), `01` (pygris), `06`
   (Esri) all need internet. If nodes are offline, run extraction on a
   login/transfer node or pre-stage imagery. **This decision drives the `02`
   redesign.**
3. **Parallelize `02` (and `05`) as job arrays** — shard by county/facility.
   Watch the shared-CSV append race: write per-shard metadata and merge.
4. **Stage inputs** — Regrid parquet, `Updates.gdb`, population tables onto
   cluster storage.
5. **Drop the Windows workarounds** on Linux (checkpoint permission / legacy
   torch save / `RUNS_DIR`-outside-repo).

### ⏳ Phase B — More annotation + full loop again
Once the code runs on HPC: expand the labeled set (round-2 tiles, especially
`aeration_basin` and geographically diverse `oxidation_pond`; TRI empties as hard
negatives), retrain `wwtp_v2`, and rerun inference → corrected coordinates → map.
Then feed corrected coordinates into the random forest pipeline.

### Later / deferred
- NDWI as a 4th input channel (round 3+) — tiles already extracted
- `negative_industrial` distractor class for TRI structures (round 3)
- Confidence / NMS threshold tuning in `05` (defaults: conf 0.25, NMS 12 m)

---

## Quick-Reference: Pipeline Flow

```
01_sample_sites.py   -> data/samples/training_sample_round2.gpkg  (plants, tri)
02_extract_tiles.py  -> data/tiles/rgb/png, data/tiles/ndwi, data/tile_metadata.csv
[annotate in Label Studio, export YOLO] -> annotation/ls_export/
03_prepare_dataset.py-> dataset/ + dataset.yaml
04_train_model.py    -> ml_artifacts/runs/wwtp_v2/weights/best.pt
05_run_inference.py  -> data/inference/detections.*, corrected_coordinates.*
06_build_map.py      -> data/inference/facility_map.html  (serve via localhost)
```

## Environment Quick-Reference
| Item | Value |
|---|---|
| Repo root | `C:\Users\AMURRA02\wastewater_infrastructure_detection` (non-OneDrive) |
| Main venv | `C:\Users\AMURRA02\yolo_env` (activate: `& "…\Activate.ps1"`) |
| LS venv | `C:\Users\AMURRA02\ls_env` |
| Training outputs | `C:\Users\AMURRA02\ml_artifacts\runs` (outside repo) |
| GPU / torch | RTX Pro 2000 Blackwell (sm_120) / cu128 nightly |
| CWNS gdb | `…/Github/Location_Correction/data/Updates.gdb` |
| Population | `…/Github/Sewersheds/Data/POPULATION_*` |
| Parcels | `…/Data/Regrid/Parquet_Storage/state=XX/{geoid}.parquet` |
