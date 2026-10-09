# Plant location viewer

Every CWNS treatment plant on a map, coloured by location status. Runs on the
work computer. Modelled on the sewershed_plus viewer (deck.gl, street/imagery
toggle, search, legend-as-filter, slide-in detail panel), but **without
tiles**: ~18k points go to the browser as one gzipped JSON (~1 MB), so there
is no tippecanoe step and no WSL.

```bash
cd viewer
python app.py            # then open http://localhost:8050
```

Needs `starlette`, `uvicorn`, `pandas`, `pyarrow`, `geopandas`, `duckdb`
(already in the review app's Python) and `h3` (`pip install h3`, for the
all-parcels layer).

## Where the data comes from

`build_data.py` runs at startup:

1. **`correction/diagnostics/output/cwns_corrected_locations.parquet`** if it
   exists. `13_build_corrected_output.py` writes this copy on the HPC; commit
   it there (`git add correction/diagnostics/output/`), `git pull` here.
2. Otherwise a **preview** from the master (`Updates.gpkg`): the human
   statuses only, every other plant `pending`.

Names and places come from the master; review history from
`review_app/data/outgoing/review_log_round*.parquet`. After pulling new data,
restart or open `http://localhost:8050/api/reload`.
`VIEWER_OUTPUT=<path>` points it at another output file.

## On the map

- Colour = status (legend; click a row to hide/show it, counts per status).
  "model decisions only" shows moved / flagged / kept.
- Moved and verified-corrected plants are drawn at the NEW location with a
  line back to the reported one; from zoom 9 a grey ring marks the reported
  point.
- Click a plant for its panel (status, coordinates, distance moved, Stage 1 /
  re-rank scores, review history, Google Maps links). `#id=<CWNS_ID>` in the
  URL opens a plant directly.

## Exploring candidates, parcels and detections (2026-10-09)

Why a plant was or was not moved, and what else is nearby.

- **Candidates.** Selecting a plant lists every candidate the models scored
  for it, best first -- including those below the move cutoff (dashed line):
  re-rank and Stage 2a scores and ranks, distance, detections, owner,
  acreage, OSM / keyword / utility-owner flags. Their outlines are drawn on
  the map with rank labels (#1 orange, #2-5 yellow, the rest grey; the
  reported parcel cyan). Click a row to fly to that parcel.
- **All parcels (zoom 15+).** Every parcel in the window, live from the local
  Regrid mirror. Parcels that are a scored candidate of any plant are purple.
- **Parcel card.** Click any parcel: its Regrid attributes, its scores if it
  is a candidate of the selected plant, and every other plant it is a
  candidate for (click to jump to that plant).
- **All detections (zoom 14+).** Every object the deployed detector found:
  around reported locations, corrected locations and every candidate parcel.

Plants Stage 1 passed (and OSM-confirmed ones) have no candidates by design;
the panel says so -- use All parcels to inspect the area.

### Data

| What | From | Size |
|---|---|---|
| candidates | `correction/diagnostics/output/viewer_candidates.parquet` | a few MB |
| detected objects | `correction/diagnostics/output/viewer_objects.parquet` | a few MB |
| parcels | the local Regrid mirror, live (never in git) | -- |

The two parquet files are written on the HPC by `export_viewer_data.py`,
which `13_build_corrected_output.slurm` runs after 13. Commit
`correction/diagnostics/output/` there and `git pull` here; restart the viewer.

Parcels are read with DuckDB from `review_app/config.py`'s `REGRID_ROOT`
(override with `VIEWER_REGRID_ROOT`): by `ll_uuid` for a plant's candidates,
and by H3 cell (`h3_index_9`) for the window, cached in memory. The first
window in a state scans that state's files and can take a few seconds;
panning back is instant. Windows wider than 4 km are not fetched.

If DuckDB's spatial extension cannot be installed (offline), geometry is
decoded with shapely instead -- slower, same result.
