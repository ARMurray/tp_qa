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

Needs `starlette`, `uvicorn`, `pandas`, `pyarrow`, `geopandas` (already in
the review app's Python).

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
