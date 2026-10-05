"""
app.py -- CWNS treatment-plant location viewer
==============================================
Every CWNS treatment plant on a map, coloured by location status (verified,
moved, kept, flagged, not assessed). Modelled on sewershed_plus's viewer
(deck.gl front end, one Python process), but with no tiles: ~18k points go
to the browser as one gzipped JSON, so nothing needs tippecanoe or WSL.

Run (work computer, from this folder):
    python app.py                      # http://localhost:8050
    python app.py --port 8051

Data is assembled at startup by build_data.py: the 13 output if
correction/diagnostics/output/cwns_corrected_locations.parquet exists,
otherwise a preview from the master. Restart after pulling new output, or
hit /api/reload.

Requires: starlette, uvicorn, pandas, pyarrow, geopandas (all already in
the review app's environment).
"""
import argparse
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

import build_data

WWW = Path(__file__).resolve().parent / "www"

_lock = threading.Lock()
_payload: bytes = b""
_sites: bytes = b""
_dets: bytes = b""
_meta: dict = {}


def load() -> None:
    global _payload, _sites, _dets, _meta
    d = build_data.build()
    sites = build_data.build_sites()
    dets = build_data.build_detections()
    with _lock:
        _payload = json.dumps(d, separators=(",", ":")).encode("utf-8")
        _sites = json.dumps(sites, separators=(",", ":")).encode("utf-8")
        _dets = json.dumps(dets, separators=(",", ":")).encode("utf-8")
        _meta = {k: d[k] for k in ("source", "cutoff", "counts")}
    print(f"[viewer] {d['source']}: {len(d['plants']):,} plants, "
          f"{len(sites['features']):,} sites, {len(dets):,} detections")


class NoCacheStatic(StaticFiles):
    """The viewer's JS/CSS change with every git pull; never serve a stale copy."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


async def index(request):
    return FileResponse(WWW / "index.html", headers={"Cache-Control": "no-cache"})


async def plants(request):
    with _lock:
        body = _payload
    return Response(body, media_type="application/json",
                    headers={"Cache-Control": "no-store"})


async def sites(request):
    with _lock:
        body = _sites
    return Response(body, media_type="application/geo+json",
                    headers={"Cache-Control": "no-store"})


async def detections(request):
    with _lock:
        body = _dets
    return Response(body, media_type="application/json",
                    headers={"Cache-Control": "no-store"})


async def meta(request):
    with _lock:
        return JSONResponse(_meta)


async def reload(request):
    load()
    return JSONResponse({"status": "reloaded", **_meta})


@asynccontextmanager
async def lifespan(app):
    load()
    yield


app = Starlette(
    routes=[
        Route("/", index),
        Route("/data/plants.json", plants),
        Route("/data/sites.geojson", sites),
        Route("/data/detections.json", detections),
        Route("/api/meta", meta),
        Route("/api/reload", reload, methods=["GET", "POST"]),
        Mount("/static", NoCacheStatic(directory=WWW), name="static"),
    ],
    middleware=[Middleware(GZipMiddleware, minimum_size=1000)],
    lifespan=lifespan,
)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
