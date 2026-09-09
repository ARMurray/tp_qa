"""
naip_fetch.py
=============
Shared Planetary Computer NAIP fetch for detection/. Single home for the
STAC search + windowed read + retry + quad-boundary fallback logic that
02_extract_tiles.py and convert_tiles_to_500m.py both need.

WHY THIS FILE EXISTS
    USDA's ArcGIS NAIP service (detection/'s original imagery source) went
    dead 2026-08-28. convert_tiles_to_500m.py ported the replacement from
    the HPC pipeline's 01b_run_object_detection.py. 02_extract_tiles.py now
    needs the same thing. Rather than a third copy of logic carrying dated,
    hard-won bug fixes (see with_retry), both import from here.

    convert_tiles_to_500m.py should be updated to import from this module
    and drop its own copies -- not done automatically, since that script is
    already-run and working; change it when you next touch it.

DIFFERENCE FROM convert_tiles_to_500m.py's COPY
    The dataset/transformer caches here are threading.local(), not plain
    dicts. convert_ runs sequentially and explicitly noted it dropped
    thread-local caching as unnecessary. 02_extract_tiles.py runs a
    ThreadPoolExecutor, so plain dicts would be shared mutable state across
    workers -- and rasterio dataset handles are NOT safe to read from
    multiple threads concurrently. This matches 01b's original design,
    which was multi-threaded for exactly this reason.

CONTRACT
    fetch_naip_tile(bbox_wgs84, item_url, image_px) -> (4, px, px) uint8
    RGB + NIR, bilinear-resampled. Raises TileOutsideItemCoverage when the
    window falls outside the item's raster -- callers should use
    timed_fetch(), which handles that case by re-resolving.
"""
import random
import re
import threading
import time

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

_local = threading.local()


def open_catalog():
    """One catalog per process is fine -- pystac_client is stateless enough
    that the modifier is applied per-signed-asset, not held on the client."""
    return pystac_client.Client.open(
        STAC_URL, modifier=planetary_computer.sign_inplace)


def _get_dataset(url: str):
    cache = getattr(_local, "datasets", None)
    if cache is None:
        cache = _local.datasets = {}
    ds = cache.get(url)
    if ds is None:
        ds = rasterio.open(url)
        cache[url] = ds
    return ds


def _get_transformer(dst_crs) -> Transformer:
    cache = getattr(_local, "transformers", None)
    if cache is None:
        cache = _local.transformers = {}
    key = str(dst_crs)
    tr = cache.get(key)
    if tr is None:
        tr = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
        cache[key] = tr
    return tr


def close_thread_datasets():
    """Release this thread's open rasterio handles. Worth calling at the end
    of a long pooled run -- each handle holds a network connection."""
    for ds in getattr(_local, "datasets", {}).values():
        try:
            ds.close()
        except Exception:
            pass
    _local.datasets = {}


class TileOutsideItemCoverage(Exception):
    """The tile's computed window falls outside the resolved NAIP item's
    actual raster extent. Distinct from a network/rate-limit failure so
    callers can retry against a freshly-resolved item rather than backing
    off against an item that was never going to cover it."""
    pass


def with_retry(fn, *args, max_retries=5, base_delay=8, **kwargs):
    """Exponential backoff + jitter for transient Planetary Computer errors.

    Two dated fixes from 01b_run_object_detection.py, both worth keeping:
      - TileOutsideItemCoverage is checked by TYPE and never retried.
        Retrying against the same item fails identically every time.
      - The 429/503/504 check uses a word-boundary regex. A plain substring
        match on "429" false-positived on a pixel offset like "11429" in a
        real error message (confirmed 2026-08-21).
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if isinstance(e, TileOutsideItemCoverage):
                raise
            msg = str(e).lower()
            is_transient = (
                "rate limit" in msg or "timeout" in msg or "timed out" in msg
                or re.search(r"\b(429|503|504)\b", msg) is not None
            )
            if not is_transient or attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
            print(f"    Transient error ({e}) -- retrying in {delay:.0f}s "
                  f"(attempt {attempt + 1}/{max_retries})", flush=True)
            time.sleep(delay)
    raise last_exc


def find_naip_item(catalog, lon: float, lat: float):
    """Most recent NAIP item covering a point, or None."""
    def _search():
        search = catalog.search(
            collections=["naip"],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            limit=5,
        )
        return list(search.items())

    items = with_retry(_search)
    if not items:
        return None
    items.sort(key=lambda it: it.datetime, reverse=True)
    return items[0]


def fetch_naip_tile(bbox_wgs84: tuple, item_url: str, image_px: int) -> np.ndarray:
    """Windowed read -> (4, image_px, image_px) uint8 (R, G, B, NIR)."""
    ds = _get_dataset(item_url)
    transformer = _get_transformer(ds.crs)
    xmin, ymin, xmax, ymax = bbox_wgs84
    left, bottom = transformer.transform(xmin, ymin)
    right, top = transformer.transform(xmax, ymax)
    window = from_bounds(left, bottom, right, top, transform=ds.transform)

    # Validate the window against the raster's real bounds BEFORE reading.
    # A 500m tile is large enough to straddle two NAIP quads even though
    # only one item was resolved from its center point.
    col_off, row_off = window.col_off, window.row_off
    win_w, win_h = window.width, window.height
    if (col_off < 0 or row_off < 0
            or col_off + win_w > ds.width or row_off + win_h > ds.height):
        raise TileOutsideItemCoverage(
            f"tile window ({col_off:.0f},{row_off:.0f} size {win_w:.0f}x{win_h:.0f}) "
            f"falls outside item raster ({ds.width}x{ds.height}) -- likely a "
            f"different NAIP quad than the one resolved for this tile")

    arr = ds.read([1, 2, 3, 4], window=window,
                  out_shape=(4, image_px, image_px),
                  resampling=Resampling.bilinear)
    return arr.astype(np.uint8)


def timed_fetch(bbox_wgs84, item_url: str, image_px: int, catalog=None) -> np.ndarray:
    """fetch_naip_tile with retry, plus quad-boundary fallback.

    On TileOutsideItemCoverage, resolves a FRESH item for this tile's own
    centroid rather than assuming the whole parcel shares one quad. Raises
    if the fresh item is the same one (genuinely uncovered, not a wrong-quad
    problem) -- that is the ~8.5% failure mode seen on 2026-08-28.
    """
    try:
        return with_retry(fetch_naip_tile, bbox_wgs84, item_url, image_px)
    except TileOutsideItemCoverage:
        if catalog is None:
            raise
        xmin, ymin, xmax, ymax = bbox_wgs84
        tile_lon, tile_lat = (xmin + xmax) / 2, (ymin + ymax) / 2
        fresh_item = with_retry(find_naip_item, catalog, tile_lon, tile_lat)
        if fresh_item is None:
            raise
        fresh_url = fresh_item.assets["image"].href
        if fresh_url == item_url:
            raise
        return with_retry(fetch_naip_tile, bbox_wgs84, fresh_url, image_px)
