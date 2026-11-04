"""A tile that crosses the edge of its NAIP quad is filled from the neighbouring quads (01b fetch_naip_mosaic),
on fake quads: no rasterio or network here.
    python correction/tests/test_naip_mosaic.py"""
import importlib.util
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
for name in ("torch", "ultralytics", "planetary_computer", "pystac_client", "rasterio", "rasterio.enums",
             "rasterio.windows"):
    sys.modules.setdefault(name, mock.MagicMock())
import config as C

spec = importlib.util.spec_from_file_location("od01b", HERE.parent / "scripts" / "01b_run_object_detection.py")
od = importlib.util.module_from_spec(spec)
spec.loader.exec_module(od)


@dataclass
class Window:
    col_off: float
    row_off: float
    width: float
    height: float


class Quad:
    """A 1 m raster covering x in [x0, x0+w), y in (y_top-h, y_top], every pixel = value."""
    def __init__(self, x0, y_top, w, h, value):
        self.x0, self.y_top, self.width, self.height, self.value = x0, y_top, w, h, value
        self.crs, self.transform, self.reads = "fake", self, 0

    def read(self, bands, window, out_shape, resampling):
        assert window.col_off >= 0 and window.row_off >= 0
        assert window.col_off + window.width <= self.width and window.row_off + window.height <= self.height
        self.reads += 1
        return np.full(out_shape, self.value, dtype=np.uint8)


def from_bounds(left, bottom, right, top, transform):
    return Window(left - transform.x0, transform.y_top - top, right - left, top - bottom)


def item(url, year):
    return SimpleNamespace(assets={"image": SimpleNamespace(href=url + "?sig=new")}, datetime=datetime(year, 6, 1))


quads = {"A": Quad(0, 1000, 1000, 1000, 10),      # the plant's own quad
         "B": Quad(1000, 1000, 1000, 1000, 20),   # east neighbour, same year
         "B_old": Quad(1000, 1000, 1000, 1000, 30)}  # east neighbour, an older year: must not be used
catalog = SimpleNamespace(search=lambda **k: SimpleNamespace(
    items=lambda: [item("B_old", 2018), item("B", 2022), item("A", 2022)]))
od.Window, od.from_bounds = Window, from_bounds
od._get_transformer = lambda crs: SimpleNamespace(transform=lambda x, y: (x, y))
od._get_dataset = lambda url: quads[url.split("?")[0]]
od.Resampling = SimpleNamespace(bilinear=None)

# 800..1300 straddles the A/B edge: the plain fetch raises, the mosaic fills 40% from A and 60% from B
try:
    od.fetch_naip_tile((800, 200, 1300, 700), "A?sig=old")
    raise AssertionError("expected TileOutsideItemCoverage")
except od.TileOutsideItemCoverage:
    pass
arr, _ = od.timed_fetch((800, 200, 1300, 700), "A?sig=old", catalog)
assert arr.shape == (4, C.IMAGE_PX, C.IMAGE_PX) and arr.dtype == np.uint8
split = round(0.4 * C.IMAGE_PX)
assert (arr[:, :, :split] == 10).all() and (arr[:, :, split:] == 20).all(), np.unique(arr)
assert quads["B_old"].reads == 0, "an older year was read although the same-year neighbour covered the tile"

# a tile inside A never searches the catalog
boom = SimpleNamespace(search=lambda **k: (_ for _ in ()).throw(AssertionError("searched for an in-quad tile")))
arr, _ = od.timed_fetch((100, 200, 600, 700), "A?sig=old", boom)
assert (arr == 10).all()

# a tile hanging off the edge of all coverage (only 40% inside B) still fails, with a reason
try:
    od.timed_fetch((1800, 200, 2300, 700), "B?sig=old",
                   SimpleNamespace(search=lambda **k: SimpleNamespace(items=lambda: [item("B", 2022)])))
    raise AssertionError("expected TileOutsideItemCoverage for a mostly uncovered tile")
except od.TileOutsideItemCoverage as e:
    assert "40%" in str(e), e

# no catalog (01b's old callers): the error propagates unchanged
try:
    od.timed_fetch((800, 200, 1300, 700), "A?sig=old", None)
    raise AssertionError("expected TileOutsideItemCoverage without a catalog")
except od.TileOutsideItemCoverage:
    pass
print("naip mosaic tests passed")
