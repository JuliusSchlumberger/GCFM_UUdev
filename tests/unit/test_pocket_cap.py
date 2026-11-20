"""Tests for src.protection_weir.cap_pocket_cells."""

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import LineString
from src.protection_weir import cap_pocket_cells

N_PX = 10
DX = 100.0
CREST = 11.0
# 1 x 4 grid, north-up: river cell (0, 0) | pocket (0, 1) | full floodplain (0, 2) | high slope (0, 3)
TRANSFORM = Affine(DX, 0.0, 0.0, 0.0, -DX, DX)


def _elevation():
    z = np.full((N_PX, 4 * N_PX), 30.0, dtype=np.float32)
    z[:, 0:N_PX] = 5.0
    z[:, N_PX : 2 * N_PX] = 20.0
    z[:2, N_PX : N_PX + 3] = 8.0  # 6 pixels (600 m2) below the crest
    z[:, 2 * N_PX : 3 * N_PX] = 10.8
    return z


def _weirs():
    # Vertical edges between cells 0|1, 1|2 and 2|3, all at the same crest.
    lines = [LineString([(x, 0.0), (x, DX)]) for x in (DX, 2 * DX, 3 * DX)]
    return gpd.GeoDataFrame(
        {"elevation": [CREST] * 3}, geometry=lines, crs="EPSG:32631"
    )


def _cap(**kw):
    args = (
        dict(max_wet_area_m2=2000.0, overtopping_m=(0.02, 0.5), below_crest_m=0.0) | kw
    )
    return cap_pocket_cells(_weirs(), _elevation(), TRANSFORM, N_PX, **args)


def test_only_the_small_wet_area_cell_is_a_pocket():
    _, pocket = _cap()
    assert pocket.tolist() == [[False, True, False, False]]


def test_pocket_pixels_above_the_crest_are_lowered_to_it_and_low_pixels_kept():
    overlay, _ = _cap()
    z = _elevation()
    block, orig = overlay[:, N_PX : 2 * N_PX], z[:, N_PX : 2 * N_PX]
    assert np.all(block[orig > CREST] == CREST)
    assert np.all(np.isnan(block[orig <= CREST]))
    assert np.all(np.isnan(np.delete(overlay, np.s_[N_PX : 2 * N_PX], axis=1)))


def test_overlay_never_raises_the_elevation():
    overlay, _ = _cap(below_crest_m=0.5)
    z = _elevation()
    changed = np.isfinite(overlay)
    assert changed.any() and np.all(overlay[changed] <= z[changed])
    assert np.all(overlay[changed] == CREST - 0.5)


def test_cell_entirely_above_the_overtopping_range_is_left_alone():
    _, pocket = _cap(overtopping_m=(0.02, 0.1))
    assert not pocket[0, 3]
    _, pocket = _cap(max_wet_area_m2=500.0)
    assert not pocket.any()
