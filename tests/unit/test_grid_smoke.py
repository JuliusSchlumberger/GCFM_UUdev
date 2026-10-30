"""Smoke tests for the shared SFINCS grid definition in src.grid."""

import json

import geopandas as gpd
import pytest
from affine import Affine
from shapely import affinity
from shapely.geometry import box
from src.grid import (
    build_grid_def,
    cell_area_m2,
    cell_size_m,
    footprint,
    load_grid_def,
)

UTM31N = "EPSG:32631"


def _oblique_domain():
    rect = box(500_000, 5_700_000, 510_000, 5_703_000)
    return gpd.GeoDataFrame(
        geometry=[affinity.rotate(rect, 30.0, origin="centroid")], crs=UTM31N
    )


def test_cell_size_and_area_of_axis_aligned_transform():
    t = Affine.translation(500_000, 5_700_000) * Affine.scale(50, 50)
    assert cell_size_m(t) == pytest.approx(50.0)
    assert cell_area_m2(t) == pytest.approx(2500.0)


def test_footprint_area_equals_cell_count_times_cell_area():
    t = Affine.translation(0, 0) * Affine.scale(10, 10)
    assert footprint(t, (20, 30)).area == pytest.approx(20 * 30 * cell_area_m2(t))


def test_grid_is_axis_aligned_south_up_and_covers_domain_within_one_cell():
    domain = _oblique_domain()
    grid_def = build_grid_def(domain, resolution_m=500.0)
    assert grid_def["rotation"] == 0.0
    assert grid_def["epsg"] == 32631
    t = Affine(*grid_def["transform"])
    assert t.b == 0.0 and t.d == 0.0
    assert t.e > 0.0  # row 0 on the southern side
    assert cell_size_m(t) == pytest.approx(500.0)
    outline = footprint(t, (grid_def["nmax"], grid_def["mmax"]))
    # hydromt's align=True rounds the origin, so the edges may fall short by < 1 cell.
    assert outline.buffer(500.0).contains(domain.union_all())


def test_grid_def_json_round_trip_restores_transform_and_shape(tmp_path):
    grid_def = build_grid_def(_oblique_domain(), resolution_m=500.0)
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(grid_def))
    back = load_grid_def(path)
    assert back["shape"] == (grid_def["nmax"], grid_def["mmax"])
    assert isinstance(back["transform"], Affine)
    assert tuple(back["transform"])[:6] == tuple(grid_def["transform"])
