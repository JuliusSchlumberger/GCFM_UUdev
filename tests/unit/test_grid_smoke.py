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
    fit_grid_frame,
    footprint,
    frame_footprint,
    load_grid_def,
)

UTM31N = "EPSG:32631"


def _rotated_domain():
    rect = box(500_000, 5_700_000, 510_000, 5_703_000)
    return gpd.GeoDataFrame(
        geometry=[affinity.rotate(rect, 30.0, origin="centroid")], crs=UTM31N
    )


@pytest.mark.parametrize("rotation", [0.0, 30.0, 137.5])
def test_cell_size_and_area_independent_of_rotation(rotation):
    t = (
        Affine.translation(500_000, 5_700_000)
        * Affine.rotation(rotation)
        * Affine.scale(50, 50)
    )
    assert cell_size_m(t) == pytest.approx(50.0)
    assert cell_area_m2(t) == pytest.approx(2500.0)


def test_footprint_area_equals_cell_count_times_cell_area():
    t = Affine.translation(0, 0) * Affine.rotation(25.0) * Affine.scale(10, 10)
    assert footprint(t, (20, 30)).area == pytest.approx(20 * 30 * cell_area_m2(t))


def test_fit_grid_frame_unrotated_stores_no_rectangle():
    assert fit_grid_frame(_rotated_domain(), rotated=False) == {"rotated": False}


def test_rotated_frame_covers_domain_and_grid_covers_frame():
    domain = _rotated_domain()
    frame = fit_grid_frame(domain, rotated=True)
    assert 0.0 <= frame["rotation"] < 360.0
    outline = frame_footprint(frame)
    assert outline.buffer(1.0).contains(domain.union_all())

    grid_def = build_grid_def(domain, frame, resolution_m=500.0)
    assert grid_def["mmax"] * 500.0 >= frame["length_x_m"]
    assert grid_def["nmax"] * 500.0 >= frame["length_y_m"]
    assert grid_def["epsg"] == 32631
    t = Affine(*grid_def["transform"])
    assert cell_size_m(t) == pytest.approx(500.0)
    assert (
        footprint(t, (grid_def["nmax"], grid_def["mmax"])).buffer(1.0).contains(outline)
    )


def test_grid_def_json_round_trip_restores_transform_and_shape(tmp_path):
    domain = _rotated_domain()
    grid_def = build_grid_def(
        domain, fit_grid_frame(domain, rotated=True), resolution_m=500.0
    )
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(grid_def))
    back = load_grid_def(path)
    assert back["shape"] == (grid_def["nmax"], grid_def["mmax"])
    assert isinstance(back["transform"], Affine)
    assert tuple(back["transform"])[:6] == tuple(grid_def["transform"])
