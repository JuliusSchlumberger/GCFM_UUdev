"""Smoke tests for the domain bounding-box helpers in src.domain."""

import json

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box
from src.domain import bounds_to_wgs84, load_domain

UTM31N = "EPSG:32631"
UTM_BOUNDS = {
    "xmin": 500_000.0,
    "ymin": 5_700_000.0,
    "xmax": 520_000.0,
    "ymax": 5_720_000.0,
}


def _write_meta(tmp_path, bounds=UTM_BOUNDS, crs=UTM31N):
    path = tmp_path / "domain_bbox.json"
    path.write_text(json.dumps({"bounds": bounds, "crs": crs}))
    return path


def test_bounds_to_wgs84_identity_for_wgs84_input():
    bounds = {"xmin": 3.0, "ymin": 51.0, "xmax": 4.0, "ymax": 52.0}
    assert bounds_to_wgs84(bounds, "EPSG:4326") == (3.0, 51.0, 4.0, 52.0)


def test_bounds_to_wgs84_utm_bounds_are_ordered_and_in_zone():
    lon_min, lat_min, lon_max, lat_max = bounds_to_wgs84(UTM_BOUNDS, UTM31N)
    assert lon_min < lon_max and lat_min < lat_max
    # Zone 31N spans 0-6 E; easting 500 km is the 3 E central meridian.
    assert 0.0 <= lon_min <= 3.0 + 1e-9 < lon_max <= 6.0
    assert 51.0 < lat_min < lat_max < 52.0


def test_load_domain_without_gpkg_returns_bbox_polygon(tmp_path):
    wgs84, crs, poly = load_domain(_write_meta(tmp_path))
    assert crs == UTM31N
    assert isinstance(poly, Polygon)
    assert poly.bounds == pytest.approx(wgs84)


def test_load_domain_with_gpkg_returns_its_polygon_in_wgs84(tmp_path):
    triangle = Polygon(
        [(502_000, 5_702_000), (518_000, 5_702_000), (510_000, 5_718_000)]
    )
    gpkg = tmp_path / "domain.gpkg"
    gpd.GeoDataFrame(geometry=[triangle], crs=UTM31N).to_file(gpkg)
    wgs84, _, poly = load_domain(_write_meta(tmp_path), gpkg)
    assert box(*wgs84).contains(poly)
    assert poly.area < box(*wgs84).area
