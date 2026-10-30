"""Smoke tests for the CRS and point-snapping helpers in src.geometry."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box
from src.geometry import buffered_bbox, pick_utm_crs, snap_points_into_region

REGION = box(0.0, 0.0, 1.0, 1.0)


def _points(*xy):
    return gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in xy], crs="EPSG:4326")


@pytest.mark.parametrize(
    "lon, lat, expected",
    [
        (3.5, 51.5, "EPSG:32631"),
        (-58.5, -34.5, "EPSG:32721"),
        (106.7, 10.5, "EPSG:32648"),
    ],
)
def test_pick_utm_crs_matches_zone_and_hemisphere(lon, lat, expected):
    assert pick_utm_crs(_points((lon, lat))) == expected


def test_buffered_bbox_grows_by_buffer_on_every_side():
    gdf = gpd.GeoDataFrame(geometry=[box(3.0, 51.0, 3.2, 51.2)], crs="EPSG:4326")
    bbox_gdf, bounds = buffered_bbox(gdf, 1000.0, "EPSG:32631")
    pminx, pminy, pmaxx, pmaxy = gdf.to_crs("EPSG:32631").total_bounds
    assert bounds == pytest.approx(
        (pminx - 1000, pminy - 1000, pmaxx + 1000, pmaxy + 1000)
    )
    assert bbox_gdf.crs == "EPSG:32631"
    assert list(bbox_gdf.total_bounds) == pytest.approx(list(bounds))


def test_buffered_bbox_sets_source_crs_when_missing():
    gdf = gpd.GeoDataFrame(geometry=[box(3.0, 51.0, 3.2, 51.2)])
    _, bounds = buffered_bbox(gdf, 0.0, "EPSG:32631", source_crs="EPSG:4326")
    assert bounds[0] < bounds[2] and bounds[1] < bounds[3]
    assert 400_000 < bounds[0] < 600_000


def test_snap_points_drops_far_points_and_moves_near_ones_inside():
    pts = _points((0.5, 0.5), (1.0005, 0.5), (3.0, 3.0))
    out, keep = snap_points_into_region(pts, REGION, buffer_deg=0.01)
    assert keep.tolist() == [True, True, False]
    assert len(out) == 2
    assert all(REGION.contains(g) for g in out.geometry)
    assert out.geometry.iloc[0].equals(Point(0.5, 0.5))


def test_snap_points_always_keep_snaps_every_point_inside():
    pts = _points((3.0, 3.0), (-2.0, 0.5))
    out, keep = snap_points_into_region(pts, REGION, buffer_deg=0.01, always_keep=True)
    assert keep.dtype == bool and keep.all()
    assert all(REGION.contains(g) for g in out.geometry)


def test_snap_points_returns_empty_when_none_kept():
    out, keep = snap_points_into_region(_points((5.0, 5.0)), REGION, buffer_deg=0.01)
    assert out.empty
    assert np.array_equal(keep, [False])
