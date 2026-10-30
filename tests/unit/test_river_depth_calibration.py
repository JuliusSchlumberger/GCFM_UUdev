"""Smoke tests for the cross-section helpers of src/river_depth_calibration.py."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point
from src.river_depth_calibration import bifurcation_cross_sections, mouth_sections

CRS = "EPSG:32631"


def test_mouth_section_sits_upstream_of_the_mouth_and_spans_the_channel():
    x = np.arange(0.0, 5001.0, 500.0)  # reach runs east, mouth at x = 5000
    nodes = gpd.GeoDataFrame(
        {
            "reach_id": ["1"] * len(x),
            "node_id": np.arange(len(x)),
            "chainage_m": x,
            "width_m": 200.0,
            "is_mouth": x == x[-1],
        },
        geometry=[Point(v, 0.0) for v in x],
        crs=CRS,
    )
    (sec,) = mouth_sections(nodes, upstream_m=1000.0, margin_m=150.0).itertuples()
    assert sec.name == "mouth_1"
    assert (sec.x, sec.y) == pytest.approx((4000.0, 0.0))
    (x0, y0), (x1, y1) = sec.geometry.coords
    assert x0 == pytest.approx(4000.0) and x1 == pytest.approx(
        4000.0
    )  # across the reach
    assert abs(y1 - y0) == pytest.approx(200.0 + 2 * 150.0)


def test_bifurcation_sections_cross_each_arm_once():
    rivers = gpd.GeoDataFrame(
        {
            "reach_id": ["1", "2", "3"],
            "rch_id_dn": ["2 3", "", ""],
            "width": [300.0, 200.0, 100.0],
        },
        geometry=[
            LineString([(0, 0), (10000, 0)]),
            LineString([(10000, 0), (20000, 0)]),
            LineString([(10000, 0), (15000, 5000)]),
        ],
        crs=CRS,
    )
    crs = bifurcation_cross_sections(rivers, ["1"], offset_m=500.0, margin_m=100.0)
    assert sorted(crs["arm_reach"]) == ["2", "3"]
    assert set(crs["parent_reach"]) == {"1"}
    arm2 = crs[crs["arm_reach"] == "2"].geometry.iloc[0]
    assert arm2.length == pytest.approx(200.0 + 2 * 100.0)
    assert arm2.intersects(rivers.geometry.iloc[1])
