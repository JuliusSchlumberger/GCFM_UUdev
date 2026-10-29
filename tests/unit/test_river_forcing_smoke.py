"""Smoke tests for river boundary forcing assembly (src.river_forcing)."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import LineString, Point, box
from src.river_forcing import (
    build_design_discharge_matrix,
    derive_forcing_mode,
    find_boundary_crossings,
    has_downstream_in_domain,
    interpolate_discharge_at_rp,
    load_forcing_crossings,
    resolve_inside_domain_reaches,
    snap_crossings_to_reaches,
)

RPS = np.array([2.0, 10.0, 100.0, 1000.0])
LEAD_DAYS, PERIOD_HR = 1.0, 48.0


def _river_ds() -> xr.Dataset:
    times = np.arange(0.0, 96.0 + 1.0, 1.0)
    table = np.array([[100.0, 200.0, 400.0, 600.0], [10.0, 30.0, 50.0, 80.0]])
    return xr.Dataset(
        {
            "discharge_rp_table": (("crossing", "rp_yr"), table),
            "mean_discharge_m3s": ("crossing", [50.0, 5.0]),
            "bankfull_discharge_m3s": ("crossing", [150.0, 20.0]),
            "has_glofas": ("crossing", [1, 1]),
            "inside_reach_id": ("crossing", ["11", "12"]),
            "longitude": ("crossing", [4.0, 4.1]),
            "latitude": ("crossing", [51.0, 51.1]),
        },
        coords={"time": times, "rp_yr": RPS},
        attrs={"lead_days": LEAD_DAYS, "period_hr": PERIOD_HR},
    )


def test_forcing_mode_follows_which_return_periods_are_set():
    assert derive_forcing_mode(100, 10) == "compound"
    assert derive_forcing_mode("mean", None) == "river_only"
    assert derive_forcing_mode(None, 10) == "coastal_only"
    with pytest.raises(ValueError):
        derive_forcing_mode(None, None)


def test_has_downstream_in_domain_parses_list_strings():
    assert has_downstream_in_domain("[11, 12]", {"12"})
    assert has_downstream_in_domain("11 12.0", {"12"})
    assert not has_downstream_in_domain("[]", {"12"})
    assert not has_downstream_in_domain(np.nan, {"12"})


def test_discharge_at_rp_increases_with_return_period_and_hits_table_nodes():
    table = _river_ds()["discharge_rp_table"].values
    q = np.stack(
        [
            interpolate_discharge_at_rp(table, RPS, rp_yr)
            for rp_yr in (2, 5, 10, 50, 100)
        ]
    )
    assert q.shape == (5, 2)
    assert np.all(np.diff(q, axis=0) > 0)
    np.testing.assert_allclose(
        interpolate_discharge_at_rp(table, RPS, 100.0), table[:, 2]
    )


def test_design_discharge_matrix_shape_non_negative_and_peaks_at_design_q():
    ds = _river_ds()
    active = np.array([True, True])
    q = build_design_discharge_matrix(ds, active, 100.0)
    assert q.shape == (2, ds.sizes["time"])
    assert np.all(q >= 0)
    np.testing.assert_allclose(q.max(axis=1), [400.0, 50.0])
    np.testing.assert_allclose(q[:, 0], ds["mean_discharge_m3s"].values)

    assert np.all(build_design_discharge_matrix(ds, active, None) == 0)
    mean_discharge_m3s = build_design_discharge_matrix(ds, active, "Mean")
    np.testing.assert_allclose(
        mean_discharge_m3s, ds["mean_discharge_m3s"].values[:, None] * np.ones_like(q)
    )
    assert (
        build_design_discharge_matrix(ds, np.array([False, True]), 10.0).shape[0] == 1
    )


def test_design_discharge_matrix_unknown_event_source_raises():
    ds = _river_ds()
    with pytest.raises(ValueError, match="glofas_event or cosine"):
        build_design_discharge_matrix(
            ds, np.array([True, True]), 100.0, event={"source": "cosin"}
        )


def test_load_and_snap_crossings_keep_largest_discharge_per_reach(tmp_path):
    ds = _river_ds()
    ds["has_glofas"] = ("crossing", [1, 0])
    path = tmp_path / "river_forcing.nc"
    ds.to_netcdf(path)
    crossings = load_forcing_crossings(path, "bankfull_discharge_m3s")
    assert len(crossings) == 1 and crossings.crs == "EPSG:4326"
    assert snap_crossings_to_reaches(crossings) == {"11": 150.0}

    multi = gpd.GeoDataFrame(
        {
            "inside_reach_id": ["7", "7.0", "8", ""],
            "bankfull_discharge_m3s": [5.0, 9.0, np.nan, 3.0],
        },
        geometry=[Point(0, 0)] * 4,
        crs="EPSG:4326",
    )
    assert snap_crossings_to_reaches(multi) == {"7": 9.0}
    assert snap_crossings_to_reaches(multi.iloc[0:0]) == {}


def test_boundary_crossing_resolves_to_first_reach_entering_domain():
    domain = box(0.0, 0.0, 1.0, 1.0)
    rivers = gpd.GeoDataFrame(
        {
            "reach_id": [1, 2, 3, 4],
            "rch_id_dn": ["[2]", "[]", "[4]", "[]"],
            "width": [50.0, 80.0, 30.0, 40.0],
        },
        geometry=[
            LineString([(-1.0, 0.5), (0.5, 0.5)]),
            LineString([(0.5, 0.5), (0.8, 0.5)]),
            LineString([(0.5, 3.0), (0.5, 2.0)]),  # never reaches the domain
            LineString([(0.5, 2.0), (0.5, 1.2)]),
        ],
        crs="EPSG:4326",
    )
    crossings = find_boundary_crossings(rivers, domain)
    assert len(crossings) == 1
    assert domain.exterior.distance(crossings.geometry.iloc[0]) < 1e-12

    outside = gpd.GeoDataFrame(
        {"reach_id": [3], "width": [30.0]}, geometry=[Point(0.5, 3.0)], crs="EPSG:4326"
    )
    resolved = resolve_inside_domain_reaches(
        gpd.GeoDataFrame(
            pd.concat([crossings, outside], ignore_index=True), crs="EPSG:4326"
        ),
        rivers,
        domain,
    )
    assert resolved["inside_reach_id"].tolist() == ["1", None]
    assert resolved.geometry.iloc[0].distance(Point(0.0, 0.5)) < 1e-12
    assert resolved.geometry.iloc[1] == Point(0.5, 3.0)
    assert resolved["width"].tolist() == [50.0, 30.0]


def test_entry_point_is_the_upstream_end_inside_the_domain_for_an_upstream_running_line():
    domain = box(0.0, 0.0, 1.0, 1.0)
    rivers = gpd.GeoDataFrame(
        {"reach_id": [1, 2], "rch_id_dn": ["[2]", "[]"], "width": [50.0, 80.0]},
        geometry=[
            LineString(
                [(0.5, 0.5), (-1.0, 0.5)]
            ),  # SWORD-style: drawn from its downstream end
            LineString([(0.8, 0.5), (0.5, 0.5)]),
        ],
        crs="EPSG:4326",
    )
    crossings = gpd.GeoDataFrame(
        {"reach_id": [1], "width": [50.0]}, geometry=[Point(0.0, 0.5)], crs="EPSG:4326"
    )
    resolved = resolve_inside_domain_reaches(crossings, rivers, domain)
    assert resolved["inside_reach_id"].tolist() == ["1"]
    assert resolved.geometry.iloc[0].distance(Point(0.0, 0.5)) < 1e-12


def test_reach_touching_boundary_walks_to_downstream_reach():
    domain = box(0.0, 0.0, 1.0, 1.0)
    rivers = gpd.GeoDataFrame(
        {"reach_id": [1, 2], "rch_id_dn": ["[2]", "[]"], "width": [50.0, 80.0]},
        geometry=[
            LineString([(-1.0, 0.5), (0.0, 0.5)]),  # ends on the boundary
            LineString([(0.0, 0.5), (0.5, 0.5)]),
        ],
        crs="EPSG:4326",
    )
    crossings = gpd.GeoDataFrame(
        {"reach_id": [1], "width": [50.0]}, geometry=[Point(0.0, 0.5)], crs="EPSG:4326"
    )
    resolved = resolve_inside_domain_reaches(crossings, rivers, domain)
    assert resolved["inside_reach_id"].tolist() == ["2"]
    assert resolved["width"].tolist() == [80.0]
    assert resolved.geometry.iloc[0].distance(Point(0.0, 0.5)) < 1e-12
