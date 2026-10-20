"""Smoke tests for the GVF pipeline driver (src.gvf)."""

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import LineString
from src.gvf import (
    GvfRun,
    depth_segments,
    mean_seed_discharge,
    run_gvf,
    slot_levels,
    values_at_points,
)

CRS = "EPSG:32631"
CFG = {"node_spacing_m": 500, "target_smoothing_m": 3000, "manning_channel": 0.03}
H_MIN, H_MAX = 0.5, 30.0


def _rivers() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "reach_id": ["1", "2"],
            "rch_id_dn": ["[2]", "[]"],
            "width": [120.0, 150.0],
            "wse": [6.0, 3.0],
            "obstr_type": [0, 0],
        },
        geometry=[
            LineString([(0.0, 0.0), (10000.0, 0.0)]),
            LineString([(10000.0, 0.0), (20000.0, 0.0)]),
        ],
        crs=CRS,
    )


@pytest.fixture(scope="module")
def gvf_run() -> GvfRun:
    return run_gvf(
        _rivers(),
        {"1": 400.0, "2": 400.0},
        np.zeros(2),
        lambda x, y, w: np.zeros(len(x)),
        lambda x, y: np.full(len(x), np.nan),
        sea_level_m=0.0,
        gebco_floor_m=-10.0,
        outflow_reaches=set(),
        cfg=CFG,
    )


def test_mean_seed_discharge_uses_active_crossings_and_max_per_reach():
    ds = xr.Dataset(
        {
            "mean_discharge": ("crossing", [10.0, 30.0, 99.0]),
            "inside_reach_id": ("crossing", ["5", "5", "6"]),
            "longitude": ("crossing", [4.0, 4.1, 4.2]),
            "latitude": ("crossing", [51.0, 51.1, 51.2]),
        }
    )
    assert mean_seed_discharge(ds, np.array([True, True, False])) == {"5": 30.0}


def test_gvf_depths_positive_and_within_solver_bounds(gvf_run):
    h = slot_levels(gvf_run) - gvf_run.result.z
    assert h.shape == gvf_run.result.z.shape
    assert np.all(np.isfinite(h))
    assert np.all(h > 0)
    assert np.all(h <= H_MAX + 1e-6)


def test_depth_segments_cover_reaches_with_positive_depth(gvf_run):
    seg = depth_segments(gvf_run, crs=CRS)
    assert set(seg["reach_id"]) == {"1", "2"}
    assert seg.crs == CRS
    assert (seg["depth"] > 0).all()
    assert not seg["fallback"].any()
    for rid, line in zip(_rivers()["reach_id"], _rivers().geometry):
        assert seg.loc[seg["reach_id"] == rid].length.sum() == pytest.approx(
            line.length
        )


def test_values_at_points_bounded_by_slot_values_and_nan_off_network(gvf_run):
    z = gvf_run.result.z
    x = np.array([0.0, 2500.0, 15000.0, 20000.0, 5000.0])
    y = np.zeros(5)
    out = values_at_points(gvf_run, z, ["1", "1", "2", "2", "99"], x, y)
    assert out.shape == (5,)
    assert np.isnan(out[4])
    assert np.all((out[:4] >= z.min()) & (out[:4] <= z.max()))
    # bed falls downstream along this monotone-target chain
    assert out[0] >= out[3]
