"""Smoke tests for the GVF pipeline driver (src.gvf)."""

from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import LineString
from src.gvf import (
    GvfRun,
    carry_bed_to_sea,
    depth_segments,
    mean_seed_discharge,
    run_gvf,
    slot_levels,
    values_at_points,
)
from src.water_surface import build_water_surface, surface_table

CRS = "EPSG:32631"
H_MAX = 30.0


def _rivers() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "reach_id": ["1", "2"],
            "rch_id_dn": ["[2]", "[]"],
            "width": [120.0, 150.0],
            "obstr_type": [0, 0],
        },
        geometry=[
            LineString([(0.0, 0.0), (10000.0, 0.0)]),
            LineString([(10000.0, 0.0), (20000.0, 0.0)]),
        ],
        crs=CRS,
    )


def _samples(lines):
    """FathomDEM-like centreline samples: reach 1 at ~6 m, reach 2 at ~3 m."""
    base = {"1": 6.0, "2": 3.0}
    out = {}
    for r, line in lines.items():
        along = np.arange(0.0, line.length, 30.0)
        out[r] = (along, base[r] - 1e-4 * along)
    return out


@pytest.fixture(scope="module")
def gvf_run(gvf_cfg, water_surface_cfg) -> GvfRun:
    surface, _ = build_water_surface(_rivers(), _samples, water_surface_cfg)
    return run_gvf(
        _rivers(),
        {"1": 400.0, "2": 400.0},
        surface_table(surface, CRS),
        surface.net.structure_edges,
        lambda x, y, w: np.zeros(len(x)),
        lambda x, y: np.full(len(x), np.nan),
        sea_level_m=0.0,
        gebco_floor_m=-10.0,
        outflow_reaches=set(),
        manning_n=0.03,
        surface_cfg=water_surface_cfg,
        cfg=gvf_cfg,
    )


def test_mean_seed_discharge_uses_active_crossings_and_max_per_reach():
    ds = xr.Dataset(
        {
            "mean_discharge_m3s": ("crossing", [10.0, 30.0, 99.0]),
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
    for reach_id, line in zip(_rivers()["reach_id"], _rivers().geometry):
        assert seg.loc[seg["reach_id"] == reach_id].length.sum() == pytest.approx(
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


def _carry_run(reaches, z, cls):
    """Fake GvfRun: reaches = {id: (slot ids downstream, downstream ids)}."""
    downstream = {r: list(d) for r, (_, d) in reaches.items()}

    def upstream():
        up = {r: [] for r in reaches}
        for r, dns in downstream.items():
            for d in dns:
                up[d].append(r)
        return up

    net = SimpleNamespace(
        reach_ids=list(reaches),
        downstream=downstream,
        upstream=upstream,
        reach_slots={r: np.array(s) for r, (s, _) in reaches.items()},
    )
    return SimpleNamespace(
        targets=SimpleNamespace(net=net),
        problem=SimpleNamespace(slot_class=np.array(cls, dtype=object)),
        result=SimpleNamespace(z=np.array(z, dtype=float)),
    )


def test_sea_slots_take_the_river_bed_unless_gebco_is_deeper():
    run = _carry_run(
        {"a": ([0, 1], ["c"]), "b": ([2, 3], ["c"]), "c": ([4, 5, 6], [])},
        z=[-9.0, -10.0, -7.0, -8.0, -2.7, -12.0, -1.5],
        cls=["invert"] * 4 + ["fixed_bed", "water_body", "fixed_bed"],
    )
    z = carry_bed_to_sea(run)
    # confluence: the deepest incoming river bed (-10) is carried into the sea reach
    np.testing.assert_array_equal(z, [-9.0, -10.0, -7.0, -8.0, -10.0, -12.0, -10.0])
    assert np.all(z <= run.result.z)  # never raises a bed
