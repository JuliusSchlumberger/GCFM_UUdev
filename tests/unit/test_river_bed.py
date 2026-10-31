"""Tests for the river bed below rule 10's surface (src.river_bed)."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString
from src.river_bed import (
    estimate_river_bed,
    forward_deviation_summary,
    manning_depth,
    node_widths,
    values_at_points,
)
from src.water_surface import build_water_surface, surface_table

CRS = "EPSG:32631"
Q = 400.0
N = 0.03
NO_SWORD_NODES = pd.DataFrame(columns=["reach_id", "x", "y", "width"])


def _rivers(width2=150.0) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "reach_id": ["1", "2"],
            "rch_id_dn": ["[2]", "[]"],
            "width": [120.0, width2],
            "obstr_type": [0, 0],
        },
        geometry=[
            LineString([(0.0, 0.0), (10000.0, 0.0)]),
            LineString([(10000.0, 0.0), (20000.0, 0.0)]),
        ],
        crs=CRS,
    )


def _surface(level_fn, rivers, water_surface_cfg, sample_nodes=None):
    def samples(lines):
        out = {}
        for r, line in lines.items():
            along = np.arange(0.0, line.length, 30.0)
            x0 = line.coords[0][0]
            out[r] = (along, level_fn(x0 + along))
        return out

    surface, _ = build_water_surface(
        rivers,
        samples,
        water_surface_cfg,
        sea_level_m=0.0,
        sample_nodes=sample_nodes,
        gebco_floor_m=-10.0,
    )
    return surface_table(surface, CRS)


def _run(
    rivers, level_fn, water_surface_cfg, bed_cfg, sample_nodes=None, reach_q_mean=None
):
    return estimate_river_bed(
        rivers,
        {"1": Q, "2": Q},
        _surface(level_fn, rivers, water_surface_cfg, sample_nodes),
        NO_SWORD_NODES,
        sea_level_m=0.0,
        gebco_shallow_m=-2.0,
        manning_n=N,
        surface_cfg=water_surface_cfg,
        cfg=bed_cfg,
        reach_q_mean=reach_q_mean,
    )


def test_mean_discharge_level_lies_below_the_design_level_and_meets_the_sea(
    water_surface_cfg, bed_cfg
):
    rb = _run(
        _rivers(),
        lambda x: 1e-4 * (20000.0 - x),
        water_surface_cfg,
        bed_cfg,
        reach_q_mean={"1": Q / 4, "2": Q / 4},
    )
    river = ~rb.bathymetry
    assert np.all(rb.mean_level_m[river] <= rb.forward_level_m[river] + 1e-9)
    assert np.all(rb.mean_level_m >= rb.bed_m - 1e-9)
    assert np.mean(rb.forward_level_m[river] - rb.mean_level_m[river]) > 0.3
    assert rb.mean_level_m[rb.mouths].tolist() == pytest.approx([0.0])
    assert np.isnan(
        _run(
            _rivers(), lambda x: 1e-4 * (20000.0 - x), water_surface_cfg, bed_cfg
        ).mean_level_m
    ).all()


def test_manning_depth_carries_the_discharge_at_the_slope():
    w, s = 150.0, 1e-4
    h = float(manning_depth(Q, w, N, s))
    r = w * h / (w + 2.0 * h)
    assert w * h * r ** (2 / 3) * np.sqrt(s) / N == pytest.approx(Q, rel=1e-5)


def test_uniform_surface_gives_manning_depth_below_it_and_the_mouth_at_the_sea(
    water_surface_cfg, bed_cfg
):
    s = 1e-4
    rb = _run(_rivers(), lambda x: s * (20000.0 - x), water_surface_cfg, bed_cfg)
    assert np.median(rb.slope) == pytest.approx(s, rel=0.05)
    assert np.allclose(
        rb.depth_m, manning_depth(Q, rb.width_m, N, rb.depth_slope), rtol=1e-9
    )
    assert np.allclose(rb.bed_m, rb.level_m - rb.depth_m)
    assert rb.level_m[rb.mouths].tolist() == [0.0]
    assert not rb.on_floor.any()


def test_forward_solve_on_the_bed_returns_the_surface_where_the_slope_steepens(
    water_surface_cfg, bed_cfg
):
    def level_fn(x):  # 1e-4 near the sea, 4e-4 upstream of x = 10 km
        return np.where(x < 10000.0, 1.0 + 4e-4 * (10000.0 - x), 1e-4 * (20000.0 - x))

    rb = _run(_rivers(), level_fn, water_surface_cfg, bed_cfg)
    np.testing.assert_allclose(rb.forward_level_m, rb.level_m, atol=0.05)
    fwd = forward_deviation_summary(rb)
    assert fwd["river_within_0.5m"] == 1.0
    assert abs(fwd["river_median_m"]) < 0.05


def test_flat_surface_keeps_its_level_and_the_depth_takes_the_floor(
    water_surface_cfg, bed_cfg
):
    rb = _run(_rivers(), lambda x: np.full_like(x, -0.4), water_surface_cfg, bed_cfg)
    assert np.allclose(rb.level_m, 0.0)  # held at the sea, not lifted by the floor
    assert rb.on_floor.all()
    assert np.allclose(
        rb.depth_m, manning_depth(Q, rb.width_m, N, bed_cfg["slope_floor"]), rtol=1e-9
    )


def test_gebco_bathymetry_kept_and_river_bed_blends_into_it(water_surface_cfg, bed_cfg):
    def sea_beyond_15km(x, y, w):
        x = np.asarray(x)
        return np.where(x > 15000.0, -6.0, np.nan), (x > 15000.0).astype(float)

    rb = _run(
        _rivers(width2=1500.0),
        lambda x: 1e-4 * (20000.0 - x),
        water_surface_cfg,
        bed_cfg,
        sea_beyond_15km,
    )
    assert rb.bathymetry.any()
    assert np.allclose(rb.bed_m[rb.bathymetry], -6.0)
    x = rb.net.node_x
    river_near = ~rb.bathymetry & (x > 15000.0 - bed_cfg["bathymetry_handover_m"])
    unblended = (rb.level_m - rb.depth_m)[river_near]
    lo, hi = np.minimum(unblended, -6.0), np.maximum(unblended, -6.0)
    assert (
        (rb.bed_m[river_near] >= lo - 1e-9) & (rb.bed_m[river_near] <= hi + 1e-9)
    ).all()


def test_node_widths_scaled_to_reach_width_and_smoothed_across_reach_ends(
    water_surface_cfg, bed_cfg
):
    rb = _run(_rivers(), lambda x: 1e-4 * (20000.0 - x), water_surface_cfg, bed_cfg)
    sword = pd.DataFrame(
        {
            "reach_id": ["1", "1", "1"],
            "x": [1000.0, 5000.0, 9000.0],
            "y": [0.0, 0.0, 0.0],
            "width": [60.0, 120.0, 240.0],
        }
    )
    owner = np.asarray(rb.net.owner)
    raw = node_widths(rb.net, {"1": 120.0, "2": 150.0}, sword, smoothing_m=0.0)
    assert raw[owner == "1"].min() == pytest.approx(60.0) and raw[
        owner == "1"
    ].max() == pytest.approx(240.0)
    assert np.allclose(raw[owner == "2"], 150.0)
    smooth = node_widths(rb.net, {"1": 120.0, "2": 150.0}, sword, smoothing_m=4000.0)
    order = np.argsort(rb.net.node_x)
    assert np.max(np.abs(np.diff(smooth[order]))) < np.max(np.abs(np.diff(raw[order])))


def test_forward_check_map_renders_from_the_depth_segments(
    tmp_path, water_surface_cfg, bed_cfg
):
    from shapely.geometry import box
    from src.plots import plot_river_depth_width
    from src.river_bed import depth_segments

    rb = _run(_rivers(), lambda x: np.full_like(x, -0.4), water_surface_cfg, bed_cfg)
    seg = depth_segments(rb, crs=CRS)
    assert np.isfinite(seg["forward_minus_surface_m"]).all()
    land = tmp_path / "land.gpkg"
    gpd.GeoDataFrame(geometry=[box(0, -2000, 20000, -500)], crs=CRS).to_file(land)
    out = tmp_path / "forward_check.png"
    plot_river_depth_width(
        seg,
        str(land),
        box(-500, -3000, 20500, 3000),
        str(out),
        "forward - surface (m)",
        "test",
        value_col="forward_minus_surface_m",
        value_range=(-3.0, 3.0),
        cmap_name="RdBu_r",
    )
    assert out.stat().st_size > 0


def test_values_at_points_interpolate_along_the_reach(water_surface_cfg, bed_cfg):
    rb = _run(_rivers(), lambda x: 1e-4 * (20000.0 - x), water_surface_cfg, bed_cfg)
    out = values_at_points(
        rb.net,
        rb.level_m,
        ["1", "2", "99"],
        np.array([5000.0, 15000.0, 0.0]),
        np.zeros(3),
    )
    assert out[0] == pytest.approx(1.5, abs=0.15) and out[1] == pytest.approx(
        0.5, abs=0.15
    )
    assert np.isnan(out[2])
