"""Tests for src.water_surface: FathomDEM water surface, its slope, and channel conditioning."""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString
from src.water_surface import (
    build_water_surface,
    condition_channels_to_surface,
    read_water_surface,
    river_cells_at_surface,
    sample_centrelines,
    surface_table,
    write_water_surface,
)

CRS = "EPSG:32631"
WATER = 80


def _rivers(spec):
    rows = [
        {
            "reach_id": rid,
            "rch_id_dn": "[" + ", ".join(s.get("dn", [])) + "]",
            "width": float(s.get("width", 100.0)),
            "obstr_type": 0,
            "geometry": LineString(s["xy"]),
        }
        for rid, s in spec.items()
    ]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)


def _sampler(profile):
    """Samples every 30 m from profile(reach, along_m)."""

    def sample(lines):
        out = {}
        for r, line in lines.items():
            along = np.arange(0.0, line.length + 1e-9, 30.0)
            out[r] = (along, profile(r, along))
        return out

    return sample


def _reach_profile(surface, r):
    ids = surface.net.reach_nodes[r]
    return surface.net.reach_chainage[r], surface.level_m[ids]


def test_median_removes_isolated_peaks_and_dips(water_surface_cfg):
    rivers = _rivers({"1": {"xy": [(0, 0), (20000, 0)]}})

    def profile(r, a):
        z = 5.0 - 1e-4 * a
        z[(a > 6000) & (a < 6100)] += 6.0  # bridge / bar
        z[(a > 14000) & (a < 14100)] -= 9.0  # single low pixels / GEBCO
        return z

    surface, _ = build_water_surface(rivers, _sampler(profile), water_surface_cfg)
    ch, lvl = _reach_profile(surface, "1")
    assert np.max(np.abs(lvl - (5.0 - 1e-4 * ch))) < 0.05
    assert np.all(np.diff(lvl) <= 1e-9)


def test_long_slope_smoothing_keeps_the_level_and_lifts_tread_slopes(water_surface_cfg):
    rivers = _rivers({"1": {"xy": [(0, 0), (40000, 0)]}})
    # staircase: 10 km flat treads dropping 1 m at each riser
    sample = _sampler(lambda r, a: 4.0 - np.floor(a / 10000.0))
    short, _ = build_water_surface(rivers, sample, water_surface_cfg)
    long_, _ = build_water_surface(
        rivers, sample, dict(water_surface_cfg, slope_smoothing_m=20000)
    )
    np.testing.assert_array_equal(short.level_m, long_.level_m)
    tread = short.net.reach_nodes["1"][
        2:8
    ]  # within the first tread, away from its riser
    assert np.max(short.slope[tread]) < 1e-5
    assert np.min(long_.slope[tread]) > 2e-5


def test_stepped_flat_surface_becomes_monotone_with_one_junction_value(
    water_surface_cfg,
):
    rivers = _rivers(
        {
            "1": {"xy": [(0, 0), (10000, 0)], "dn": ["2", "3"]},
            "2": {"xy": [(10000, 0), (20000, 0)]},
            "3": {"xy": [(10000, 0), (20000, 3000)]},
        }
    )
    # FathomDEM-like: flat steps of 0.2 m per reach, noise, a downstream rise
    rng = np.random.default_rng(3)
    base = {"1": 2.4, "2": 2.2, "3": 2.25}
    surface, _ = build_water_surface(
        rivers,
        _sampler(lambda r, a: base[r] + 0.05 * rng.standard_normal(a.size)),
        water_surface_cfg,
    )
    net = surface.net
    for a, b, _ in net.node_edges():
        assert surface.level_m[a] >= surface.level_m[b] - 1e-9
    j = net.reach_nodes["1"][-1]
    assert net.reach_nodes["2"][0] == j == net.reach_nodes["3"][0]
    assert 2.15 < surface.level_m[j] < 2.45


def test_structure_is_smoothed_across_by_default(water_surface_cfg):
    rivers = _rivers(
        {
            "1": {"xy": [(0, 0), (10000, 0)], "dn": ["2"]},
            "2": {"xy": [(10000, 0), (20000, 0)]},
        }
    )
    surface, _ = build_water_surface(
        rivers,
        _sampler(lambda r, a: np.full(a.size, 6.0 if r == "1" else 3.0)),
        water_surface_cfg,
    )
    net = surface.net
    assert surface.structures == {("1", "2")}  # detected and reported
    assert net.structure_edges == set()  # but not split
    assert net.reach_nodes["1"][-1] == net.reach_nodes["2"][0]
    drops = [surface.level_m[a] - surface.level_m[b] for a, b, _ in net.node_edges()]
    assert max(drops) < 1.5  # the 3 m drop is spread over the smoothing length
    assert min(drops) >= -1e-9


def _rasters(tmp_path):
    """3 km x 0.6 km DEM at 30 m: channel along y = 300 m, sea for x > 2400 m."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    nx, ny, px = 100, 20, 30.0
    tr = from_origin(0.0, ny * px, px, px)
    x = (np.arange(nx) + 0.5) * px
    dem = np.tile(4.0 - 1e-3 * x, (ny, 1)).astype("float32")
    dem[:, 30] += 5.0  # bridge across the channel
    lu = np.full((ny, nx), 80, dtype="uint8")
    lu[:, x > 2400] = 200
    dem[:, x > 2400] = -6.0  # GEBCO bed
    paths = {}
    for name, arr, dtype in [("dem", dem, "float32"), ("lu", lu, "uint8")]:
        paths[name] = str(tmp_path / f"{name}.tif")
        with rasterio.open(
            paths[name],
            "w",
            driver="GTiff",
            width=nx,
            height=ny,
            count=1,
            dtype=dtype,
            crs=CRS,
            transform=tr,
            nodata=-9999 if dtype == "float32" else None,
        ) as dst:
            dst.write(arr, 1)
    return paths


def test_conditioning_sets_land_channel_pixels_and_keeps_sea_bed(
    tmp_path, water_surface_cfg
):
    paths = _rasters(tmp_path)
    rivers = _rivers({"1": {"xy": [(0, 300), (3000, 300)], "width": 120.0}})
    surface, rivers_o = build_water_surface(
        rivers,
        lambda lines: sample_centrelines(lines, paths["dem"], paths["lu"], WATER),
        water_surface_cfg,
    )
    out = str(tmp_path / "cond.tif")
    condition_channels_to_surface(surface, rivers_o, paths["dem"], paths["lu"], out)
    with rasterio.open(paths["dem"]) as a, rasterio.open(out) as b:
        before, after = a.read(1), b.read(1)
    row = 10  # channel row (y = 300 m)
    assert after[row, 30] < before[row, 30] - 4.0  # bridge removed
    assert np.all(after[row, 81:] == before[row, 81:])  # sea pixels keep GEBCO
    assert np.all(after[0] == before[0])  # outside the 120 m buffer untouched
    assert np.all(np.diff(after[row, :80]) <= 1e-5)  # monotone along the channel


def test_island_under_the_centreline_is_skipped(tmp_path, water_surface_cfg):
    rivers = _rivers({"1": {"xy": [(0, 300), (2400, 300)], "width": 120.0}})

    def surface_and_samples(paths):
        def sample(lines):
            return sample_centrelines(lines, paths["dem"], paths["lu"], WATER)

        surface, _ = build_water_surface(rivers, sample, water_surface_cfg)
        return _reach_profile(surface, "1")[1], sample({"1": rivers.geometry.iloc[0]})[
            "1"
        ]

    plain, _ = surface_and_samples(_rasters(tmp_path / "plain"))
    island_dir = tmp_path / "island"
    paths = _rasters(island_dir)
    with (
        rasterio.open(paths["lu"], "r+") as lu,
        rasterio.open(paths["dem"], "r+") as dem,
    ):
        lu_arr, dem_arr = lu.read(1), dem.read(1)
        lu_arr[:, 40:55] = 30  # grassland island, 450 m long, 2.5 m above the water
        dem_arr[:, 40:55] += 2.5
        lu.write(lu_arr, 1)
        dem.write(dem_arr, 1)
    with_island, (along, _, water) = surface_and_samples(paths)
    on_island = (along >= 1200.0) & (along < 1650.0)
    assert on_island.any() and not water[on_island].any()  # island pixels skipped
    # The gap leaves one-sided median windows next to it: on this 1 m/km slope
    # that shifts a node by ~0.1 m, against the island's 2.5 m.
    assert np.max(np.abs(with_island - plain)) < 0.25


def test_fast_chainage_within_half_a_step_of_the_projection():
    from src.river_network import line_window_cells

    line = LineString([(0, 0), (3000, 0), (3000, 2000)])
    tr = from_origin(-100.0, 2100.0, 10.0, 10.0)
    exact = line_window_cells(line.buffer(50.0), line, tr, (230, 330))
    fast = line_window_cells(line.buffer(50.0), line, tr, (230, 330), along_step_m=30.0)
    assert np.array_equal(exact[0], fast[0]) and np.array_equal(exact[1], fast[1])
    straight = (exact[2] > 100) & (exact[2] < 2900)  # away from the bend
    assert np.max(np.abs(exact[2][straight] - fast[2][straight])) <= 15.0 + 1e-6


def test_water_surface_profile_tables_drive_the_figure(tmp_path, water_surface_cfg):
    from src.plots import plot_water_surface_profiles, water_surface_profile_tables

    rivers = _rivers(
        {
            "1": {"xy": [(0, 0), (10000, 0)], "dn": ["2"]},
            "2": {"xy": [(10000, 0), (20000, 0)]},
        }
    )
    rivers["obstr_type"] = [2, 0]
    surface, rivers_o = build_water_surface(
        rivers,
        _sampler(lambda r, a: (6.0 if r == "1" else 3.0) - 1e-4 * a),
        water_surface_cfg,
    )
    points, paths = water_surface_profile_tables(surface, rivers_o, ["1"])
    assert set(points["kind"]) >= {"water", "node"}
    assert paths["n_reaches"].iloc[0] == 2
    nodes = points[points["kind"] == "node"]
    assert nodes["along_m"].is_monotonic_increasing
    out = tmp_path / "10_river_surface.png"
    plot_water_surface_profiles(points, paths, str(out), "test")
    assert out.stat().st_size > 0


def test_short_high_reach_is_outvoted_by_the_river_around_it(water_surface_cfg):
    # A 400 m lock reach whose FathomDEM pixels are walls (+3 m) between river at 0 m.
    rivers = _rivers(
        {
            "1": {"xy": [(0, 0), (5000, 0)], "dn": ["2"]},
            "2": {"xy": [(5000, 0), (5400, 0)], "dn": ["3"]},
            "3": {"xy": [(5400, 0), (10000, 0)]},
        }
    )
    surface, _ = build_water_surface(
        rivers,
        _sampler(lambda r, a: np.full(a.size, 3.0 if r == "2" else 0.0)),
        water_surface_cfg,
    )
    assert surface.rising_edges == []
    assert np.max(surface.level_m) < 0.5


def test_dry_channel_cells_take_nearest_surface_level_and_sea_keeps_its_own():
    from affine import Affine

    nodes = gpd.GeoDataFrame(
        {"level_m": [3.0, 1.0]},
        geometry=gpd.points_from_xy([50.0, 350.0], [-50.0, -50.0]),
        crs=CRS,
    )
    ini = np.full((2, 4), np.nan, dtype=np.float32)
    ini[0, 3] = 0.2  # sea
    channel = np.zeros((2, 4), dtype=bool)
    channel[0, :] = True
    for transform in (
        from_origin(0.0, 0.0, 100.0, 100.0),
        Affine.translation(0.0, 0.0) * Affine.scale(100.0, -100.0),
    ):
        out, n = river_cells_at_surface(ini, nodes, channel, transform)
        assert n == 3
        assert out[0, 0] == pytest.approx(3.0) and out[0, 2] == pytest.approx(1.0)
        assert out[0, 3] == pytest.approx(0.2)
        assert np.isnan(out[1]).all()
        assert np.isnan(ini[0, 0])  # input untouched


def test_surface_round_trip_and_rule_11_rebuilds_the_same_network(
    tmp_path, water_surface_cfg
):
    from src.gvf_target import build_node_network, orient_downstream
    from src.river_bed import surface_on_network
    from src.river_network import build_downstream_adjacency

    rivers = _rivers(
        {
            "1": {"xy": [(0, 0), (10000, 0)], "dn": ["2", "3"]},
            "2": {"xy": [(10000, 0), (20000, 0)]},
            "3": {"xy": [(10000, 0), (20000, 3000)]},
        }
    )
    surface, _ = build_water_surface(
        rivers,
        _sampler(lambda r, a: (6.0 if r == "1" else 3.0) - 1e-4 * a),
        water_surface_cfg,
        sea_level_m=1.6,
    )
    path = tmp_path / "surface.gpkg"
    write_water_surface(surface, path, CRS)
    nodes, structures = read_water_surface(path)
    assert structures == surface.structures
    assert np.allclose(nodes["level_m"], surface_table(surface, CRS)["level_m"])
    dn = build_downstream_adjacency(rivers)
    net = build_node_network(
        orient_downstream(rivers, dn)[0], dn, water_surface_cfg["node_spacing_m"]
    )
    t = surface_on_network(net, nodes)
    assert np.allclose(t["level_m"], surface.level_m) and np.allclose(
        t["slope"], surface.slope
    )


def test_mouth_is_pinned_at_the_sea_and_nothing_lies_below_it(water_surface_cfg):
    sea = 0.3
    rivers = _rivers({"1": {"xy": [(0, 0), (20000, 0)]}})
    # FathomDEM-like: the lower 8 km flattened to the geoid, below the sea.
    surface, _ = build_water_surface(
        rivers,
        _sampler(lambda r, a: np.maximum(2.0 - 1.6e-4 * a, -0.2)),
        water_surface_cfg,
        sea_level_m=sea,
    )
    ch, lvl = _reach_profile(surface, "1")
    assert surface.level_m[surface.mouths].tolist() == [sea]
    assert lvl[-1] == sea and lvl.min() >= sea - 1e-12
    assert np.all(np.diff(lvl) <= 1e-9)
    assert np.all(surface.slope >= 0)


def test_all_sea_reach_takes_the_sea_level_at_its_mouth(water_surface_cfg):
    rivers = _rivers(
        {
            "1": {"xy": [(0, 0), (10000, 0)], "dn": ["2"]},
            "2": {"xy": [(10000, 0), (15000, 0)]},
        }
    )

    def sample(lines):
        a1 = np.arange(0.0, lines["1"].length, 30.0)
        return {"1": (a1, 3.0 - 1e-4 * a1), "2": (np.array([]), np.array([]))}

    def sea_beyond_10km(x, y, w):  # reach 2 is open sea on GEBCO bathymetry
        return np.full(len(x), -5.0), (np.asarray(x) > 10000.0).astype(float)

    surface, _ = build_water_surface(
        rivers, sample, water_surface_cfg, sea_level_m=0.1, sample_nodes=sea_beyond_10km
    )
    assert surface.level_m[surface.net.reach_nodes["2"][-1]] == 0.1
    assert surface.level_m.min() >= 0.1 - 1e-12


def test_no_sea_mouth_fails_loudly(water_surface_cfg):
    rivers = _rivers({"1": {"xy": [(0, 0), (10000, 0)]}})
    with pytest.raises(ValueError, match="no sea mouth"):
        build_water_surface(
            rivers,
            _sampler(lambda r, a: 50.0 - 1e-4 * a),
            water_surface_cfg,
            sea_level_m=0.0,
        )
