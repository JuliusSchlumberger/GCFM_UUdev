"""Smoke tests for the failure modes of src.plots writers."""

import errno

import numpy as np
import pytest
import src.plots as plots
import xarray as xr
from affine import Affine


class _FakeFigure:
    def __init__(self, errors: list[OSError]):
        self.errors = list(errors)
        self.calls = 0

    def savefig(self, path, **_kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        path.write_bytes(b"png")


@pytest.fixture
def no_wait(monkeypatch):
    monkeypatch.setattr(plots, "_SAVE_RETRY_WAIT_S", 0.0)


def test_save_figure_retries_windows_file_lock(tmp_path, no_wait):
    fig = _FakeFigure([OSError(errno.EINVAL, "Invalid argument")])
    out = tmp_path / "fig.png"
    plots.save_figure(fig, out, close=False)
    assert fig.calls == 2
    assert out.exists()


def test_save_figure_reraises_non_lock_oserror_immediately(tmp_path, no_wait):
    fig = _FakeFigure([OSError(errno.ENOSPC, "No space left on device")])
    with pytest.raises(OSError) as excinfo:
        plots.save_figure(fig, tmp_path / "fig.png", close=False)
    assert excinfo.value.errno == errno.ENOSPC
    assert fig.calls == 1


def test_animate_flood_progression_without_time_dim_raises(tmp_path):
    da_h = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"))
    out = tmp_path / "anim.mp4"
    with pytest.raises(ValueError, match="no 'time' dimension"):
        plots.animate_flood_progression(da_h, None, "", "", str(out))
    assert not out.exists()


def test_crest_gap_map_without_crest_cells_raises(tmp_path):
    out = tmp_path / "crest_gap.png"
    with pytest.raises(ValueError, match="no finite cell"):
        plots.plot_crest_gap_map(
            np.full((3, 3), np.nan),
            Affine.identity(),
            "",
            None,
            None,
            np.array([]),
            np.array([]),
            str(out),
        )
    assert not out.exists()


def _round_profile():
    import pandas as pd

    along = np.array([0.0, 500.0, 1000.0])
    return pd.DataFrame(
        {
            "along_path_m": along,
            "dem_m": [2.0, 1.0, 0.0],
            "river_depth_m": [1.5, 1.0, 0.0],
            "weir_crest_m": [3.0, 2.0, np.nan],
            "zs_m": [2.5, 1.5, 0.2],
        }
    )


def test_crest_profiles_one_panel_per_seed_with_dem_bed_and_crest(
    monkeypatch, tmp_path
):
    figs = []
    monkeypatch.setattr(
        plots, "save_figure_deliverable", lambda fig, *_a, **_k: figs.append(fig)
    )
    plots.plot_crest_profiles(
        "0", {"1": _round_profile(), "2": _round_profile()}, str(tmp_path / "p.png")
    )
    axes = figs[0].axes
    assert len(axes) == 2
    for letter, ax in zip("ab", axes):
        assert f"({letter})" in [t.get_text() for t in ax.texts]
        bed = ax.get_lines()[1].get_ydata()
        assert np.allclose(bed, [0.5, 0.0, 0.0])  # dem_m - river_depth_m
    (legend,) = figs[0].legends  # one legend below the plot for all panels
    assert [t.get_text() for t in legend.get_texts()] == [
        "DEM",
        "River bed",
        "Dike crest",
    ]
    caption = (tmp_path / "p.txt").read_text(encoding="utf-8")
    assert "(a) seed reach 1" in caption and "(b) seed reach 2" in caption


def test_crest_profiles_without_seed_still_writes(tmp_path):
    out = tmp_path / "p.png"
    plots.plot_crest_profiles("0", {}, str(out))
    assert out.exists()


def test_run_forcing_shows_only_tstart_to_tstop(monkeypatch, tmp_path):
    (tmp_path / "sfincs.inp").write_text(
        "tref = 20000101 000000\ntstart = 20000101 010000\ntstop = 20000101 030000\n"
        "bzsfile = sfincs.bzs\ndisfile = sfincs.dis\n"
    )
    np.savetxt(
        tmp_path / "sfincs.bzs",
        [[0, 0.1, 0.2], [7200, 0.5, 0.6], [14400, 0.3, 0.4]],
    )
    np.savetxt(tmp_path / "sfincs.dis", [[0, 10.0], [14400, 50.0]])
    figs = []
    monkeypatch.setattr(
        plots, "save_figure_deliverable", lambda fig, *_a, **_k: figs.append(fig)
    )
    plots.plot_run_forcing(tmp_path, tmp_path / "f.png", "0", "default")
    ax_wl, ax_q = figs[0].axes
    assert len(ax_wl.get_lines()) == 1  # mean of the two points, no start marker
    assert len(ax_wl.collections) == 1  # range band across the two points
    assert not ax_q.collections  # a single point gets no band
    days = ax_wl.get_lines()[0].get_xdata()
    assert np.isclose(days[0], 0.0) and np.isclose(days[-1], 2 / 24)
    assert np.isclose(
        ax_q.get_lines()[0].get_ydata()[0], 20.0
    )  # interpolated at tstart
    caption = (tmp_path / "f.txt").read_text(encoding="utf-8")
    assert "default" in caption and "maximum 0.60 m" in caption


def test_storm_tide_shapes_drawn_from_the_hydrograph_file(tmp_path):
    import xarray as xr

    rp = np.array([1, 10, 50, 100, 250, 1000], dtype=float)
    t = np.linspace(-24, 24, 49)
    shape = np.exp(-((t / 6.0) ** 2))[None, None, :].repeat(len(rp), 1).repeat(2, 0)
    ds = xr.Dataset(
        {
            "average_tide": (
                ("station", "time_hr"),
                np.cos(t / 12.42 * 2 * np.pi)[None].repeat(2, 0),
            ),
            "surge_shape": (("station", "rp_yr", "time_hr"), shape),
            "storm_tide_level_m": (
                ("station", "rp_yr"),
                np.tile(np.linspace(1.5, 3.0, len(rp)), (2, 1)),
            ),
            "shape_n_events": (("station", "rp_yr"), np.full((2, len(rp)), 5)),
            "shape_dur50_hr": (("station", "rp_yr"), np.full((2, len(rp)), 12.0)),
            "shape_mode": (("station", "rp_yr"), np.full((2, len(rp)), "window")),
            "high_water_m": (("station",), [1.0, 1.0]),
        },
        coords={"time_hr": t, "rp_yr": rp, "gtsm_station": ("station", [7, 8])},
        attrs={"match_on": "surge_peak"},
    )
    ds.to_netcdf(tmp_path / "h.nc")
    out = tmp_path / "shapes.png"
    plots.plot_storm_tide_shapes(tmp_path / "h.nc", "0", out)
    assert out.stat().st_size > 0


def test_run_forcing_stops_at_tstop_when_the_file_runs_longer(monkeypatch, tmp_path):
    (tmp_path / "sfincs.inp").write_text(
        "tref = 20000101 000000\ntstart = 20000101 000000\ntstop = 20000103 000000\n"
        "bzsfile = sfincs.bzs\n"
    )
    np.savetxt(tmp_path / "sfincs.bzs", [[0, 0.0], [86400 * 14, 1.4]])
    figs = []
    monkeypatch.setattr(
        plots, "save_figure_deliverable", lambda fig, *_a, **_k: figs.append(fig)
    )
    plots.plot_run_forcing(tmp_path, tmp_path / "f.png", "0", "spinup")
    ax_wl = figs[0].axes[0]
    days = ax_wl.get_lines()[0].get_xdata()
    assert days[-1] == 2.0
    assert ax_wl.get_xlim()[1] == 2.0
    caption = (tmp_path / "f.txt").read_text(encoding="utf-8")
    assert "maximum 0.20 m" in caption  # interpolated at tstop, not the file's 1.40 m


def test_skeleton_maps_write_three_maps_and_keep_inactive_cells_out(tmp_path):
    import geopandas as gpd
    import hydromt  # noqa: F401 -- registers the .raster accessor
    import xarray as xr
    from shapely.geometry import Point, box

    x = 500_000 + 100.0 * np.arange(20) + 50
    y = 4_500_000 + 100.0 * np.arange(15)[::-1] + 50
    mask = np.ones((15, 20), dtype=np.uint8)
    mask[:, :5] = 0  # inactive
    mask[:, -1] = 2  # water-level boundary
    dep = np.linspace(-25.0, 40.0, 300).reshape(15, 20).astype(np.float32)
    manning = np.where(dep < 0, 0.02, 0.06).astype(np.float32)
    grid = xr.Dataset(
        {
            "dep": (("y", "x"), dep),
            "mask": (("y", "x"), mask),
            "manning": (("y", "x"), manning),
        },
        coords={"x": x, "y": y},
    )
    grid.raster.set_crs(32631)
    land = tmp_path / "land.gpkg"
    gpd.GeoDataFrame(
        geometry=[box(500_000, 4_500_000, 501_000, 4_501_500)], crs=32631
    ).to_crs(4326).to_file(land)
    domain = gpd.GeoSeries(
        [box(500_100, 4_500_100, 501_900, 4_501_400)], crs=32631
    ).to_crs(4326)[0]
    obs = gpd.GeoDataFrame(geometry=[Point(501_000, 4_500_700)], crs=32631)
    paths = [tmp_path / f"{n}.png" for n in ("elevation", "mask", "roughness")]

    plots.plot_skeleton_maps(grid, obs, domain, str(land), *paths, basin_id="0")

    for p in paths:
        assert p.exists() and p.with_suffix(".pdf").exists()
    elevation_caption = paths[0].with_suffix(".txt").read_text(encoding="utf-8")
    # inactive cells (the -25 m corner) are left out; +40 m is beyond the +30 m cap
    assert "-25.0" not in elevation_caption and "40.0 m" in elevation_caption
    assert "water-level boundary cell" in paths[1].with_suffix(".txt").read_text(
        encoding="utf-8"
    )


def test_initial_water_level_counts_sea_river_and_dried_pocket_cells(tmp_path):
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box

    calm = -0.05
    zsini = np.full((10, 12), -9999.0, dtype=np.float32)
    zsini[:, 6:] = calm  # sea
    zsini[4, :6] = np.linspace(0.5, 3.0, 6)  # river channel, above the calm level
    zsini[0, 11] = -2.0  # isolated sea pocket, dry at bed level
    tif = tmp_path / "zsini.tif"
    with rasterio.open(
        tif,
        "w",
        driver="GTiff",
        height=10,
        width=12,
        count=1,
        dtype="float32",
        crs="EPSG:32631",
        transform=from_origin(500_000, 4_501_000, 100, 100),
        nodata=-9999.0,
    ) as dst:
        dst.write(zsini, 1)
    land = tmp_path / "land.gpkg"
    gpd.GeoDataFrame(
        geometry=[box(500_000, 4_500_000, 500_600, 4_501_000)], crs=32631
    ).to_crs(4326).to_file(land)
    domain = gpd.GeoSeries(
        [box(500_000, 4_500_000, 501_200, 4_501_000)], crs=32631
    ).to_crs(4326)[0]
    out = tmp_path / "zsini.png"

    plots.plot_initial_water_level(tif, domain, str(land), out, calm, basin_id="0")

    assert out.exists() and out.with_suffix(".pdf").exists()
    caption = out.with_suffix(".txt").read_text(encoding="utf-8")
    assert "59 sea cells" in caption and "6 river channel cells" in caption
    assert "1 isolated sea-pocket cells" in caption
