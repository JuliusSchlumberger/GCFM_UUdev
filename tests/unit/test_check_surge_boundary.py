"""Smoke tests for analysis/check_surge_boundary.py on a synthetic coast."""

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))

from check_surge_boundary import (  # noqa: E402
    BasinPaths,
    boundary_share,
    check_basin,
    level_anomaly,
    plot_basin,
    two_nearest_weights,
)

CRS = "EPSG:32631"
X0, Y1, DX = 500_000.0, 4_500_000.0, 100.0
NROW, NCOL = 30, 40  # land in the west half, sea in the east, boundary on the east edge
PEAK_M = 1.0


def test_two_nearest_weights_sum_to_one_and_favour_the_closer_point():
    pts = np.array([[0.0, 0.0], [100.0, 0.0], [1000.0, 0.0]])
    idx, w = two_nearest_weights(np.array([[25.0, 0.0]]), pts)
    assert set(idx[0]) == {0, 1}
    assert w.sum() == pytest.approx(1.0)
    assert w[0][list(idx[0]).index(0)] > 0.5


def test_boundary_share_sums_to_one():
    idx, w = two_nearest_weights(
        np.random.default_rng(0).uniform(0, 1000, (50, 2)),
        np.array([[0.0, 0], [500, 0], [1000, 0]]),
    )
    assert boundary_share(idx, w, 3).sum() == pytest.approx(1.0)


def test_level_anomaly_flags_the_outlier_only():
    a = level_anomaly(np.array([0.5, 0.52, 0.48, 1.4]))
    assert a[3] == pytest.approx(0.89)
    assert np.abs(a[:3]).max() < 0.05


def _write_tif(path, arr, transform, dtype, nodata=None):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=dtype,
        crs=CRS,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr.astype(dtype), 1)


@pytest.fixture
def synthetic_basin(tmp_path):
    t = from_origin(X0, Y1, DX, DX)
    mask = np.ones((NROW, NCOL))
    mask[:, -1] = 2
    dep = np.where(np.arange(NCOL)[None, :] >= NCOL // 2, -5.0, 2.0) * np.ones(
        (NROW, 1)
    )
    landuse = np.where(dep < 0, 200, 40).astype("uint8")
    for name, arr, dt in (("mask", mask, "float64"), ("dep", dep, "float32")):
        _write_tif(tmp_path / f"{name}.tif", arr, t, dt)
    _write_tif(tmp_path / "landuse_on_grid.tif", landuse, t, "uint8", 255)

    gpd.GeoDataFrame(
        geometry=[box(X0, Y1 - NROW * DX, X0 + NCOL * DX, Y1)], crs=CRS
    ).to_file(tmp_path / "domain.gpkg")
    (tmp_path / "domain_bbox.json").write_text(json.dumps({"crs": CRS}))

    # station 0 inside the sea, 1 just offshore, 2 far offshore with an outlying level
    st_xy = np.array(
        [
            [X0 + 30.5 * DX, Y1 - 10.5 * DX],
            [X0 + 41 * DX, Y1 - 20 * DX],
            [X0 + 60 * DX, Y1 - 15 * DX],
        ]
    )
    lon, lat = (
        gpd.GeoSeries(gpd.points_from_xy(st_xy[:, 0], st_xy[:, 1]), crs=CRS)
        .to_crs("EPSG:4326")
        .pipe(lambda g: (g.x.values, g.y.values))
    )
    rp = [1.0, 100.0]
    xr.Dataset(
        {
            "storm_tide_rp_table": (
                ("station", "rp_yr"),
                [[0.3, 0.6], [0.3, 0.62], [0.3, 1.5]],
            ),
            "station_calm_sea_level_m": ("station", [0.0, 0.0, 0.0]),
            "calm_sea_level_m": 0.0,
            "mdt": ("station", [0.0, 0.0, 0.0]),
            "slr_fingerprint": ("station", [1.0, 1.0, 1.0]),
            "hg_match_km": ("station", [1.0, 1.0, 1.0]),
        },
        coords={
            "rp_yr": rp,
            "longitude": ("station", lon),
            "latitude": ("station", lat),
        },
    ).to_netcdf(tmp_path / "surge_forcing.nc")

    run = tmp_path / "run"
    run.mkdir()
    np.savetxt(run / "sfincs.bnd", st_xy[:2], fmt="%.1f")
    times = np.arange(0, 7200 * 4, 3600.0)
    bzs = np.column_stack([times, np.full((len(times), 2), 0.2)])
    bzs[2, 1:] = PEAK_M
    np.savetxt(run / "sfincs.bzs", bzs, fmt="%.3f")

    # SFINCS writes rows south to north
    zsmax = np.where(mask > 0, PEAK_M, -99999.0)
    zsmax[10, 30] = PEAK_M + 0.3  # station 0's cell
    zs = np.repeat(np.where(mask > 0, 0.2, -99999.0)[None], len(times), axis=0)
    zs[3, 10, 30] = PEAK_M + 0.3
    yy = Y1 - np.arange(NROW + 1)[::-1] * DX
    xx = X0 + np.arange(NCOL + 1) * DX
    xr.Dataset(
        {
            "msk": (("n", "m"), mask[::-1]),
            "zsmax": (("timemax", "n", "m"), zsmax[None, ::-1]),
            "zs": (("time", "n", "m"), zs[:, ::-1]),
            "corner_x": (
                ("corner_n", "corner_m"),
                np.repeat(xx[None, :], NROW + 1, axis=0),
            ),
            "corner_y": (
                ("corner_n", "corner_m"),
                np.repeat(yy[:, None], NCOL + 1, axis=1),
            ),
        },
        coords={"time": times},
    ).to_netcdf(run / "sfincs_map.nc")
    return BasinPaths(
        surge_forcing=tmp_path / "surge_forcing.nc",
        domain_bbox=tmp_path / "domain_bbox.json",
        domain_gpkg=tmp_path / "domain.gpkg",
        mask_tif=tmp_path / "mask.tif",
        dep_tif=tmp_path / "dep.tif",
        landuse_on_grid=tmp_path / "landuse_on_grid.tif",
        run_dir=run,
    )


def test_check_basin_locates_stations_and_flags_anomaly_and_excess(
    synthetic_basin, tmp_path
):
    res = check_basin(synthetic_basin, "synthetic", 100.0, 0.0, None, 0.10, 0.10, 0.5)
    st = res["stations"]
    assert st["in_active_cell"].tolist() == [True, False, False]
    assert st["dist_wl_boundary_m"].iloc[0] < 0 < st["dist_wl_boundary_m"].iloc[1]
    assert st["flag_anomaly"].tolist() == [False, False, True]
    assert st["used_in_bnd"].tolist() == [True, True, False]
    assert st["boundary_share"].sum() == pytest.approx(1.0)
    assert st["excess_cell_m"].iloc[0] == pytest.approx(0.3, abs=1e-5)
    assert st["peak_lag_hr"].iloc[0] == pytest.approx(1.0)
    assert np.nanmedian(res["excess"]) == pytest.approx(0.0, abs=1e-5)
    assert np.isnan(res["excess"][:, : NCOL // 2]).all()  # land is not sea
    figs = plot_basin(res, "synthetic", "default", 100.0, tmp_path, "test")
    assert all(f.exists() for f in figs) and len(figs) == 2
