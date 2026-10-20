"""Smoke tests for src.postprocessing on synthetic grids (no SFINCS run needed)."""

from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from hydromt_sfincs import utils as sfincs_utils
from shapely.geometry import box
from src import postprocessing
from src.postprocessing import (
    _coarsen_for_memory,
    compute_excess_volume,
    compute_flood_progression,
    compute_flood_timeseries_stats,
    compute_max_inundation,
    compute_risk_metrics,
)

CRS = "EPSG:32631"
X0, Y0 = 500_000.0, 5_700_000.0


def _grid(values, res, name, extra_dims=()):
    values = np.asarray(values, dtype="float32")
    ny, nx = values.shape[-2:]
    x = X0 + res * (np.arange(nx) + 0.5)
    y = Y0 - res * (np.arange(ny) + 0.5)
    dims = (*extra_dims, "y", "x")
    coords = {"x": x, "y": y}
    da = xr.DataArray(values, dims=dims, coords=coords, name=name)
    da.raster.set_crs(CRS)
    da.raster.set_nodata(np.nan)
    return da


@pytest.fixture
def synthetic_run(monkeypatch):
    """Fake 4x4 coarse SFINCS output over an 8x8 bed-level grid; sea in column 0."""
    rng = np.random.default_rng(0)
    zb = _grid(np.linspace(-1.0, 2.0, 16).reshape(4, 4), 100.0, "zb")
    zs = _grid(
        np.stack([zb.values + rng.uniform(-0.5, 1.0, zb.shape) for _ in range(3)]),
        100.0,
        "zs",
        extra_dims=("time",),
    ).assign_coords(time=pd.date_range("2000-01-01", periods=3, freq="h"))
    zsmax = zs.rename({"time": "timemax"}).rename("zsmax")
    dep = _grid(rng.uniform(-1.0, 2.0, (8, 8)), 50.0, "dep")
    sea = np.zeros((4, 4), dtype="float32")
    sea[:, 0] = 1.0
    da_sea = _grid(sea, 100.0, "sea")

    mod = SimpleNamespace(
        output=SimpleNamespace(data={"zs": zs, "zb": zb, "zsmax": zsmax}),
        data_catalog=SimpleNamespace(get_rasterdataset=lambda _p: da_sea),
    )
    monkeypatch.setattr(postprocessing, "load_sfincs_output", lambda _d: mod)
    monkeypatch.setattr(postprocessing, "get_bed_level", lambda *_a, **_k: dep)
    return SimpleNamespace(zs=zs, zsmax=zsmax, dep=dep)


def test_max_depth_is_upper_bound_of_every_timestep(synthetic_run):
    hmax, dep = compute_max_inundation("run", "root", "sea.tif", hmin=0.0)
    assert hmax.shape == dep.shape == synthetic_run.dep.shape
    for t in range(synthetic_run.zsmax.sizes["timemax"]):
        zs_t = synthetic_run.zsmax.isel(timemax=t)
        h_t = sfincs_utils.downscale_floodmap(zs_t, synthetic_run.dep, hmin=0.0)
        both = hmax.notnull().values & h_t.notnull().values
        assert np.all(hmax.values[both] >= h_t.values[both] - 1e-6)


def test_max_depth_non_negative_and_sea_masked(synthetic_run):
    hmax, dep = compute_max_inundation("run", "root", "sea.tif", hmin=0.05)
    assert np.nanmin(hmax.values) >= 0.05
    # sea column 0 of the 100 m grid covers columns 0-1 of the 50 m grid
    assert dep.isel(x=slice(0, 2)).isnull().all()
    assert hmax.isel(x=slice(0, 2)).isnull().all()
    assert dep.isel(x=slice(2, None)).notnull().all()


def test_max_inundation_returns_none_without_zsmax(synthetic_run, monkeypatch):
    mod = SimpleNamespace(output=SimpleNamespace(data={}))
    monkeypatch.setattr(postprocessing, "load_sfincs_output", lambda _d: mod)
    assert compute_max_inundation("run", "root", "sea.tif") == (None, None)


def test_flood_progression_depth_non_negative_and_sea_masked(synthetic_run):
    h = compute_flood_progression("run", "sea.tif")
    assert h.dims == synthetic_run.zs.dims
    assert np.nanmin(h.values) >= 0.0
    assert h.isel(x=0).isnull().all()
    level = compute_flood_progression("run", "sea.tif", variable="level")
    assert level.notnull().all()


def test_flood_progression_rejects_unknown_variable():
    with pytest.raises(ValueError, match="variable"):
        compute_flood_progression("run", "sea.tif", variable="velocity")


def test_timeseries_area_bounded_by_domain_and_volume_non_negative(synthetic_run):
    df = compute_flood_timeseries_stats("run", "root", "sea.tif", threshold_m=0.1)
    assert len(df) == synthetic_run.zs.sizes["time"]
    domain_km2 = synthetic_run.dep.size * 50.0 * 50.0 / 1e6
    assert (df["flooded_area_km2"] >= 0).all()
    assert (df["flooded_area_km2"] <= domain_km2).all()
    assert (df["flood_volume_m3"] >= 0).all()


def test_coarsen_for_memory_is_noop_when_it_fits_and_shrinks_otherwise(synthetic_run):
    dep = synthetic_run.dep
    assert _coarsen_for_memory(dep, max_bytes=1e9) is dep
    coarse = _coarsen_for_memory(dep, max_bytes=dep.nbytes / 4)
    assert coarse.size < dep.size
    assert coarse.rio.crs == dep.rio.crs


def _write(da, path, nodata):
    da = da.copy()
    da.rio.write_crs(CRS, inplace=True)
    da.rio.write_nodata(nodata, inplace=True)
    da.rio.to_raster(path)
    return path


def test_risk_metrics_areas_are_consistent(synthetic_run, tmp_path):
    hmax, dep = compute_max_inundation("run", "root", "sea.tif", hmin=0.05)
    lu = np.full(dep.shape, 10, dtype="int16")
    lu[:4, :] = 50
    lu_path = _write(
        xr.DataArray(lu, dims=("y", "x"), coords=dep.coords),
        tmp_path / "lu.tif",
        -1,
    )
    delta_path = tmp_path / "delta.geojson"
    gpd.GeoDataFrame(geometry=[box(X0, Y0 - 400, X0 + 400, Y0)], crs=CRS).to_file(
        delta_path
    )

    m = compute_risk_metrics(hmax, dep, lu_path, delta_path)
    assert 0 <= m["flooded_area_km2"] <= m["land_area_km2"] <= m["model_domain_km2"]
    assert 0 <= m["urban_exposed_km2"] <= m["urban_area_km2"] <= m["land_area_km2"]
    assert 0 <= m["flood_extent_pct"] <= 100
    assert 0 <= m["mean_depth_m"] <= m["max_depth_m"]
    assert m["volume_m3"] >= 0


def test_excess_volume_grows_with_included_classes(tmp_path):
    rng = np.random.default_rng(1)
    depth = rng.uniform(0.0, 2.0, (6, 6)).astype("float32")
    depth[0, :] = -9999.0
    attr = rng.integers(0, 4, (6, 6)).astype("uint8")
    coords = _grid(depth, 10.0, "d").coords
    fm = _write(
        xr.DataArray(depth, dims=("y", "x"), coords=coords), tmp_path / "f.tif", -9999.0
    )
    am = _write(
        xr.DataArray(attr, dims=("y", "x"), coords=coords), tmp_path / "a.tif", 255
    )

    v_river = compute_excess_volume(fm, am, classes=(1,))
    v_default = compute_excess_volume(fm, am)
    v_all = compute_excess_volume(fm, am, classes=(0, 1, 2, 3))
    assert 0 <= v_river <= v_default <= v_all
    # nodata row is excluded, so the total is bounded by the valid pixels
    assert v_all == pytest.approx(depth[1:].sum() * 100.0, rel=1e-5)
