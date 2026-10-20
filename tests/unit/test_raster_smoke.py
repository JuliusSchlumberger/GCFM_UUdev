"""Smoke tests for the raster helpers in src.raster."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from src.landuse import LANDUSE_SEA_CODE
from src.raster import (
    blend_coast,
    dry_isolated_sea_cells,
    find_fathomdem_tiles,
    merge_tiled_raster,
    open_water_mask,
    reproject_nan_aware,
    restrict_water_level_boundary_to_sea,
    vectorize_land_from_landuse,
    vectorize_land_mask_on_grid,
)

UTM31N = "EPSG:32631"


def _write(path, arr, transform, crs, nodata=None):
    profile = dict(
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return path


def _landuse():
    lu = np.full((20, 20), LANDUSE_SEA_CODE, dtype=np.uint8)
    lu[:, :10] = 10
    lu[0, 0] = 255
    return lu


def test_reproject_nan_aware_preserves_shape_dtype_and_bounds():
    src = np.linspace(0, 10, 400, dtype=np.float32).reshape(20, 20)
    src[:, :5] = np.nan
    t_src = from_origin(500_000, 5_702_000, 100, 100)
    t_dst = from_origin(500_000, 5_702_000, 200, 200)
    out = reproject_nan_aware(src, t_src, UTM31N, (10, 10), t_dst, UTM31N)
    assert out.shape == (10, 10) and out.dtype == np.float32
    finite = out[np.isfinite(out)]
    assert finite.min() >= np.nanmin(src) - 1e-4
    assert finite.max() <= np.nanmax(src) + 1e-4
    assert np.isnan(out[:, :2]).all()
    assert np.isfinite(out[:, 3:]).all()


def test_reproject_nan_aware_constant_field_stays_constant_near_nan_edge():
    src = np.full((20, 20), 3.0, dtype=np.float32)
    src[8:12, 8:12] = np.nan
    t_src = from_origin(0, 2000, 100, 100)
    t_dst = from_origin(0, 2000, 50, 50)
    out = reproject_nan_aware(src, t_src, UTM31N, (40, 40), t_dst, UTM31N)
    assert np.isfinite(out).sum() > 0
    assert np.allclose(out[np.isfinite(out)], 3.0, atol=1e-5)


def test_find_fathomdem_tiles_returns_only_existing_tiles(tmp_path):
    for name in ["N51E003.tif", "S01W090.tif"]:
        (tmp_path / name).touch()
    tiles = find_fathomdem_tiles(tmp_path, (3.2, 51.2, 4.8, 52.5))
    assert [Path(p).name for p in tiles] == ["N51E003.tif"]
    tiles_sw = find_fathomdem_tiles(tmp_path, (-89.5, -0.5, -89.2, -0.2))
    assert [Path(p).name for p in tiles_sw] == ["S01W090.tif"]


def test_merge_tiled_raster_keeps_pixel_lattice_dtype_and_nodata(tmp_path):
    a = np.full((10, 10), 1.0, dtype=np.float32)
    b = np.full((10, 10), 2.0, dtype=np.float32)
    ta = _write(
        tmp_path / "a.tif", a, from_origin(0.0, 1.0, 0.1, 0.1), "EPSG:4326", -9999.0
    )
    tb = _write(
        tmp_path / "b.tif", b, from_origin(1.0, 1.0, 0.1, 0.1), "EPSG:4326", -9999.0
    )
    out = tmp_path / "sub" / "merged.tif"
    merge_tiled_raster([str(ta), str(tb)], (0.25, 0.25, 1.75, 0.95), out)
    with rasterio.open(out) as ds:
        arr = ds.read(1)
        assert ds.nodata == -9999.0 and ds.dtypes[0] == "float32"
        assert ds.res == pytest.approx((0.1, 0.1))
        origin = (ds.transform.c, ds.transform.f)
    assert [v / 0.1 for v in origin] == pytest.approx([round(v / 0.1) for v in origin])
    assert set(np.unique(arr)) == {1.0, 2.0}


def test_merge_tiled_raster_without_tiles_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        merge_tiled_raster([], (0, 0, 1, 1), tmp_path / "x.tif")


def test_vectorize_land_from_landuse_excludes_sea_and_nodata(tmp_path):
    t = from_origin(3.0, 52.0, 0.01, 0.01)
    path = _write(tmp_path / "lu.tif", _landuse(), t, "EPSG:4326", 255)
    gdf = vectorize_land_from_landuse(path, (3.0, 51.8, 3.2, 52.0))
    assert gdf.crs == "EPSG:4326"
    minx, _, maxx, _ = gdf.total_bounds
    assert minx == pytest.approx(3.0) and maxx == pytest.approx(3.1)
    assert gdf.area.sum() == pytest.approx(199 * 0.01**2)


def test_vectorize_land_mask_on_grid_returns_wgs84_land(tmp_path):
    t = from_origin(500_000, 5_702_000, 100, 100)
    path = _write(tmp_path / "lu_grid.tif", _landuse(), t, UTM31N, 255)
    gdf = vectorize_land_mask_on_grid(path)
    assert gdf.crs == "EPSG:4326"
    assert gdf.to_crs(UTM31N).area.sum() == pytest.approx(199 * 100.0**2, rel=1e-6)


def test_restrict_waterlevel_boundary_moves_land_cells_only():
    mask = np.ones((4, 4), dtype=np.uint8)
    mask[0, :] = 2
    lu = np.full((4, 4), LANDUSE_SEA_CODE, dtype=np.uint8)
    lu[0, :2] = 10
    out, n_moved = restrict_water_level_boundary_to_sea(mask, lu)
    assert n_moved == 2
    assert out[0].tolist() == [1, 1, 2, 2]
    assert np.array_equal(out[1:], mask[1:])
    assert mask[0, 0] == 2


def test_restrict_waterlevel_boundary_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        restrict_water_level_boundary_to_sea(np.ones((3, 3)), np.ones((4, 4)))


def test_open_water_mask_drops_narrow_creek_keeps_open_sea():
    water = np.zeros((20, 20), dtype=bool)
    water[:, 12:] = True
    water[10, :12] = True
    out = open_water_mask(water, pixel_size_m=10.0, min_half_width_m=20.0)
    assert out.dtype == bool and out.shape == water.shape
    assert not (out & ~water).any()
    assert out[:, 14:].all()
    assert not out[10, :8].any()


def test_blend_coast_leaves_land_unchanged_and_deepens_seaward():
    land = np.zeros((20, 20), dtype=bool)
    land[:, :8] = True
    merged = np.where(land, 5.0, -10.0)
    sea_bed = np.full((20, 20), -10.0)
    blended, n_blended = blend_coast(
        merged, land, sea_bed, pixel_size_m=10.0, blend_m=50.0
    )
    assert n_blended > 0
    assert np.array_equal(blended[land], merged[land])
    assert blended.min() >= -10.0 and blended.max() <= 5.0
    assert (np.diff(blended[:, 8:], axis=1) <= 0).all()


def test_blend_coast_zero_band_returns_input_unchanged():
    merged = np.zeros((5, 5))
    land = np.zeros((5, 5), dtype=bool)
    land[0] = True
    out, n_blended = blend_coast(merged, land, merged, pixel_size_m=1.0, blend_m=0.0)
    assert n_blended == 0 and out is merged


def test_dry_isolated_sea_cells_dries_only_pockets_below_baseline():
    calm_sea_level_m = 0.3
    dep = np.full((20, 20), -1.0, dtype=np.float32)
    sea = np.zeros((20, 20), bool)
    sea[:, :8] = True  # open ocean, largest component
    sea[2:4, 12:14] = True  # pocket below baseline
    sea[15:17, 15:17] = True  # pocket above baseline
    dep[15:17, 15:17] = 1.0
    sea[10, 18] = True  # pocket cell with nodata bed
    dep[10, 18] = -9999.0
    zsini = np.where(sea, np.float32(calm_sea_level_m), np.nan)

    out, n_dried = dry_isolated_sea_cells(zsini, dep, -9999.0, calm_sea_level_m)

    assert n_dried == 4
    np.testing.assert_array_equal(out[2:4, 12:14], dep[2:4, 12:14])
    assert np.allclose(out[:, :8], calm_sea_level_m)
    assert np.allclose(out[15:17, 15:17], calm_sea_level_m)
    assert np.isclose(out[10, 18], calm_sea_level_m)
    assert np.isnan(out[~sea]).all()


def test_dry_isolated_sea_cells_without_sea_is_noop():
    zsini = np.full((5, 5), np.nan)
    out, n_dried = dry_isolated_sea_cells(zsini, np.zeros((5, 5)), -9999.0, 0.0)
    assert n_dried == 0
    assert np.isnan(out).all()
