"""Smoke tests for land-use source preparation, warping and roughness in src.landuse."""

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from shapely.geometry import box
from src.landuse import (
    LANDUSE_NODATA,
    LANDUSE_SEA_CODE,
    ROUGHNESS_NODATA,
    aggregate_manning,
    landuse_to_manning,
    read_roughness_lookup,
    warp_landuse_to_grid,
    write_channel_landuse,
    write_landuse_source,
    write_roughness_raster,
)

SRC_CODES = (40, 50, 80, LANDUSE_SEA_CODE)
SRC_NODATA = 0
N_PX = 20
RES_DEG = 0.001
LON0, LAT0 = 4.0, 52.0
MANNING = {40: 0.035, 50: 0.10, 80: 0.02, LANDUSE_SEA_CODE: 0.02}


def _lc100(tmp_path, nodata_corner=True):
    rng = np.random.default_rng(0)
    codes = rng.choice(SRC_CODES, size=(N_PX, N_PX)).astype(np.uint8)
    if nodata_corner:
        codes[:3, :3] = SRC_NODATA
    path = tmp_path / "lc100.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        dtype="uint8",
        count=1,
        height=N_PX,
        width=N_PX,
        crs="EPSG:4326",
        transform=from_origin(LON0, LAT0, RES_DEG, RES_DEG),
        nodata=SRC_NODATA,
    ) as dst:
        dst.write(codes, 1)
    return path, codes


def _bounds():
    return (LON0, LAT0 - N_PX * RES_DEG, LON0 + N_PX * RES_DEG, LAT0)


def _utm_ref_meta(n=6, res_m=150.0):
    lon_min, lat_min, lon_max, lat_max = _bounds()
    x0, y0, x1, y1 = transform_bounds(
        "EPSG:4326", "EPSG:32631", lon_min, lat_min, lon_max, lat_max
    )
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = 0.5 * n * res_m
    return {
        "height": n,
        "width": n,
        "crs": "EPSG:32631",
        "transform": from_origin(cx - half, cy + half, res_m, res_m),
    }


def _lookup_csv(tmp_path):
    path = tmp_path / "lookup.csv"
    rows = ["copernicus_worldcover,manning_n"]
    rows += [f"{code},{n}" for code, n in MANNING.items()]
    rows.append("255,-999")
    path.write_text("\n".join(rows) + "\n")
    return path


def test_lc100_source_keeps_codes_and_maps_nodata(tmp_path):
    lc100, codes = _lc100(tmp_path)
    out = tmp_path / "landuse_source.tif"
    meta, stats = write_landuse_source("copernicus_lc100", _bounds(), lc100, out)
    with rasterio.open(out) as src:
        arr = src.read(1)
        assert src.crs.to_epsg() == 4326
        assert src.nodata == LANDUSE_NODATA
        assert src.tags()["sea_code"] == str(LANDUSE_SEA_CODE)
    assert arr.dtype == np.uint8
    assert arr.shape == codes.shape
    assert set(np.unique(arr)) <= set(SRC_CODES) | {LANDUSE_NODATA}
    np.testing.assert_array_equal(arr == LANDUSE_NODATA, codes == SRC_NODATA)
    np.testing.assert_array_equal(arr[codes != SRC_NODATA], codes[codes != SRC_NODATA])
    assert stats["sea_px"] == int((codes == LANDUSE_SEA_CODE).sum())
    assert meta["nodata"] == LANDUSE_NODATA


def test_unknown_landuse_source_is_rejected(tmp_path):
    lc100, _ = _lc100(tmp_path)
    with pytest.raises(ValueError, match="landuse.source"):
        write_landuse_source("corine", _bounds(), lc100, tmp_path / "x.tif")


def test_warp_to_grid_by_mode_yields_only_source_classes(tmp_path):
    lc100, _ = _lc100(tmp_path, nodata_corner=False)
    source = tmp_path / "landuse_source.tif"
    write_landuse_source("copernicus_lc100", _bounds(), lc100, source)
    ref = _utm_ref_meta()
    out = tmp_path / "landuse_grid.tif"
    histogram = warp_landuse_to_grid(source, ref, out)
    with rasterio.open(out) as src:
        arr = src.read(1)
        assert src.crs.to_epsg() == 32631
        assert src.transform == ref["transform"]
    assert arr.shape == (ref["height"], ref["width"])
    assert arr.dtype == np.uint8
    assert set(np.unique(arr)) <= set(SRC_CODES) | {LANDUSE_NODATA}
    assert sum(histogram.values()) == arr.size


def test_manning_lookup_marks_unmapped_and_nodata_as_nan(tmp_path):
    lu_to_n = read_roughness_lookup(_lookup_csv(tmp_path))
    assert lu_to_n[50] == pytest.approx(0.10)
    landuse = np.array([[40, 50], [LANDUSE_NODATA, 77]], dtype=np.uint8)
    n, unmapped = landuse_to_manning(landuse, lu_to_n)
    assert n.dtype == np.float32
    assert unmapped == {77}
    assert np.isnan(n[1, 0]) and np.isnan(n[1, 1])
    assert n[0, 0] == pytest.approx(0.035) and n[0, 1] == pytest.approx(0.10)


def test_conveyance_mean_does_not_exceed_area_mean():
    rng = np.random.default_rng(1)
    n_fine = rng.choice([0.02, 0.05, 0.1], size=(20, 20)).astype(np.float32)
    n_fine[0, :5] = np.nan
    src_transform = Affine(10.0, 0, 0, 0, -10.0, 200.0)
    dst_transform = Affine(50.0, 0, 0, 0, -50.0, 200.0)
    kwargs = dict(
        src_transform=src_transform,
        src_crs="EPSG:32631",
        dst_shape=(4, 4),
        dst_transform=dst_transform,
        dst_crs="EPSG:32631",
    )
    mean = aggregate_manning(n_fine, method="area_mean", **kwargs)
    conv = aggregate_manning(n_fine, method="conveyance", **kwargs)
    assert mean.shape == conv.shape == (4, 4)
    assert np.all(np.isfinite(mean))
    assert np.all(conv <= mean + 1e-6)
    assert np.all((mean >= 0.02 - 1e-6) & (mean <= 0.1 + 1e-6))


def test_roughness_raster_stays_within_lookup_bounds(tmp_path):
    lc100, _ = _lc100(tmp_path)
    source = tmp_path / "landuse_source.tif"
    write_landuse_source("copernicus_lc100", _bounds(), lc100, source)
    ref = _utm_ref_meta()
    out = tmp_path / "manning.tif"
    unmapped, info = write_roughness_raster(source, _lookup_csv(tmp_path), ref, out)
    with rasterio.open(out) as src:
        n = src.read(1)
        assert src.nodata == ROUGHNESS_NODATA
        assert src.crs.to_epsg() == 32631
    assert unmapped == set()
    assert n.dtype == np.float32
    assert n.shape == info["shape"] == (ref["height"], ref["width"])
    valid = n != ROUGHNESS_NODATA
    assert valid.any()
    lo, hi = min(MANNING.values()), max(MANNING.values())
    assert np.all((n[valid] >= lo - 1e-6) & (n[valid] <= hi + 1e-6))


def _ref_box(ref, fraction_x=1.0, margin_m=500.0):
    """Box over the left fraction_x of a reference grid, past its outer edges (its CRS)."""
    t = ref["transform"]
    x0, y1 = t.c, t.f
    x1 = x0 + ref["width"] * t.a * fraction_x
    y0 = y1 + ref["height"] * t.e
    return box(x0 - margin_m, y0 - margin_m, x1, y1 + margin_m)


def test_channel_roughness_is_channel_n_inside_and_lookup_outside(tmp_path):
    lc100, _ = _lc100(tmp_path, nodata_corner=False)
    source = tmp_path / "landuse_source.tif"
    write_landuse_source("copernicus_lc100", _bounds(), lc100, source)
    ref = _utm_ref_meta()
    plain, channel = tmp_path / "plain.tif", tmp_path / "channel.tif"
    write_roughness_raster(source, _lookup_csv(tmp_path), ref, plain)
    write_roughness_raster(
        source,
        _lookup_csv(tmp_path),
        ref,
        channel,
        channel=([_ref_box(ref, 0.5)], ref["crs"], 0.0333),
    )
    with rasterio.open(plain) as a, rasterio.open(channel) as b:
        n_plain, n_channel = a.read(1), b.read(1)
    left, right = n_channel[:, :2], n_channel[:, 4:]
    assert np.allclose(left, 0.0333, atol=1e-6)  # fully inside the channel
    np.testing.assert_array_equal(right, n_plain[:, 4:])  # fully outside


def test_channel_landuse_turns_land_into_water_and_keeps_sea(tmp_path):
    ref = _utm_ref_meta(n=10, res_m=30.0)
    lu = np.full((10, 10), 50, dtype=np.uint8)  # built-up
    lu[:, 7:] = LANDUSE_SEA_CODE
    lu[0, 0] = 40
    path, out = tmp_path / "lu.tif", tmp_path / "lu_channels.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        dtype="uint8",
        count=1,
        height=10,
        width=10,
        crs=ref["crs"],
        transform=ref["transform"],
        nodata=LANDUSE_NODATA,
    ) as dst:
        dst.write(lu, 1)
    t = ref["transform"]
    channel = box(t.c, t.f - 4 * 30.0, t.c + 10 * 30.0, t.f)  # rows 0-3
    converted = write_channel_landuse(path, [channel], ref["crs"], 80, out)
    with rasterio.open(out) as src:
        after = src.read(1)
    assert np.all(after[:4, :7] == 80)  # land in the channel is water
    assert np.all(after[:, 7:] == LANDUSE_SEA_CODE)  # sea stays sea
    np.testing.assert_array_equal(after[4:], lu[4:])  # outside untouched
    assert converted == {40: 1, 50: 4 * 7 - 1}


def test_border_water_body_above_sea_max_elevation_is_not_sea():
    from src.landuse import _derive_sea_mask

    water = np.zeros((10, 10), dtype=bool)
    water[0:3, :] = True  # coastal sea along the top edge
    water[7:, 4:6] = True  # river reaching the bottom edge
    lc100_sea = np.zeros_like(water)
    lc100_sea[0, 0] = lc100_sea[9, 4] = True  # LC100 also calls the riverbed sea
    elevation_m = np.where(np.arange(10)[:, None] >= 7, 35.0, 0.2) * np.ones((10, 10))
    no = np.zeros_like(water)

    sea_all, _ = _derive_sea_mask(water, no, lc100_sea, no)
    sea, stats = _derive_sea_mask(water, no, lc100_sea, no, elevation_m, 5.0)

    assert sea_all[8, 4] and not sea[8, 4]
    assert sea[1, 5] and sea[0:3].all()
    assert stats["bodies_above_sea_max_elevation"] == 1
