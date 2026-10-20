"""Smoke tests for DEM conditioning along river channels (src.river_conditioning)."""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from shapely.geometry import LineString
from src.river_conditioning import enforce_river_monotonicity

CRS = "EPSG:32631"
RES_M = 10.0
SHAPE = (30, 30)
TRANSFORM = from_origin(0.0, 300.0, RES_M, RES_M)
NODATA = -9999.0
ROW = 15  # river runs along the centre of this row (y = 145 m)


def _rivers() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "reach_id": ["1", "2"],
            "rch_id_dn": ["[2]", "[]"],
            "is_seed": [True, False],
            "width": [20.0, 30.0],
        },
        geometry=[
            LineString([(25.0, 145.0), (145.0, 145.0)]),
            LineString([(145.0, 145.0), (275.0, 145.0)]),
        ],
        crs=CRS,
    )


@pytest.fixture
def conditioned(tmp_path):
    rng = np.random.default_rng(0)
    cols = np.arange(SHAPE[1]) * RES_M
    dem = ((10.0 - 0.02 * cols)[None, :] + rng.uniform(0.0, 3.0, SHAPE)).astype(
        np.float32
    )
    dem[ROW, 20] = NODATA  # on the centreline
    dem[ROW + 1, 8] = NODATA  # inside the channel buffer, off the centreline
    dem[2, 2] = NODATA  # far from the river
    src_path, out_path = tmp_path / "dem.tif", tmp_path / "out" / "conditioned.tif"
    with rasterio.open(
        src_path,
        "w",
        driver="GTiff",
        height=SHAPE[0],
        width=SHAPE[1],
        count=1,
        dtype="float32",
        crs=CRS,
        transform=TRANSFORM,
        nodata=NODATA,
    ) as dst:
        dst.write(dem, 1)
    n_mod, n_checked, z_max = enforce_river_monotonicity(_rivers(), src_path, out_path)
    with rasterio.open(out_path) as src:
        out = src.read(1)
        assert src.nodata == NODATA and src.crs == CRS
    return dem, out, n_mod, n_checked, z_max


def _channel_mask() -> np.ndarray:
    polys = [g.buffer(w / 2.0) for g, w in zip(_rivers().geometry, _rivers()["width"])]
    return geometry_mask(
        polys, out_shape=SHAPE, transform=TRANSFORM, invert=True, all_touched=True
    )


def test_conditioned_dem_non_increasing_downstream(conditioned):
    _, out, n_mod, n_checked, z_max = conditioned
    profile = out[ROW][_channel_mask()[ROW]]
    profile = profile[profile != NODATA]
    assert np.all(np.diff(profile) <= 0)
    assert n_checked > 0 and sum(n_mod.values()) > 0
    assert z_max == pytest.approx(float(profile.max()))


def test_conditioning_changes_only_channel_cells(conditioned):
    dem, out, *_ = conditioned
    assert out.shape == dem.shape and out.dtype == np.float32
    outside = ~_channel_mask()
    np.testing.assert_array_equal(out[outside], dem[outside])


def test_conditioning_preserves_nodata(conditioned):
    dem, out, *_ = conditioned
    np.testing.assert_array_equal(out == NODATA, dem == NODATA)
