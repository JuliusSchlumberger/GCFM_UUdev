"""Smoke tests for the channel burn and channel/centreline rasterisation (src.river_burn)."""

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point
from src.river_burn import (
    build_centerline_cells_regular,
    build_channel_mask_regular,
    burn_river_channel,
    constrain_to_coarse_channel_mask,
    mean_elevation_on_coarse_grid,
    snap_points_to_centerline_cells,
)

CRS = "EPSG:32631"
RES_M = 10.0
SHAPE = (30, 30)
TRANSFORM = from_origin(0.0, 300.0, RES_M, RES_M)
Y_RIVER = 145.0  # centre of row 15


def _rivers() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "reach_id": ["1", "2"],
            "rch_id_dn": ["[2]", "[]"],
            "width": [30.0, 40.0],
        },
        geometry=[
            LineString([(25.0, Y_RIVER), (145.0, Y_RIVER)]),
            LineString([(145.0, Y_RIVER), (275.0, Y_RIVER)]),
        ],
        crs=CRS,
    )


def _anchors() -> gpd.GeoDataFrame:
    xs = np.arange(25.0, 280.0, 20.0)
    rid = np.where(xs <= 145.0, "1", "2")
    # One anchor far above the terrain: the burn must not raise the bed there.
    rivbed = 4.0 - 0.01 * xs
    rivbed[3] = 50.0
    return gpd.GeoDataFrame(
        {"reach_id": rid, "rivbed": rivbed},
        geometry=[Point(x, Y_RIVER) for x in xs],
        crs=CRS,
    )


def _write_dem(path) -> np.ndarray:
    rng = np.random.default_rng(0)
    cols = np.arange(SHAPE[1]) * RES_M
    dem = (10.0 - 0.02 * cols)[None, :] + rng.uniform(0.0, 2.0, SHAPE)
    dem = dem.astype(np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SHAPE[0],
        width=SHAPE[1],
        count=1,
        dtype="float32",
        crs=CRS,
        transform=TRANSFORM,
        nodata=-9999.0,
    ) as dst:
        dst.write(dem, 1)
    return dem


def test_burned_channel_never_above_dem_and_only_in_channel_mask(tmp_path):
    dem_path = tmp_path / "dem.tif"
    dem = _write_dem(dem_path)
    mask = build_channel_mask_regular(_rivers(), "width", SHAPE, TRANSFORM)
    burned, transform, terrain, stats = burn_river_channel(
        _rivers(),
        _anchors(),
        dem_path,
        CRS,
        RES_M,
        out_transform=TRANSFORM,
        out_shape=SHAPE,
        channel_mask=mask,
    )
    assert burned.shape == terrain.shape == SHAPE
    assert burned.dtype == np.float32
    assert transform == TRANSFORM
    assert mask.any()
    np.testing.assert_array_equal(np.isfinite(burned), mask)
    assert np.all(burned[mask] <= terrain[mask])
    assert np.all(burned[mask] <= dem[mask] + 1e-3)
    assert stats["n_reaches_burned"] == 2
    # Junction pixels inside both reaches' buffers are counted once per reach.
    assert stats["n_pixels_burned"] >= int(mask.sum())


def test_burn_without_mask_matches_channel_mask_footprint(tmp_path):
    dem_path = tmp_path / "dem.tif"
    _write_dem(dem_path)
    mask = build_channel_mask_regular(_rivers(), "width", SHAPE, TRANSFORM)
    burned, *_ = burn_river_channel(
        _rivers(),
        _anchors(),
        dem_path,
        CRS,
        RES_M,
        out_transform=TRANSFORM,
        out_shape=SHAPE,
    )
    np.testing.assert_array_equal(np.isfinite(burned), mask)


def test_coarse_mean_elevation_bounded_by_block_and_nan_without_burn():
    rng = np.random.default_rng(1)
    terrain = rng.uniform(5.0, 10.0, (6, 6)).astype(np.float32)
    burned = np.full((6, 6), np.nan, dtype=np.float32)
    burned[0:3, 0:3] = 1.0  # fully inside coarse cell (0, 0)
    burned[4, 4] = 2.0  # one pixel of coarse cell (1, 1)
    coarse = mean_elevation_on_coarse_grid(burned, terrain, 3)
    assert coarse.shape == (2, 2) and coarse.dtype == np.float32
    assert coarse[0, 0] == np.float32(1.0)
    block = np.where(np.isfinite(burned[3:, 3:]), burned[3:, 3:], terrain[3:, 3:])
    assert block.min() <= coarse[1, 1] <= block.max()
    assert coarse[1, 1] < terrain[3:, 3:].mean()
    assert np.isnan(coarse[0, 1]) and np.isnan(coarse[1, 0])


def test_constrain_to_coarse_mask_only_removes_pixels():
    burned = np.full((6, 6), 3.0, dtype=np.float32)
    burned[0, 0] = np.nan
    coarse_mask = np.array([[True, False], [False, True]])
    coarse_transform = from_origin(0.0, 60.0, 30.0, 30.0)
    out = constrain_to_coarse_channel_mask(
        burned, from_origin(0.0, 60.0, 10.0, 10.0), CRS, coarse_mask, coarse_transform
    )
    assert out.shape == burned.shape
    assert not np.any(np.isfinite(out) & ~np.isfinite(burned))
    assert np.all(np.isnan(out[0:3, 3:6])) and np.all(np.isnan(out[3:6, 0:3]))
    np.testing.assert_array_equal(out[3:6, 3:6], burned[3:6, 3:6])


def test_centerline_cells_sorted_and_points_snap_to_own_reach():
    cells = build_centerline_cells_regular(_rivers(), SHAPE, TRANSFORM)
    assert set(cells["reach_id"]) == {"1", "2"}
    assert (cells["row"] == 15).all()
    for _, grp in cells.groupby("reach_id"):
        assert grp["along_m"].is_monotonic_increasing
    # Point at the junction: the reach_id hint decides which reach's cell wins.
    pts = gpd.GeoDataFrame(geometry=[Point(146.0, 160.0)] * 2, crs=CRS)
    snapped = snap_points_to_centerline_cells(pts, cells, reach_ids=["1", "2"])
    on_cells = set(zip(cells["row"], cells["col"]))
    assert set(zip(snapped["row"], snapped["col"])) <= on_cells
    assert (snapped["snap_distance_m"] >= 0).all()
    c1 = cells[cells["reach_id"] == "1"]
    assert (snapped["row"].iloc[0], snapped["col"].iloc[0]) in set(
        zip(c1["row"], c1["col"])
    )
