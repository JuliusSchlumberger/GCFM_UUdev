"""Smoke tests for the grid-traced coastal protection weir in src.protection_weir."""

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import LineString
from src.protection_weir import (
    LANDUSE_SEA,
    GridArrays,
    build_coastal_protection_weir,
    discard_small_components,
    merge_weir_preserve_unmatched,
    ocean_linked_wetland_mask,
)

N = 20
DX = 100.0
CRS = "EPSG:32631"
LAND = 40


def _grid(n=N):
    return GridArrays(
        CRS,
        bed_elevation=np.zeros((n, n), dtype=np.float32),
        valid_mask=np.ones((n, n), dtype=bool),
        cell_size_m=DX,
        transform=Affine(DX, 0, 0, 0, -DX, n * DX),
    )


def _half_sea(n=N):
    landuse = np.full((n, n), LAND, dtype=np.uint8)
    landuse[:, : n // 2] = LANDUSE_SEA
    return landuse


def test_small_land_island_is_discarded_large_kept():
    mask = np.zeros((N, N), dtype=bool)
    mask[:, 10:] = True
    mask[2, 2] = True
    out, n_cells, n_comp = discard_small_components(mask, _grid(), min_cells=5)
    assert (n_cells, n_comp) == (1, 1)
    assert not out[2, 2]
    assert out[:, 10:].all()


def test_flat_weir_follows_coastline_at_ceiled_crest():
    gdf, diag = build_coastal_protection_weir(
        _grid(),
        _half_sea(),
        river_channel_mask=np.zeros((N, N), dtype=bool),
        crest_elevation_m=2.341,
        min_component_cells=3,
        weir_par1=0.6,
    )
    assert diag["applicable"]
    assert len(gdf) > 0
    assert gdf.crs == CRS
    assert np.all(gdf["elevation"] >= 2.341)
    assert np.allclose(gdf["elevation"], 2.35)
    assert (gdf["par1"] == 0.6).all()
    coast_x = (N // 2) * DX
    for geom in gdf.geometry:
        xs = [c[0] for c in geom.coords]
        assert np.allclose(xs, coast_x)
    assert diag["land_mask"][:, N // 2 :].all()
    assert not diag["land_mask"][:, : N // 2].any()


def test_water_side_crest_never_below_floor():
    landuse = _half_sea()
    channel = np.zeros((N, N), dtype=bool)
    channel[9:11, 10:18] = True
    water_level = np.full((N, N), np.nan)
    water_level[9:11, 10:18] = 3.0
    water_level[:, :10] = 0.5
    gdf, diag = build_coastal_protection_weir(
        _grid(),
        landuse,
        channel,
        crest_elevation_m=1.0,
        min_component_cells=3,
        weir_par1=0.6,
        water_side_crest_on_grid=water_level,
        channel_mask_gap_free=True,
    )
    elev = gdf["elevation"].to_numpy(dtype=float)
    assert len(gdf) > 0
    assert np.all(np.isfinite(elev))
    assert np.all(elev >= 1.0)
    assert np.isclose(elev.max(), 3.0)
    assert np.all(diag["crest_surface"] >= 1.0)


def test_inland_basin_gets_no_weir():
    gdf, diag = build_coastal_protection_weir(
        _grid(),
        np.full((N, N), LAND, dtype=np.uint8),
        np.zeros((N, N), dtype=bool),
        crest_elevation_m=2.0,
        min_component_cells=3,
        weir_par1=0.6,
    )
    assert gdf.empty
    assert diag == {"applicable": False}


def test_wetland_linked_only_by_edge_to_sea():
    landuse = _half_sea()
    landuse[5, 10] = 90  # shares an edge with sea column 9
    landuse[15, 12] = 80  # isolated, inland
    linked = ocean_linked_wetland_mask(landuse, landuse == LANDUSE_SEA, _grid())
    assert linked[5, 10]
    assert not linked[15, 12]
    assert linked.sum() == 1


def test_merge_replaces_matched_baseline_and_keeps_river():
    coast = LineString([(0, 0), (100, 0)])
    river = LineString([(500, 500), (500, 600)])
    baseline = gpd.GeoDataFrame(
        {"elevation": [1.0, 2.0], "par1": [0.6, 0.6]}, geometry=[coast, river], crs=CRS
    )
    new = gpd.GeoDataFrame(
        {"elevation": [5.0, 5.0], "par1": [0.6, 0.6]},
        geometry=[
            LineString([(0.05, 0.08), (100.05, 0.08)]),
            LineString([(100, 0), (100, 100)]),
        ],
        crs=CRS,
    )
    merged = merge_weir_preserve_unmatched(baseline, new)
    assert len(merged) == 3
    assert merged.crs == CRS
    assert sorted(merged["elevation"].tolist()) == [2.0, 5.0, 5.0]
    assert any(g.equals(river) for g in merged.geometry)
