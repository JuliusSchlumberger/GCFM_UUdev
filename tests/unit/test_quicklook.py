"""Smoke tests for the quick-look figure renderer in src.quicklook."""

from pathlib import Path

import geopandas as gpd
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import LineString, Point, box
from src.quicklook import STAGES, render

FIGURES = Path(__file__).resolve().parents[2] / ".baseline" / "figures"
CRS = "EPSG:32631"
N = 40
DX = 100.0


@pytest.fixture(scope="module")
def delta():
    """Synthetic delta: sea in the west, land rising east, a river along the middle row."""
    x = np.arange(N) * DX + DX / 2
    y = np.arange(N) * DX + DX / 2
    xx, yy = np.meshgrid(x, y)
    elevation = -6.0 + 14.0 * xx / xx.max()
    river_row = np.abs(yy - y[N // 2]) < 1.5 * DX
    land = elevation > 0
    elevation_burned = np.where(river_row & land, elevation - 4.0, elevation)
    elevation_conditioned = np.where(
        river_row & land, np.minimum.accumulate(elevation, axis=1) - 0.5, elevation
    )
    landuse = np.where(land, 40, 80).astype(float)
    landuse[(xx > 2500) & (xx < 3200) & (yy > 2600) & (yy < 3400)] = 50
    manning = np.select([landuse == 80, landuse == 50], [0.02, 0.08], 0.04)
    max_depth = np.clip(1.8 - elevation, 0.0, None) * land
    max_depth_adapted = np.where(xx < 2000, 0.0, max_depth)
    attribution = np.where(max_depth > 0, np.where(xx < 2000, 1, 2), 0).astype(float)

    def da(values):
        return xr.DataArray(values, coords={"y": y, "x": x}, dims=("y", "x"))

    coast_x = float(x[np.argmax(elevation[0] > 0)])
    river = gpd.GeoDataFrame(
        {"width": [900.0, 400.0, 250.0], "depth_m": [12.0, 7.0, 4.0]},
        geometry=[
            LineString([(4000, 2000), (3000, 2000)]),
            LineString([(3000, 2000), (coast_x, 2000)]),
            LineString([(3000, 2000), (coast_x, 3000)]),
        ],
        crs=CRS,
    )
    weir = gpd.GeoDataFrame(
        {"crest_m": [2.5, 3.5]},
        geometry=[
            LineString([(coast_x, 0), (coast_x, 1900)]),
            LineString([(coast_x, 2100), (coast_x, 4000)]),
        ],
        crs=CRS,
    )
    domain = gpd.GeoDataFrame(geometry=[box(0, 0, N * DX, N * DX)], crs=CRS)
    land_poly = gpd.GeoDataFrame(geometry=[box(coast_x, 0, N * DX, N * DX)], crs=CRS)
    outflow = gpd.GeoDataFrame(
        geometry=[Point(coast_x, 2000), Point(coast_x, 3000)], crs=CRS
    )

    t = pd.date_range("2020-01-01", periods=168, freq="h")
    hours = np.arange(len(t))
    tide = 0.8 * np.sin(2 * np.pi * hours / 12.42)
    surge = 2.5 * np.exp(-(((hours - 84) / 10.0) ** 2))
    discharge = 900 + 3000 * np.exp(-(((hours - 96) / 30.0) ** 2))
    flooded_km2 = np.clip(np.cumsum(np.maximum(surge - 1.0, 0)) * 2.0, 0, None)

    return {
        "01_separate_deltas": {"delta_polygon": land_poly},
        "02_model_domain": {"domain": domain},
        "02b_prepare_landuse": {"landuse_source": da(landuse)},
        "03_land_polygons": {"land_polygons": land_poly},
        "05a_elevation": {"elevation_m": da(elevation)},
        "05b_landuse": {"landuse": da(landuse), "sea_mask": da((~land).astype(float))},
        "05c_roughness": {"manning": da(manning)},
        "06_river_network": {"river_network": river},
        "07_boundary_forcings": {
            "river_discharge": pd.Series(discharge, index=t),
            "surge_water_level": pd.DataFrame(
                {"tide": tide, "tide + surge": tide + surge}, index=t
            ),
        },
        "07c_storm_tide_hydrographs": {"storm_tide": pd.Series(tide + surge, index=t)},
        "08_clean_river_network": {
            "river_network_clean": river,
            "outflow_points": outflow,
        },
        "09_condition_elevation": {
            "elevation_conditioned_m": da(elevation_conditioned)
        },
        "09b_grid_align_landuse": {
            "landuse_on_grid": da(landuse),
            "roughness_on_grid": da(manning),
        },
        "10_depth_estimation": {
            "river_burned_dem_m": da(elevation_burned),
            "river_depth_m": river,
            "weir_crest_m": weir,
        },
        "13_build_sfincs_skeleton": {"zsini_m": da(np.where(land, np.nan, 0.0))},
        "14_run_spinup": {"max_depth_m": da(max_depth * 0.2)},
        "16_run_event": {
            "max_depth_m": da(max_depth),
            "flooded_area_km2": pd.Series(flooded_km2, index=t),
        },
        "18_adaptation": {"max_flood_depth_m": da(max_depth_adapted)},
        "18c_attribution_mask": {"attribution_mask": da(attribution)},
    }


def test_every_stage_has_a_synthetic_case(delta):
    assert set(delta) == set(STAGES)


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_render_writes_one_panel_per_variable(delta, stage):
    path = render(stage, delta[stage], scenario="synthetic", out_dir=FIGURES)
    assert path == FIGURES / f"{stage}_synthetic.png"
    height, width = mpimg.imread(path).shape[:2]
    n_panels = len(STAGES[stage])
    assert width == pytest.approx(4.2 * n_panels * 80, abs=2)
    assert height == pytest.approx(3.8 * 80, abs=2)


def test_render_rejects_missing_variable(delta, tmp_path):
    with pytest.raises(KeyError, match="missing variables"):
        render(
            "05b_landuse",
            {"landuse": delta["05b_landuse"]["landuse"]},
            "synthetic",
            tmp_path,
            commit="x",
        )


def test_render_rejects_unknown_stage(tmp_path):
    with pytest.raises(KeyError, match="unknown stage"):
        render("99_nope", {}, "synthetic", tmp_path, commit="x")


def test_raster_panel_requires_2d_data(tmp_path):
    with pytest.raises(ValueError, match="2D"):
        render(
            "05a_elevation",
            {"elevation_m": np.zeros(5)},
            "synthetic",
            tmp_path,
            commit="x",
        )
