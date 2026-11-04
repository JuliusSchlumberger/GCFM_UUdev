"""Smoke tests for src.adaptation_method_post parameter validation."""

import numpy as np
import pytest
from src.adaptation_method_post import (
    _unpumped_river_volume_m3,
    validate_measure_params,
)

METHOD = "postprocessing"
MEASURE_DEF = {
    "params": {
        "elevation": {"min": 0.0, "max": 10.0, "required_for": [METHOD]},
        "width_m": {"min": 1.0, "max": 500.0},
        "locations": {"required_for": [METHOD]},
    }
}


def test_valid_params_pass():
    params = {"elevation": 3.0, "width_m": 50, "locations": "x.geojson"}
    validate_measure_params("coastal_levee", params, METHOD, MEASURE_DEF)


def test_bounds_are_inclusive():
    for elevation in (0.0, 10.0):
        params = {"elevation": elevation, "locations": "x.geojson"}
        validate_measure_params("coastal_levee", params, METHOD, MEASURE_DEF)


@pytest.mark.parametrize("value", [-0.1, 10.5])
def test_out_of_bounds_param_is_rejected(value):
    params = {"elevation": value, "locations": "x.geojson"}
    with pytest.raises(ValueError, match="'elevation'.*out of bounds"):
        validate_measure_params("coastal_levee", params, METHOD, MEASURE_DEF)


def test_missing_required_param_is_rejected():
    with pytest.raises(ValueError, match="'locations' is required"):
        validate_measure_params(
            "coastal_levee", {"elevation": 2.0}, METHOD, MEASURE_DEF
        )


def test_required_only_for_its_own_method():
    validate_measure_params("coastal_levee", {}, "other_method", MEASURE_DEF)


def test_resolved_current_protection_dict_skips_bounds_check():
    params = {"elevation": {"current": True, "fill_crest": 99.0}, "locations": "x"}
    validate_measure_params("coastal_levee", params, METHOD, MEASURE_DEF)


@pytest.mark.parametrize("volume", [0.0, 5.0, 40.0, 1e6])
def test_uniform_layer_removes_requested_volume_capped_by_available_water(volume):
    from src.adaptation_method_post import _uniform_layer_depth

    depths = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
    cell_area = 10.0
    d = _uniform_layer_depth(depths, volume, cell_area)
    removed = np.minimum(depths, d).sum() * cell_area
    assert 0.0 <= d <= depths.max()
    assert removed == pytest.approx(min(volume, depths.sum() * cell_area))


def test_uniform_layer_depth_is_monotonic_in_volume():
    from src.adaptation_method_post import _uniform_layer_depth

    depths = np.array([0.2, 0.4, 1.5])
    ds = [_uniform_layer_depth(depths, v, 1.0) for v in np.linspace(0, 3, 13)]
    assert np.all(np.diff(ds) >= 0)


def test_unpumped_volume_is_total_discharge_above_capacity(tmp_path):
    (tmp_path / "sfincs.inp").write_text(
        "tref = 20200101 000000\ntstart = 20200101 000000\ntstop = 20200101 020000\n"
    )
    # two points, total 100 -> 300 m3/s linearly over 2 h
    np.savetxt(tmp_path / "sfincs.dis", [[0, 50.0, 50.0], [7200, 150.0, 150.0]])
    vols = [
        _unpumped_river_volume_m3(str(tmp_path), q) for q in (0.0, 100.0, 200.0, 300.0)
    ]
    assert vols[0] == pytest.approx(
        200.0 * 7200, rel=0.01
    )  # mean total discharge x 2 h
    assert vols[2] == pytest.approx(0.5 * 100.0 * 3600, rel=0.02)  # triangle above 200
    assert vols[3] == 0.0
    assert np.all(np.diff(vols) <= 0)
    assert _unpumped_river_volume_m3(str(tmp_path / "none"), 10.0) == 0.0


def test_lagoon_storage_is_coastline_length_times_width_times_depth(tmp_path):
    import geopandas as gpd
    from shapely.geometry import LineString
    from src.adaptation_method_post import _lagoon_storage_m3

    path = tmp_path / "coast.geojson"
    gpd.GeoDataFrame(
        geometry=[
            LineString([(0, 0), (1000, 0)]),
            LineString([(0, 50), (0, 550)]),
            None,
        ],
        crs="EPSG:32631",
    ).to_file(path)
    # 1500 m coastline (the empty feature skipped) x 2000 m x 0.1 m
    assert _lagoon_storage_m3(str(path), 2000.0, 0.1) == pytest.approx(300_000.0)
    assert _lagoon_storage_m3(str(path), 2000.0, 0.0) == 0.0


def test_lagoon_storage_removes_its_volume_from_river_coastal_and_compound(tmp_path):
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import LineString
    from src.adaptation_method_post import apply_lagoon_storage

    nodata = -9999.0
    flood = np.array([[0.5, 1.0, 2.0, nodata, 0.3]], dtype="float32")
    attr = np.array([[1, 2, 3, 2, 4]], dtype="uint8")  # 4: embanked water, kept
    prof = dict(
        driver="GTiff",
        height=1,
        width=5,
        count=1,
        crs="EPSG:32631",
        transform=from_origin(0, 10, 10, 10),
    )  # 100 m2 cells
    with rasterio.open(
        tmp_path / "flood.tif", "w", dtype="float32", nodata=nodata, **prof
    ) as dst:
        dst.write(flood, 1)
    with rasterio.open(
        tmp_path / "attribution_mask.tif", "w", dtype="uint8", **prof
    ) as dst:
        dst.write(attr, 1)
    coast = tmp_path / "coast.geojson"
    gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (100, 0)])], crs="EPSG:32631"
    ).to_file(coast)

    # 100 m coastline x 10 m x 0.1 m = 100 m3 -> a 1/3 m layer off the 3 cells
    out = apply_lagoon_storage(
        str(tmp_path / "flood.tif"),
        str(tmp_path / "sfincs"),
        str(tmp_path / "out"),
        str(coast),
        10.0,
        0.1,
    )
    with rasterio.open(out["out_raster"]) as src:
        new = src.read(1)
    np.testing.assert_allclose(new[0, :3], flood[0, :3] - 1 / 3, rtol=1e-5)
    assert new[0, 3] == nodata and new[0, 4] == pytest.approx(0.3)
    assert (flood[0, :3] - new[0, :3]).sum() * 100 == pytest.approx(100.0, rel=1e-5)


def test_bool_param_rejects_non_bool_value():
    spec = {
        "params": {
            "merge": {"min": None, "max": None, "type": "bool", "required_for": []}
        }
    }
    validate_measure_params("coastal_levee", {"merge": False}, METHOD, spec)
    with pytest.raises(ValueError, match="'merge'.*true or false"):
        validate_measure_params("coastal_levee", {"merge": "no"}, METHOD, spec)
