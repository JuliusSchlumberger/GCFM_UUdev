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


def test_bool_param_rejects_non_bool_value():
    spec = {
        "params": {
            "merge": {"min": None, "max": None, "type": "bool", "required_for": []}
        }
    }
    validate_measure_params("coastal_levee", {"merge": False}, METHOD, spec)
    with pytest.raises(ValueError, match="'merge'.*true or false"):
        validate_measure_params("coastal_levee", {"merge": "no"}, METHOD, spec)
