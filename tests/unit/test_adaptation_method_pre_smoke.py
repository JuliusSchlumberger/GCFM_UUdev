"""Smoke tests for src.adaptation_method_pre parameter validation."""

import pytest
from src.adaptation_method_pre import validate_measure_params

METHOD = "preprocessing"
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


def test_bool_param_rejects_non_bool_value():
    spec = {
        "params": {
            "merge": {"min": None, "max": None, "type": "bool", "required_for": []}
        }
    }
    validate_measure_params("coastal_levee", {"merge": False}, METHOD, spec)
    with pytest.raises(ValueError, match="'merge'.*true or false"):
        validate_measure_params("coastal_levee", {"merge": "no"}, METHOD, spec)
