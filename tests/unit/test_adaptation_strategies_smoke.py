"""Smoke tests for src.adaptation_strategies on a tiny synthetic config."""

from pathlib import Path

import pytest
from src.adaptation_strategies import (
    expand_strategies,
    location_path,
    resolve_locations,
    scale_tag,
)

MEASURES = {
    "adaptation_measures": {
        "dike": {
            "params": {
                "crest": {"scale": True, "min": 0.0, "max": 10.0},
                "locations": {},
            }
        },
        "pumps": {"params": {"discharge": {"scale": True}, "keep": {}}},
    }
}
SCENARIOS = {
    "coast": {"event": "coastal"},
    "river": {"event": "river"},
    "plain": {},
}
FILES = {
    "strategies_file": "strategies.yml",
    "measures_file": "measures.yml",
    "scenarios_file": "scenarios.yml",
}


def _strategies(**dike_params):
    return {
        "scales": [0.5, 1.0],
        "wall": {
            "measures": {
                "dike": {"locations": "site/dike.geojson", **dike_params},
            }
        },
        "pump": {
            "measures": {
                "pumps": {"discharge": "max_discharge", "keep": {"fixed": 3}},
            }
        },
    }


def _expand(strategies):
    return expand_strategies(strategies, MEASURES, SCENARIOS, **FILES)


def test_scale_tag_drops_the_decimal_point():
    assert [scale_tag(s) for s in (0.4, 0.9, 1.0)] == ["04", "09", "1"]


def test_each_family_expands_into_one_strategy_per_scale():
    exp = _expand(_strategies(crest=4.0))
    assert list(exp.defs) == ["wall_05", "wall_1", "pump_05", "pump_1"]
    assert exp.defs["wall_05"]["family"] == "wall"
    assert exp.defs["wall_05"]["scale"] == 0.5
    assert set(exp.families) == {"wall", "pump"}


def test_scaled_param_multiplied_and_unscaled_params_kept():
    exp = _expand(_strategies(crest=4.0))
    dike = exp.strategy_def_for("wall_05", "plain")["measures"]["dike"]
    assert dike == {"locations": "site/dike.geojson", "crest": 2.0}


def test_per_event_value_follows_scenario_event():
    exp = _expand(_strategies(crest={"coastal": 4.0, "river": 2.0, "compound": 6.0}))
    assert exp.strategy_def_for("wall_1", "coast")["measures"]["dike"]["crest"] == 4.0
    assert exp.strategy_def_for("wall_1", "river")["measures"]["dike"]["crest"] == 2.0
    with pytest.raises(ValueError, match="no `event:` in scenarios.yml"):
        exp.strategy_def_for("wall_1", "plain")


def test_derived_param_stays_a_marker_and_fixed_value_unscaled():
    exp = _expand(_strategies(crest=4.0))
    pumps = exp.strategy_def_for("pump_05", "coast")["measures"]["pumps"]
    assert pumps == {
        "discharge": {"from_forcing": "max_discharge", "scale": 0.5},
        "keep": 3,
    }


def test_out_of_range_value_fails_at_expansion():
    with pytest.raises(ValueError, match=r"crest=12 outside \[0.0, 10.0\]"):
        _expand(_strategies(crest=12.0))


def test_unknown_measure_fails_with_measures_file_name():
    strategies = _strategies(crest=1.0)
    strategies["wall"]["measures"]["moat"] = {}
    with pytest.raises(ValueError, match="'moat' not defined in measures.yml"):
        _expand(strategies)


def test_auto_location_resolves_to_the_basin_lines_and_others_to_raw_root(tmp_path):
    lines_dir = (tmp_path / "results" / "1" / "adaptation_lines").as_posix()
    assert (
        location_path("auto/pump_river_mouth", "/raw", lines_dir, "1")
        == f"{lines_dir}/pump_river_mouth.geojson"
    )
    assert location_path("site/dike.geojson", "/raw", lines_dir, "1") == str(
        Path("/raw") / "site/dike.geojson"
    )
    assert location_path("site_{basin_id}/dike.geojson", "/raw", lines_dir, "1") == str(
        Path("/raw") / "site_1/dike.geojson"
    )


def test_auto_location_becomes_absolute_and_survives_the_raw_root_join(tmp_path):
    exp = _expand(_strategies(crest=4.0, locations="auto/coastal_protection_weir_PC"))
    lines_dir = tmp_path / "adaptation_lines"
    sdef = resolve_locations(exp.strategy_def_for("wall_1", "plain"), lines_dir, "1")
    resolved = sdef["measures"]["dike"]["locations"]
    assert resolved == f"{lines_dir.as_posix()}/coastal_protection_weir_PC.geojson"
    assert Path("/raw") / resolved == Path(resolved)
    assert sdef["measures"]["dike"]["crest"] == 4.0
    plain = resolve_locations(
        _expand(_strategies(crest=4.0)).strategy_def_for("wall_1", "plain"),
        lines_dir,
        "1",
    )
    assert plain["measures"]["dike"]["locations"] == "site/dike.geojson"
    per_basin = resolve_locations(
        _expand(
            _strategies(crest=4.0, locations="site_{basin_id}/dike.geojson")
        ).strategy_def_for("wall_1", "plain"),
        lines_dir,
        "7",
    )
    assert per_basin["measures"]["dike"]["locations"] == "site_7/dike.geojson"


def test_unknown_auto_line_fails_at_expansion():
    with pytest.raises(ValueError, match="unknown adaptation line 'moat'"):
        _expand(_strategies(crest=1.0, locations="auto/moat"))
