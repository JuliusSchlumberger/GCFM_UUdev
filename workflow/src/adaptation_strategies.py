"""Expand adaptation strategy families into per-scale strategies (00_common.smk).

adaptation_strategies.yml holds strategy families written at scale 1; each
family expands into one strategy per entry of `scales` (0.4 -> <family>_04,
0.9 -> <family>_09, 1.0 -> <family>_1). A param is multiplied by the scale only
when measures.yml flags it `scale: true`. A param value is one of:

- number: same scale-1 value for every event type
- {coastal: .., river: .., compound: ..}: scale-1 value per scenario `event`
- one of DERIVED_PARAMS: scale-1 value read from the scenario's own baseline
  run at run time (forcing/map files don't exist yet at DAG-build time),
  resolved by src.adaptation_scaling
- current_protection: today's baseline weir crest, never scaled
- {fixed: x}: x, never scaled

A measure's file params (FILE_PARAMS: `locations`, water_retention's `inlet`)
are paths under adaptation_data_root, in which `{basin_id}` stands for the
basin, or `auto/<name>`: the basin's <name>.geojson written by rule
build_adaptation_lines.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

from src.io import require_keys

STRATEGY_EVENTS = ("coastal", "river", "compound")
DERIVED_PARAMS = {"max_coastal_water_level", "max_river_water_level", "max_discharge"}
UNSCALED_DERIVED_PARAMS = {"current_protection"}

FILE_PARAMS = ("locations", "inlet")
AUTO_LOCATIONS_PREFIX = "auto/"
BASIN_PLACEHOLDER = "{basin_id}"
# Files rule build_adaptation_lines writes per basin as <name>.geojson.
ADAPTATION_LINE_NAMES = (
    "coastal_protection_weir_PC",  # protect_closed: coast + mouth closure(s)
    "coastal_protection_weir_PO",  # grey_protect_open coastal_levee: coast only
    "river_protection_weir_PO",  # grey_protect_open river_levee: river + estuary banks
    "pump_river_mouth",  # pumps: river centreline across each closure, inland -> sea
)


def auto_line_name(location: str) -> str | None:
    """Return <name> of an `auto/<name>` locations value, else None."""
    if not location.startswith(AUTO_LOCATIONS_PREFIX):
        return None
    name = location[len(AUTO_LOCATIONS_PREFIX) :]
    if name not in ADAPTATION_LINE_NAMES:
        raise ValueError(
            f"locations {location!r}: unknown adaptation line {name!r} "
            f"(rule build_adaptation_lines writes {list(ADAPTATION_LINE_NAMES)})"
        )
    return name


def location_path(
    location: str, adaptation_data_root: str, lines_dir: str, basin_id: str
) -> str:
    """Return the file of a locations value: lines_dir for auto/, else adaptation_data_root."""
    name = auto_line_name(location)
    if name is not None:
        return f"{lines_dir}/{name}.geojson"
    return str(
        Path(adaptation_data_root) / location.replace(BASIN_PLACEHOLDER, str(basin_id))
    )


def resolve_locations(strategy_def: dict, lines_dir: str | Path, basin_id: str) -> dict:
    """Return strategy_def with its file params (FILE_PARAMS) resolved for one basin.

    `auto/<name>` becomes an absolute path into lines_dir: the adapt scripts join
    file params onto adaptation_data_root, and an absolute path survives that join
    unchanged. Other values keep their adaptation_data_root-relative form with
    `{basin_id}` filled in.
    """
    lines_dir = Path(os.path.abspath(lines_dir)).as_posix()

    def _resolve(param, value):
        if param not in FILE_PARAMS or not isinstance(value, str):
            return value
        name = auto_line_name(value)
        if name is not None:
            return f"{lines_dir}/{name}.geojson"
        return value.replace(BASIN_PLACEHOLDER, str(basin_id))

    return {
        **strategy_def,
        "measures": {
            m: {p: _resolve(p, v) for p, v in params.items()}
            for m, params in strategy_def["measures"].items()
        },
    }


def scale_tag(scale: float) -> str:
    """Return the strategy-name suffix of a scale: 0.4 -> '04', 1.0 -> '1'."""
    return f"{scale:g}".replace(".", "")


@dataclass(frozen=True)
class StrategyExpansion:
    """Expanded strategies plus the definitions needed to resolve them per scenario."""

    families: dict
    defs: dict
    measures: dict
    scenario_defs: dict
    scalable_params: frozenset
    scenarios_file: str

    def strategy_def_for(self, strategy: str, scenario: str) -> dict:
        """Return {"measures": {...}} for one strategy in one scenario.

        Per-event values follow the scenario's `event`; scale-flagged params are
        multiplied by the strategy scale; run-time-derived params stay as
        {"from_forcing": ...} markers for src.adaptation_scaling.
        """
        event = self.scenario_defs[scenario].get("event")
        if event is not None and event not in STRATEGY_EVENTS:
            raise ValueError(
                f"scenario {scenario!r}: event must be one of {STRATEGY_EVENTS}, got {event!r}"
            )
        return {
            "measures": self._resolve_measures(
                strategy, event, f"{strategy}/{scenario}"
            )
        }

    def _resolve_measures(self, strategy, event, where):
        sdef = self.defs[strategy]
        return {
            m: {
                p: self._resolve_param(
                    m, p, v, sdef["scale"], event, f"{where} {m}.{p}"
                )
                for p, v in params.items()
            }
            for m, params in sdef["measures"].items()
        }

    def _resolve_param(self, measure, param, value, scale, event, where):
        if isinstance(value, dict) and set(value) <= set(STRATEGY_EVENTS) and value:
            if set(value) != set(STRATEGY_EVENTS):
                raise ValueError(
                    f"{where}: per-event map must set all of {STRATEGY_EVENTS}, got {sorted(value)}"
                )
            if event is None:
                raise ValueError(
                    f"{where}: value depends on the event type, but the scenario has no "
                    f"`event:` in {self.scenarios_file}"
                )
            value = value[event]

        scalable = (measure, param) in self.scalable_params
        if isinstance(value, str) and value in UNSCALED_DERIVED_PARAMS:
            return {"from_forcing": value}
        if isinstance(value, str) and value in DERIVED_PARAMS:
            if not scalable:
                raise ValueError(
                    f"{where}: {value!r} only allowed on a param flagged `scale: true` in measures.yml"
                )
            return {"from_forcing": value, "scale": scale}

        if isinstance(value, dict) and set(value) == {"fixed"}:
            value, scalable = value["fixed"], False
        elif isinstance(value, dict):
            raise ValueError(f"{where}: unrecognised value {value!r}")
        elif isinstance(value, str) and scalable:
            raise ValueError(
                f"{where}: unknown keyword {value!r} (expected a number, "
                f"{sorted(DERIVED_PARAMS | UNSCALED_DERIVED_PARAMS)} or {{fixed: x}})"
            )

        if not scalable:
            return value

        value = float(value) * scale
        spec = self.measures[measure]["params"][param]
        lo, hi = spec.get("min"), spec.get("max")
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            raise ValueError(
                f"{where}: {param}={value:g} outside [{lo}, {hi}] from measures.yml"
            )
        return value


def expand_strategies(
    strategies_raw: dict,
    measures_raw: dict,
    scenario_defs: dict,
    strategies_file: str,
    measures_file: str,
    scenarios_file: str,
) -> StrategyExpansion:
    """Expand and validate every strategy family against measures and event types.

    The *_file arguments only label error messages. Every strategy x event type
    is resolved once here, so a typo or an out-of-range value fails at
    DAG-build time instead of in a single job mid-run.
    """
    require_keys(measures_raw, ("adaptation_measures",), measures_file)
    measures = measures_raw["adaptation_measures"]

    require_keys(strategies_raw, ("scales",), strategies_file)
    scales = strategies_raw["scales"]
    for m, mdef in measures.items():
        require_keys(mdef, ("params",), f"{measures_file}: adaptation_measures.{m}")
    scalable_params = frozenset(
        (m, p)
        for m, mdef in measures.items()
        for p, spec in mdef["params"].items()
        if isinstance(spec, dict) and spec.get("scale")
    )

    families = {k: v for k, v in strategies_raw.items() if k != "scales"}
    if not families:
        raise ValueError(f"{strategies_file} defines no strategies (besides 'scales')")

    defs = {}
    for fam, fdef in families.items():
        require_keys(fdef or {}, ("measures",), f"{strategies_file}: {fam}")
        for m, mparams in fdef["measures"].items():
            if m not in measures:
                raise ValueError(
                    f"strategy {fam!r}: measure {m!r} not defined in {measures_file}"
                )
            for p in FILE_PARAMS:
                if isinstance((mparams or {}).get(p), str):
                    auto_line_name(mparams[p])
        for s in scales:
            name = f"{fam}_{scale_tag(s)}"
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError(
                    f"strategy name {name!r} invalid (use letters/digits/_/- only)"
                )
            if name in defs:
                raise ValueError(
                    f"strategy name {name!r} generated twice -- check family names/scales"
                )
            # `measures` stays the raw family definition: strategy_measure_input_paths
            # and water_retention_excess_volume_input only read measure names and
            # `locations` strings, which never vary per scale or event.
            defs[name] = {
                "family": fam,
                "scale": float(s),
                "measures": fdef["measures"],
            }

    expansion = StrategyExpansion(
        families=families,
        defs=defs,
        measures=measures,
        scenario_defs=scenario_defs,
        scalable_params=scalable_params,
        scenarios_file=scenarios_file,
    )
    for strategy in defs:
        for event in STRATEGY_EVENTS:
            expansion._resolve_measures(strategy, event, f"{strategy}[{event}]")
    return expansion
