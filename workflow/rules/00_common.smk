# Shared setup included first: paths, basins, grid/scenario/strategy config, wildcards.
import json
import os

from src.io import load_catalogue, catalogue_entry,read_geometry, raw_input_path, local_path
from src.io import general_path as _gen_path
from src.river_forcing import derive_forcing_mode
import geopandas as gpd

import re as _re
import yaml as _yaml

CATALOGUE = load_catalogue(config["data_catalogue"])

# ── local machine paths ──
# Machine-specific paths come from environment variables (CONTRIBUTING.md
# "Local machine paths"): GCFM_RESULTS_DIR, GCFM_RAW_DATA_ROOT (applied in
# load_catalogue) and GCFM_SFINCS_EXE.
# Forward slashes: mixed separators break Snakemake output/input matching.
config["results_dir"] = Path(local_path("GCFM_RESULTS_DIR")).as_posix()

def sfincs_exe_path(wildcards=None):
    """Return the SFINCS executable path (GCFM_SFINCS_EXE).

    Resolved lazily as a rule param, so only solver rules require it.
    """
    return local_path("GCFM_SFINCS_EXE")

def catalogue_path(name):
    """Return the raw-data path of a catalogue entry."""
    return raw_input_path(CATALOGUE, name)

def general_path(name):
    """Return the general path of a catalogue entry."""
    return _gen_path(CATALOGUE, name)

def list_basins(name):
    """Return the sorted basin ids in the delta polygons dataset."""
    dataset = catalogue_entry(CATALOGUE, name)
    _delta_shp = catalogue_path(name)
    attribute = dataset["attributes"][0]["name"]

    _deltas = read_geometry(_delta_shp)
    _delta_ids = _deltas[attribute]
    return sorted(_delta_ids.astype(int).to_list())

BASINS = list_basins("delta_polygons")

# Optional CLI override to restrict a run to specific basin(s):
#   snakemake build --cores N --config target_basins="[2433835]" --forceall
if config.get("target_basins"):
    _target_basins = [int(b) for b in config["target_basins"]]
    _unknown = sorted(set(_target_basins) - set(BASINS))
    if _unknown:
        raise ValueError(
            f"target_basins {_unknown} not found among discovered BASINS "
            f"(from delta_polygons) — check the basin_id(s)"
        )
    BASINS = _target_basins

RESULTS_DIR = config["results_dir"]

def results_path(pattern):
    """Return pattern prefixed with the results directory."""
    return f"{RESULTS_DIR}/{pattern}"


# ── SFINCS main-grid resolution ──
# dx (m) per basin: sfincs.grid.resolution_m, or resolution_overrides_m (e.g.
# to fit a large delta in memory). Subgrid pixel = dx / nr_subgrid_pixels.
_GRID_CFG = config["sfincs"]["grid"]
_GRID_RES_OVERRIDES = {
    int(_b): _r for _b, _r in (_GRID_CFG.get("resolution_overrides_m") or {}).items()
}
for _label, _r in [("resolution_m", _GRID_CFG["resolution_m"])] + [
    (f"resolution_overrides_m[{_b}]", _r) for _b, _r in _GRID_RES_OVERRIDES.items()
]:
    if isinstance(_r, bool) or not isinstance(_r, (int, float)) or not _r > 0:
        raise ValueError(f"sfincs.grid.{_label} must be a positive number (m), got {_r!r}")

def grid_resolution_m(basin_id):
    """Return a basin's main-grid dx (m): its override, else the default."""
    return float(_GRID_RES_OVERRIDES.get(int(basin_id), _GRID_CFG["resolution_m"]))

# Grid orientation: axis-aligned, or the minimum rotated rectangle around the
# delta polygon (sfincs.grid.rotated), optionally per basin (rotated_overrides).
_GRID_ROTATED_OVERRIDES = {
    int(_b): _r for _b, _r in (_GRID_CFG.get("rotated_overrides") or {}).items()
}
for _label, _r in [("rotated", _GRID_CFG.get("rotated", False))] + [
    (f"rotated_overrides[{_b}]", _r) for _b, _r in _GRID_ROTATED_OVERRIDES.items()
]:
    if not isinstance(_r, bool):
        raise ValueError(f"sfincs.grid.{_label} must be true or false, got {_r!r}")

def grid_rotated(basin_id):
    """Return whether a basin's grid is rotated: its override, else the default."""
    return bool(_GRID_ROTATED_OVERRIDES.get(int(basin_id), _GRID_CFG.get("rotated", False)))


# ── storm-tide event window ──
# Half-length (h) of the storm-tide event around its peak, for the surge shapes
# and the boundary event; overridable per basin (window_overrides_hr), since
# surges last hours on some coasts and days on others.
_HG_CFG = config["boundary_forcings"]["surge"]["hydrograph"]
_HG_WINDOW_OVERRIDES = {
    int(_b): _w for _b, _w in (_HG_CFG.get("window_overrides_hr") or {}).items()
}
for _label, _w in [("window_hr", _HG_CFG["window_hr"])] + [
    (f"window_overrides_hr[{_b}]", _w) for _b, _w in _HG_WINDOW_OVERRIDES.items()
]:
    if isinstance(_w, bool) or not isinstance(_w, (int, float)) or not _w > 0:
        raise ValueError(
            f"boundary_forcings.surge.hydrograph.{_label} must be a positive number (h), got {_w!r}"
        )
    # The surge and river peaks sit at lead_days * 24 + window: off the
    # forcing time step, neither peak would be sampled and both would be
    # clipped below their design values.
    _dt = float(config["boundary_forcings"]["dt_hr"])
    if abs(_w / _dt - round(_w / _dt)) > 1e-9:
        raise ValueError(
            f"boundary_forcings.surge.hydrograph.{_label} = {_w!r} h must be a multiple of "
            f"boundary_forcings.dt_hr ({_dt:g} h), or the event peak falls between forcing time steps"
        )

def hydrograph_window_hr(basin_id):
    """Return a basin's storm-tide event half-window (h): its override, else the default."""
    return float(_HG_WINDOW_OVERRIDES.get(int(basin_id), _HG_CFG["window_hr"]))


# ── scenario axis ──
# Scenarios are named in config/scenarios.yml ("default" is used when no
# target_scenarios is given). forcing_mode is derived from which RPs are set.

_SURGE_RPS = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000)    # COAST-RP tabulated; no interpolation
_RIVER_RP_MIN, _RIVER_RP_MAX = 2, 1000                     # log-interpolated from discharge_rp_table

with open(config["scenarios_file"]) as _f:
    SCENARIO_DEFS = _yaml.safe_load(_f) or {}

if "default" not in SCENARIO_DEFS:
    raise ValueError(
        f"{config['scenarios_file']} must define a 'default' scenario -- "
        f"used whenever target_scenarios is not given"
    )

for _name, _s in SCENARIO_DEFS.items():
    if not _re.fullmatch(r"[A-Za-z0-9_-]+", _name):
        raise ValueError(f"scenario name {_name!r} invalid (use letters/digits/_/- only)")
    _srp, _rrp = _s.get("surge_rp"), _s.get("river_rp")
    _srp_is_tide = isinstance(_srp, str) and _srp.strip().lower() == "tide"
    if _srp is not None and not _srp_is_tide and _srp not in _SURGE_RPS:
        raise ValueError(
            f"{_name}: surge_rp must be 'Tide', null, or a COAST-RP "
            f"tabulated value {_SURGE_RPS}"
        )
    _rrp_is_mean = isinstance(_rrp, str) and _rrp.strip().lower() == "mean"
    if _rrp is not None and not _rrp_is_mean and not _RIVER_RP_MIN <= _rrp <= _RIVER_RP_MAX:
        raise ValueError(
            f"{_name}: river_rp must be 'Mean', null, or in "
            f"[{_RIVER_RP_MIN}, {_RIVER_RP_MAX}] yr"
        )
    _dm = _s.get("discharge_multiplier", 1.0)
    if not _dm > 0:
        raise ValueError(f"{_name}: discharge_multiplier must be > 0")

def scenario_params(name):
    """Return a scenario's mode, surge_rp, river_rp, discharge_multiplier and slr_m.

    discharge_multiplier (default 1.0) and slr_m (m, default 0.0) are
    per-scenario and read only by build_sfincs/adapt_build_forcing_pre, so
    changing them invalidates only that scenario.
    """
    s = SCENARIO_DEFS[name]
    river_rp, surge_rp = s.get("river_rp"), s.get("surge_rp")
    try:
        mode = derive_forcing_mode(river_rp, surge_rp)
    except ValueError as e:
        raise ValueError(f"scenario {name!r}: {e}") from e
    return {
        "mode": mode,
        "surge_rp": surge_rp,
        "river_rp": river_rp,
        "discharge_multiplier": s.get("discharge_multiplier", 1.0),
        "slr_m": s.get("slr_m", 0.0),
    }

def attribution_counterparts(scenario):
    """Return the (river_only, coastal_only) sibling scenarios with matching RPs.

    Siblings have river_rp=X & surge_rp=None, and surge_rp=Y & river_rp=None.
    Raises ValueError if scenarios.yml has no exact match; RPs are never guessed.
    """

    river_rp = SCENARIO_DEFS[scenario]["river_rp"]
    surge_rp = SCENARIO_DEFS[scenario]["surge_rp"]

    def _find(rp_key, other_key, rp_val):
        matches = [n for n, s in SCENARIO_DEFS.items()
                   if s.get(rp_key) == rp_val and s.get(other_key) is None]
        if not matches:
            raise ValueError(
                f"No {rp_key}={rp_val}-matched, {other_key}=None sibling scenario "
                f"for {scenario!r} -- add one to config/scenarios.yml"
            )
        return matches[0]

    river_only = _find("river_rp", "surge_rp", river_rp)
    coastal_only = _find("surge_rp", "river_rp", surge_rp)
    return river_only, coastal_only


# What to run: CLI override, else just the "default" scenario.
#   snakemake build --config target_scenarios="['baseline','coast_100']"
SCENARIOS = list(config.get("target_scenarios", ["default"]))
_unknown = sorted(set(SCENARIOS) - set(SCENARIO_DEFS))
if _unknown:
    raise ValueError(f"target_scenarios {_unknown} not defined in {config['scenarios_file']}")

# Basin-level restart filename: run_spinup (14) always runs at a fixed RP=1
# river + calm sea for spinup_days, so the file is shared by every scenario.
# Defined here because rules 13, 14 and 16 all need it.
from datetime import datetime as _datetime, timedelta as _timedelta

_tref       = _datetime.strptime(config["sfincs"]["simulation"]["tref"], "%Y-%m-%d %H:%M:%S")
_spinup_end = _tref + _timedelta(days=config["sfincs"]["spinup"]["spinup_days"])
RST_FNAME   = f"sfincs.{_spinup_end.strftime('%Y%m%d.%H%M%S')}.rst"



# ---- Adding adaptation wildcard (tests/KL_check_adaptation_scaling.py slices this file from here)
with open(config["adaptation"]["strategies_file"]) as _f:
    STRATEGY_DEFS_RAW = _yaml.safe_load(_f) or {}

# Strategy measure files live under the catalogue's raw-data root (GCFM_RAW_DATA_ROOT).
ADAPT_CATALOGUE_ROOT = CATALOGUE["meta"]["root"]

with open(config["adaptation"]["measures_file"]) as _f:
    MEASURES_DEFS = (_yaml.safe_load(_f) or {}).get("adaptation_measures", {})


# adaptation_strategies.yml holds strategy families written at scale 1; each
# family expands into one {strategy} per entry of `scales` (0.4 -> <family>_04,
# 0.9 -> <family>_09, 1.0 -> <family>_1). A param is multiplied by the scale
# only when measures.yml flags it `scale: true`. A param value is one of:
#   number                         same scale-1 value for every event type
#   {coastal: .., river: .., compound: ..}
#                                  scale-1 value per scenario `event` (scenarios.yml)
#   one of _DERIVED_PARAMS         scale-1 value read from the scenario's own
#                                  baseline run at run time (forcing/map files
#                                  don't exist yet at DAG-build time), resolved
#                                  by src.adaptation_scaling
#   current_protection             today's baseline weir crest, never scaled
#   {fixed: x}                     x, never scaled

_STRATEGY_EVENTS = ("coastal", "river", "compound")
_DERIVED_PARAMS = {"max_coastal_waterlevel", "max_river_waterlevel", "max_discharge"}
_UNSCALED_DERIVED_PARAMS = {"current_protection"}
_STRATEGY_SCALES = STRATEGY_DEFS_RAW.get("scales", [0.4, 0.9, 1.0])
_SCALABLE_PARAMS = {
    (_m, _p)
    for _m, _mdef in MEASURES_DEFS.items()
    for _p, _spec in (_mdef.get("params") or {}).items()
    if isinstance(_spec, dict) and _spec.get("scale")
}

def _scale_tag(scale):
    """0.4 -> '04', 0.9 -> '09', 1.0 -> '1'."""
    return f"{scale:g}".replace(".", "")

STRATEGY_FAMILIES = {k: v for k, v in STRATEGY_DEFS_RAW.items() if k not in ("meta", "scales")}
if not STRATEGY_FAMILIES:
    raise ValueError(f"{config['adaptation']['strategies_file']} defines no strategies (besides 'meta'/'scales')")


STRATEGY_DEFS = {}
for _fam, _fdef in STRATEGY_FAMILIES.items():
    for _m in (_fdef or {}).get("measures", {}):
        if _m not in MEASURES_DEFS:
            raise ValueError(f"strategy {_fam!r}: measure {_m!r} not defined in {config['adaptation']['measures_file']}")
    for _s in _fdef.get("scales", _STRATEGY_SCALES):
        _name = f"{_fam}_{_scale_tag(_s)}"
        if not _re.fullmatch(r"[A-Za-z0-9_-]+", _name):
            raise ValueError(f"strategy name {_name!r} invalid (use letters/digits/_/- only)")
        if _name in STRATEGY_DEFS:
            raise ValueError(f"strategy name {_name!r} generated twice -- check family names/scales")
        # `measures` stays the raw family definition: strategy_measure_input_paths
        # and water_retention_excess_volume_input only read measure names and
        # `locations` strings, which never vary per scale or event.
        STRATEGY_DEFS[_name] = {"family": _fam, "scale": float(_s), "measures": _fdef["measures"]}


def _resolve_strategy_param(measure, param, value, scale, event, where):

    if isinstance(value, dict) and set(value) <= set(_STRATEGY_EVENTS) and value:
        if set(value) != set(_STRATEGY_EVENTS):
            raise ValueError(f"{where}: per-event map must set all of {_STRATEGY_EVENTS}, got {sorted(value)}")
        if event is None:
            raise ValueError(f"{where}: value depends on the event type, but the scenario has no "
                             f"`event:` in {config['scenarios_file']}")
        value = value[event]

    scalable = (measure, param) in _SCALABLE_PARAMS
    if isinstance(value, str) and value in _UNSCALED_DERIVED_PARAMS:
        return {"from_forcing": value}
    if isinstance(value, str) and value in _DERIVED_PARAMS:
        if not scalable:
            raise ValueError(f"{where}: {value!r} only allowed on a param flagged `scale: true` in measures.yml")
        return {"from_forcing": value, "scale": scale}

    if isinstance(value, dict) and set(value) == {"fixed"}:
        value, scalable = value["fixed"], False
    elif isinstance(value, dict):
        raise ValueError(f"{where}: unrecognised value {value!r}")
    elif isinstance(value, str) and scalable:
        raise ValueError(f"{where}: unknown keyword {value!r} (expected a number, "
                         f"{sorted(_DERIVED_PARAMS | _UNSCALED_DERIVED_PARAMS)} or {{fixed: x}})")

    if not scalable:
        return value

    value = float(value) * scale
    spec = MEASURES_DEFS[measure]["params"][param]
    lo, hi = spec.get("min"), spec.get("max")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ValueError(f"{where}: {param}={value:g} outside [{lo}, {hi}] from measures.yml")
    return value


def _resolve_strategy_measures(strategy, event, where):
    sdef = STRATEGY_DEFS[strategy]
    return {
        m: {p: _resolve_strategy_param(m, p, v, sdef["scale"], event, f"{where} {m}.{p}")
            for p, v in params.items()}
        for m, params in sdef["measures"].items()
    }


def strategy_def_for(strategy, scenario):
    """Return {"measures": {...}} for one strategy in one scenario.

    Per-event values follow the scenario's `event`; scale-flagged params are
    multiplied by the strategy scale; run-time-derived params stay as
    {"from_forcing": ...} markers for src.adaptation_scaling.
    """
    event = SCENARIO_DEFS[scenario].get("event")
    if event is not None and event not in _STRATEGY_EVENTS:
        raise ValueError(f"scenario {scenario!r}: event must be one of {_STRATEGY_EVENTS}, got {event!r}")
    return {"measures": _resolve_strategy_measures(strategy, event, f"{strategy}/{scenario}")}


# Validate every strategy x event type once at startup, so a typo or an
# out-of-range value fails `snakemake -n` instead of a single job mid-run.
for _strategy in STRATEGY_DEFS:
    for _event in _STRATEGY_EVENTS:
        _resolve_strategy_measures(_strategy, _event, f"{_strategy}[{_event}]")


def adaptation_input_path(rel):
    """Resolve a measure's locations/dep_subgrid path against the raw-data root."""
    return str(Path(ADAPT_CATALOGUE_ROOT) / rel)

def strategy_measure_input_paths(strategy):
    """Return the raw-data files a strategy's measures reference, for rule inputs."""
    paths = []
    for _measure_params in STRATEGY_DEFS[strategy]["measures"].values():
        for _k, _v in _measure_params.items():
            if _k in ("locations", "dep_subgrid") and isinstance(_v, str):
                paths.append(adaptation_input_path(_v))
    return paths


def water_retention_excess_volume_input(wildcards):
    """Return baseline_excess_volume.json if the strategy uses water retention, else [].

    Conditional so other strategies do not require rule attribution_mask and
    its river-only/coastal-only sibling scenarios, which not every scenario has.
    """
    if {"water_retention", "water_retention_greening"} & set(STRATEGY_DEFS[wildcards.strategy]["measures"]):
        return results_path(f"{wildcards.basin_id}/runs/{wildcards.scenario}/baseline_excess_volume.json")
    return []


# Opt-in only; no default strategy.
# Each entry is a single scale or a family name (= all of its scales); both
# can be mixed in one list:
#   snakemake adapt --config target_strategies="['retreat_04']"
#   snakemake adapt --config target_strategies="['retreat']"                 # retreat_04, _09, _1
#   snakemake adapt --config target_strategies="['retreat','protect_closed_1']"
STRATEGIES = []
for _t in config.get("target_strategies", []):
    _expanded = [_t] if _t in STRATEGY_DEFS else [
        n for n, d in STRATEGY_DEFS.items() if d["family"] == _t
    ]
    if not _expanded:
        raise ValueError(
            f"target_strategies: {_t!r} is neither a strategy nor a family in "
            f"{config['adaptation']['strategies_file']} (families: {sorted(STRATEGY_FAMILIES)})"
        )
    STRATEGIES += [n for n in _expanded if n not in STRATEGIES]

# Both methods run by default, unless a strategy is selected:
#   snakemake adapt --config target_strategies="['retreat']" target_method="['pre']"
METHODS = list(config.get("target_method", ["pre", "post"]))
_unknown_methods = sorted(set(METHODS) - {"pre", "post"})
if _unknown_methods:
    raise ValueError(f"target_method {_unknown_methods} invalid -- only 'pre'/'post' supported")


wildcard_constraints:
    basin_id = r"\d+",
    scenario = r"|".join(sorted(SCENARIO_DEFS)),
    strategy = r"|".join(sorted(STRATEGY_DEFS)) if STRATEGY_DEFS else r"(?!)",
