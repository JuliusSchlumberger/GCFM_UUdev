# Shared setup included first: paths, basins, grid/scenario/strategy config, wildcards.
import json
import os

from src.io import load_catalogue, catalogue_entry,read_geometry, raw_input_path, site_setting, require_keys
from src.io import general_path as _gen_path
from src.river_forcing import derive_forcing_mode
from src.adaptation_strategies import (
    ADAPTATION_LINE_NAMES, expand_strategies, location_path, with_auto_locations,
)
import geopandas as gpd

import re as _re
import yaml as _yaml

CATALOGUE = load_catalogue(config["data_catalogue"])

# ── site settings ──
# Machine-specific paths come from config/site.yml (see config/site.example.yml):
# results_dir, raw_data_root (applied in load_catalogue) and sfincs_exe.
# Forward slashes: mixed separators break Snakemake output/input matching.
config["results_dir"] = Path(site_setting("results_dir")).as_posix()

def sfincs_exe_path(wildcards=None):
    """Return the SFINCS executable path (site setting sfincs_exe).

    Resolved lazily as a rule param, so only solver rules require it.
    """
    return site_setting("sfincs_exe")

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
for _label, _r in [("rotated", _GRID_CFG["rotated"])] + [
    (f"rotated_overrides[{_b}]", _r) for _b, _r in _GRID_ROTATED_OVERRIDES.items()
]:
    if not isinstance(_r, bool):
        raise ValueError(f"sfincs.grid.{_label} must be true or false, got {_r!r}")

def grid_rotated(basin_id):
    """Return whether a basin's grid is rotated: its override, else the default."""
    return bool(_GRID_ROTATED_OVERRIDES.get(int(basin_id), _GRID_CFG["rotated"]))


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
    _srp, _rrp = _s.get("surge_rp_yr"), _s.get("river_rp_yr")
    _srp_is_tide = isinstance(_srp, str) and _srp.strip().lower() == "tide"
    if _srp is not None and not _srp_is_tide and _srp not in _SURGE_RPS:
        raise ValueError(
            f"{_name}: surge_rp_yr must be 'Tide', null, or a COAST-RP "
            f"tabulated value {_SURGE_RPS}"
        )
    _rrp_is_mean = isinstance(_rrp, str) and _rrp.strip().lower() == "mean"
    if _rrp is not None and not _rrp_is_mean and not _RIVER_RP_MIN <= _rrp <= _RIVER_RP_MAX:
        raise ValueError(
            f"{_name}: river_rp_yr must be 'Mean', null, or in "
            f"[{_RIVER_RP_MIN}, {_RIVER_RP_MAX}] yr"
        )
    require_keys(_s, ("slr_m",), f"{config['scenarios_file']}: {_name}")

def scenario_params(name):
    """Return a scenario's mode, surge_rp_yr, river_rp_yr and slr_m.

    slr_m (m) is per-scenario and read only by
    build_sfincs/adapt_build_forcing_pre, so changing them invalidates only
    that scenario.
    """
    s = SCENARIO_DEFS[name]
    river_rp_yr, surge_rp_yr = s.get("river_rp_yr"), s.get("surge_rp_yr")
    try:
        mode = derive_forcing_mode(river_rp_yr, surge_rp_yr)
    except ValueError as e:
        raise ValueError(f"scenario {name!r}: {e}") from e
    return {
        "mode": mode,
        "surge_rp_yr": surge_rp_yr,
        "river_rp_yr": river_rp_yr,
        "slr_m": s["slr_m"],
    }

def attribution_counterparts(scenario):
    """Return the (river_only, coastal_only) sibling scenarios with matching RPs.

    Siblings have river_rp_yr=X & surge_rp_yr=None, and surge_rp_yr=Y & river_rp_yr=None.
    Raises ValueError if scenarios.yml has no exact match; RPs are never guessed.
    """

    river_rp_yr = SCENARIO_DEFS[scenario]["river_rp_yr"]
    surge_rp_yr = SCENARIO_DEFS[scenario]["surge_rp_yr"]

    def _find(rp_key, other_key, rp_val):
        matches = [n for n, s in SCENARIO_DEFS.items()
                   if s.get(rp_key) == rp_val and s.get(other_key) is None]
        if not matches:
            raise ValueError(
                f"No {rp_key}={rp_val}-matched, {other_key}=None sibling scenario "
                f"for {scenario!r} -- add one to config/scenarios.yml"
            )
        return matches[0]

    river_only = _find("river_rp_yr", "surge_rp_yr", river_rp_yr)
    coastal_only = _find("surge_rp_yr", "river_rp_yr", surge_rp_yr)
    return river_only, coastal_only


# What to run: CLI override, else just the "default" scenario.
#   snakemake build --config target_scenarios="['baseline','coast_100']"
SCENARIOS = list(config.get("target_scenarios", ["default"]))
_unknown = sorted(set(SCENARIOS) - set(SCENARIO_DEFS))
if _unknown:
    raise ValueError(f"target_scenarios {_unknown} not defined in {config['scenarios_file']}")

# Basin-level restart filename: run_spinup (14) always runs at mean river
# discharge + calm sea for spinup_days, so the file is shared by every scenario.
# Defined here because rules 13, 14 and 16 all need it.
from datetime import datetime as _datetime, timedelta as _timedelta

_tref       = _datetime.strptime(config["sfincs"]["simulation"]["tref"], "%Y-%m-%d %H:%M:%S")
_spinup_end = _tref + _timedelta(days=config["sfincs"]["spinup"]["spinup_days"])
RST_FNAME   = f"sfincs.{_spinup_end.strftime('%Y%m%d.%H%M%S')}.rst"



# ---- Adaptation strategies: families x scales, see src.adaptation_strategies
with open(config["adaptation"]["strategies_file"]) as _f:
    STRATEGY_DEFS_RAW = _yaml.safe_load(_f) or {}

# Strategy measure files live under the catalogue's raw-data root (site setting
# raw_data_root), except `auto/<name>` lines, built per basin by rule
# build_adaptation_lines.
ADAPT_CATALOGUE_ROOT = CATALOGUE["meta"]["root"]

with open(config["adaptation"]["measures_file"]) as _f:
    _MEASURES_RAW = _yaml.safe_load(_f) or {}

_STRATEGY_EXPANSION = expand_strategies(
    STRATEGY_DEFS_RAW,
    _MEASURES_RAW,
    SCENARIO_DEFS,
    strategies_file=config["adaptation"]["strategies_file"],
    measures_file=config["adaptation"]["measures_file"],
    scenarios_file=config["scenarios_file"],
)
MEASURES_DEFS = _STRATEGY_EXPANSION.measures
STRATEGY_FAMILIES = _STRATEGY_EXPANSION.families
STRATEGY_DEFS = _STRATEGY_EXPANSION.defs
strategy_def_for = _STRATEGY_EXPANSION.strategy_def_for


def adaptation_lines_dir(basin_id):
    """Return the folder rule build_adaptation_lines writes a basin's lines to."""
    return results_path(f"{basin_id}/adaptation_lines")

def strategy_measure_input_paths(strategy, basin_id):
    """Return the files a strategy's measures reference in a basin, for rule inputs."""
    paths = []
    for _measure_params in STRATEGY_DEFS[strategy]["measures"].values():
        for _k, _v in _measure_params.items():
            if _k == "locations" and isinstance(_v, str):
                paths.append(location_path(_v, ADAPT_CATALOGUE_ROOT, adaptation_lines_dir(basin_id)))
    return paths

def basin_strategy_def(wildcards):
    """Return strategy_def_for() with `auto/` locations pointing at the basin's lines."""
    return with_auto_locations(
        strategy_def_for(wildcards.strategy, wildcards.scenario),
        adaptation_lines_dir(wildcards.basin_id),
    )


def water_retention_excess_volume_input(wildcards):
    """Return baseline_excess_volume.json if the strategy uses water retention, else [].

    Conditional so other strategies do not require rule attribution_mask and
    its river-only/coastal-only sibling scenarios, which not every scenario has.
    """
    if "water_retention" in STRATEGY_DEFS[wildcards.strategy]["measures"]:
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


def build_sfincs_params():
    """Return the params shared by rules build_sfincs and adapt_build_forcing_pre."""
    return dict(
        resolution_m       = lambda wildcards: grid_resolution_m(wildcards.basin_id),
        tref               = config["sfincs"]["simulation"]["tref"],
        dtmapout           = config["sfincs"]["simulation"]["dtmapout"],
        dtmaxout           = config["sfincs"]["simulation"]["dtmaxout"],
        dthisout           = config["sfincs"]["simulation"]["dthisout"],
        storevelmax        = config["sfincs"]["simulation"]["storevelmax"],
        storetwet          = config["sfincs"]["simulation"]["storetwet"],
        include_rstart     = config["sfincs"]["spinup"]["enabled"],
        spinup_days        = config["sfincs"]["spinup"]["spinup_days"],
        rst_fname          = RST_FNAME,
        # river_only boundary 0.5 m below terrain.gebco_max_depth_m (the deepest
        # clamped bed), so the coastal boundary is always dry.
        river_only_flat_level_m = -(config["terrain"]["gebco_max_depth_m"] + 0.5),
        # Derived from which of the scenario's surge_rp_yr/river_rp_yr are set.
        forcing_mode       = lambda wildcards: scenario_params(wildcards.scenario)["mode"],
        river_rp_yr        = lambda wildcards: scenario_params(wildcards.scenario)["river_rp_yr"],
        surge_rp_yr        = lambda wildcards: scenario_params(wildcards.scenario)["surge_rp_yr"],
        compound_lag_hr    = config["sfincs"]["boundary_setup"]["compound"]["lag_hr"],
        # Real GloFAS flood event or synthetic wave; changing it only rebuilds scenarios.
        river_event        = config["boundary_forcings"]["river"]["event_hydrograph"],
        # Target global-mean SLR (m) × surge_forcing.nc's dimensionless
        # slr_fingerprint; per-scenario and applied here only.
        slr_enabled        = config["boundary_forcings"]["surge"]["slr"]["enabled"],
        slr_m              = lambda wildcards: scenario_params(wildcards.scenario)["slr_m"],
        # SLR ramps in over the event's first hours (the shared spin-up has none).
        slr_ramp_hr        = config["sfincs"]["boundary_setup"]["slr_ramp_hr"],
        flat_boundary_point_spacing_m = config["sfincs"]["boundary_setup"]["flat_boundary_point_spacing_m"],
        water_level_buffer_m = config["sfincs"]["boundary_setup"]["water_level_buffer_m"],
    )


def run_event_params():
    """Return the params shared by rules run_event and adapt_run_event_pre."""
    return dict(
        sfincs_exe                 = sfincs_exe_path,
        timeout_s                  = config["sfincs"]["simulation"]["timeout_s"],
        min_inundation_depth_m     = config["sfincs"]["sanity_checks"]["min_inundation_depth_m"],
        animation_fps              = config["sfincs"]["sanity_checks"]["animation_fps"],
    )
