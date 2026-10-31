# Shared setup included first: paths, basins, grid/scenario/strategy config, wildcards.
import json
import os

from src.io import load_catalogue, catalogue_entry,read_geometry, raw_input_path, site_setting, require_keys
from src.io import general_path as _gen_path
from src.river_forcing import derive_forcing_mode
from src.hydromt_sfincs_patch import MASK_FILES, SUBGRID_TABLE_FILES
from src.adaptation_strategies import (
    ADAPTATION_LINE_NAMES, FILE_PARAMS, expand_strategies, location_path, resolve_locations,
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

def catalogue_region_paths(name):
    """Return one raw-data path per region of a catalogue entry with a {region} file_path."""
    return [raw_input_path(CATALOGUE, name).format(region=r) for r in catalogue_entry(CATALOGUE, name)["regions"]]

def _existing_raw(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"raw input(s) missing under raw_data_root: {missing}; see docs/data.md")
    return paths

# Snakemake drops a job whose raw input is missing when its outputs exist, even
# under --forceall, and silently reuses those outputs; an input function that
# raises makes the job fail at DAG construction instead.
def raw_input(name):
    """Return an input function for a catalogue entry's path; it fails if the path is missing."""
    path = catalogue_path(name)
    return lambda wildcards: _existing_raw([path])[0]

def raw_region_inputs(name):
    """Return an input function for catalogue_region_paths(name); it fails if any is missing."""
    paths = catalogue_region_paths(name)
    return lambda wildcards: _existing_raw(paths)

# Manual SWORD reach widths (src.sword.apply_width_corrections).
SWORD_WIDTH_CORRECTIONS = "config/sword_width_corrections.csv"

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


# ── figures ──
# Core figures in {basin_id}/figures/; diagnostics (config analysis: true only)
# in {basin_id}/figures/analysis/, drawn by the analysis rules (19) from
# finished outputs and diagnostic files. Names: NN_stage_quantity[__run_label].
if not isinstance(config["analysis"], bool):
    raise ValueError(
        f"config analysis must be true or false, got {config['analysis']!r} "
        f"(on the command line: --config analysis=True)"
    )
ANALYSIS = config["analysis"]

def figure_path(name):
    """Return the path of a core figure of {basin_id}."""
    return results_path(f"{{basin_id}}/figures/{name}")

def analysis_figure_path(name):
    """Return the path of an analysis figure of {basin_id}."""
    return results_path(f"{{basin_id}}/figures/analysis/{name}")

def diagnostics_path(name):
    """Return the path of a diagnostic data file of {basin_id}, drawn by the analysis rules."""
    return results_path(f"{{basin_id}}/preprocessing_inputs/diagnostics/{{basin_id}}_{name}")


def run_label(scenario=None, method=None, strategy=None):
    """Return the label naming a run in figure names and titles.

    calib_r1 / spinup are passed as scenario; adaptation runs are
    {scenario}__{method}_{strategy}, so they sort next to their scenario.
    """
    if strategy is None:
        return str(scenario)
    return f"{scenario}__{method}_{strategy}"


CALIB_ROUNDS = [1, 2] if config["river_processing"]["river_depth_modelling"]["verify_crests"] else [1]

def calibration_round_figures():
    """Return rule 11's outputs of each SFINCS round (calib_r1, calib_r2 with verify_crests).

    Forcing and max-depth figures, the round's map (read by the analysis
    figures) and its water level at one centreline cell per reach.
    """
    outputs = {}
    for r in CALIB_ROUNDS:
        outputs[f"plot_forcing_r{r}"] = figure_path(f"20_forcing_timeseries__calib_r{r}.png")
        outputs[f"plot_max_depth_r{r}"] = figure_path(f"40_run_max_flood_depth__calib_r{r}.png")
        outputs[f"water_level_r{r}"] = diagnostics_path(f"calibration_water_level_r{r}.csv")
        outputs[f"map_r{r}"] = results_path(
            f"{{basin_id}}/preprocessing_inputs/depth_crest_calibration/round{r}/sfincs_map.nc"
        )
    return outputs


def calibration_subgrid_files():
    """Return rule 11's round-1 subgrid table and mask files, reused by rule 12a."""
    return [
        results_path(f"{{basin_id}}/preprocessing_inputs/depth_crest_calibration/round1/{name}")
        for name in SUBGRID_TABLE_FILES + MASK_FILES
    ]


# ── per-rule threads (config resources) ──
_RES_CFG = config["resources"]

def rule_threads(rule):
    """Return a rule's threads from config; "all" is the whole --cores budget."""
    value = _RES_CFG["threads"][rule]
    return workflow.cores if value == "all" else int(value)


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
    """Return the (river, coastal) scenarios the attribution mask compares `scenario` with.

    Each keeps one driver at `scenario`'s own level and puts the other at mean
    conditions, with the same slr_m (SLR is part of the sea, not a driver):
    river: river_rp_yr as `scenario`, surge_rp_yr Tide; coastal: surge_rp_yr as
    `scenario`, river_rp_yr Mean. compound_100_03SLR pairs with river_100_03SLR
    and coast_100_03SLR; a single-driver scenario pairs with itself and the
    mean-conditions run. The other driver is never null: a null sea is a flat
    boundary far below sea level, which removes the sea's backwater from the
    river run. `scenario` itself is preferred when it matches. Raises ValueError
    if scenarios.yml has no match; counterparts are never guessed.
    """
    s0 = SCENARIO_DEFS[scenario]
    slr_m = s0.get("slr_m", 0.0)

    def _norm(v):
        return v.strip().lower() if isinstance(v, str) else v

    def _find(want):
        def ok(s):
            return s.get("slr_m", 0.0) == slr_m and all(
                _norm(s.get(k)) == _norm(v) for k, v in want.items()
            )
        if ok(s0):
            return scenario
        matches = [n for n, s in SCENARIO_DEFS.items() if ok(s)]
        if not matches:
            raise ValueError(
                f"No attribution counterpart {want} with slr_m={slr_m} for "
                f"{scenario!r} -- add one to config/scenarios.yml"
            )
        return matches[0]

    river = _find({"river_rp_yr": s0["river_rp_yr"], "surge_rp_yr": "tide"})
    coastal = _find({"surge_rp_yr": s0["surge_rp_yr"], "river_rp_yr": "mean"})
    return river, coastal


# What to run: CLI override, else just the "default" scenario.
#   snakemake build --config target_scenarios="['baseline','coast_100']"
SCENARIOS = list(config.get("target_scenarios", ["default"]))
_unknown = sorted(set(SCENARIOS) - set(SCENARIO_DEFS))
if _unknown:
    raise ValueError(f"target_scenarios {_unknown} not defined in {config['scenarios_file']}")

# Basin-level restart file: run_spinup (13a) runs at the spin-up scenario's
# mean conditions (mean river discharge, calm sea + its SLR) for the basin's
# spin-up length and renames SFINCS's time-stamped restart to this fixed name,
# shared by every scenario of the basin.
RST_FNAME = "sfincs.spinup.rst"

_SPINUP_CFG = config["sfincs"]["spinup"]
_SPINUP_OVERRIDES = {
    int(_b): float(_d) for _b, _d in (_SPINUP_CFG.get("spinup_overrides_days") or {}).items()
}

def spinup_days(basin_id):
    """Return a basin's spin-up length (days), which is also its forcing lead-in (lead_days)."""
    return _SPINUP_OVERRIDES.get(int(basin_id), _SPINUP_CFG["spinup_days"])

def spinup_slr_m():
    """Return the SLR (m, global mean) of the spin-up scenario; 0 with SLR disabled."""
    if not config["boundary_forcings"]["surge"]["slr"]["enabled"]:
        return 0.0
    return float(scenario_params(_SPINUP_CFG["scenario"])["slr_m"])

_spinup_scenario = SCENARIO_DEFS.get(_SPINUP_CFG["scenario"])
if _spinup_scenario is None:
    raise ValueError(f"sfincs.spinup.scenario {_SPINUP_CFG['scenario']!r} not defined in {config['scenarios_file']}")
if str(_spinup_scenario.get("river_rp_yr")).lower() != "mean" or str(_spinup_scenario.get("surge_rp_yr")).lower() not in ("tide", "none"):
    raise ValueError(
        f"sfincs.spinup.scenario {_SPINUP_CFG['scenario']!r} must be a mean-conditions scenario "
        f"(river_rp_yr: Mean, surge_rp_yr: Tide or null); the spin-up holds the sea calm (tide-averaged)"
    )

_COMPOUND_CFG = config["sfincs"]["boundary_setup"]["compound"]
_LAG_OVERRIDES = {int(_b): float(_h) for _b, _h in (_COMPOUND_CFG.get("lag_overrides_hr") or {}).items()}

def compound_lag_hr(basin_id):
    """Return a basin's river-vs-surge shift (h) of compound scenarios; > 0 delays the river."""
    return _LAG_OVERRIDES.get(int(basin_id), float(_COMPOUND_CFG["lag_hr"]))

_AFTER_PEAK_OVERRIDES = {
    int(_b): _h
    for _b, _h in (config["sfincs"]["boundary_setup"].get("event_after_peak_overrides_hr") or {}).items()
}

def event_after_peak_hr(basin_id):
    """Return how long (h) a basin's event runs go on after the last design peak."""
    return _AFTER_PEAK_OVERRIDES.get(int(basin_id), config["sfincs"]["boundary_setup"]["event_after_peak_hr"])

# The event run starts from the restart at spinup_days on the forcing's own time
# axis, so the forcing's calm lead-in must last exactly as long: a shorter
# lead-in cuts off the start of the event window and starts the run from a calm
# state on a rising boundary.
if (
    config["sfincs"]["spinup"]["enabled"]
    and float(config["boundary_forcings"]["lead_days"]) != float(config["sfincs"]["spinup"]["spinup_days"])
):
    raise ValueError(
        f"boundary_forcings.lead_days ({config['boundary_forcings']['lead_days']}) must equal "
        f"sfincs.spinup.spinup_days ({config['sfincs']['spinup']['spinup_days']}): the event run "
        f"starts at the spin-up restart, at the end of the forcing's lead-in"
    )



# ---- Adaptation strategies: families x scales, see src.adaptation_strategies
with open(config["adaptation"]["strategies_file"]) as _f:
    STRATEGY_DEFS_RAW = _yaml.safe_load(_f) or {}

def adaptation_data_root():
    """Return the root of the strategies' measure files (site setting adaptation_data_root).

    Called only by adapt jobs, so the other targets do not require the setting.
    `auto/<name>` lines are built per basin by rule build_adaptation_lines instead.
    """
    return site_setting("adaptation_data_root")

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
            if _k in FILE_PARAMS and isinstance(_v, str):
                paths.append(location_path(_v, adaptation_data_root(), adaptation_lines_dir(basin_id), basin_id))
    return paths

def basin_strategy_def(wildcards):
    """Return strategy_def_for() with locations resolved for the basin."""
    return resolve_locations(
        strategy_def_for(wildcards.strategy, wildcards.scenario),
        adaptation_lines_dir(wildcards.basin_id),
        wildcards.basin_id,
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
        basin_id           = lambda wildcards: wildcards.basin_id,
        resolution_m       = lambda wildcards: grid_resolution_m(wildcards.basin_id),
        tref               = config["sfincs"]["simulation"]["tref"],
        dtmapout           = config["sfincs"]["simulation"]["dtmapout"],
        dtmaxout           = config["sfincs"]["simulation"]["dtmaxout"],
        dthisout           = config["sfincs"]["simulation"]["dthisout"],
        storevelmax        = config["sfincs"]["simulation"]["storevelmax"],
        storetwet          = config["sfincs"]["simulation"]["storetwet"],
        include_rstart     = config["sfincs"]["spinup"]["enabled"],
        spinup_days        = lambda wildcards: spinup_days(wildcards.basin_id),
        spinup_slr_m       = spinup_slr_m(),
        rst_fname          = RST_FNAME,
        # river_only boundary 0.5 m below terrain.gebco_max_depth_m (the deepest
        # clamped bed), so the coastal boundary is always dry.
        river_only_flat_level_m = -(config["terrain"]["gebco_max_depth_m"] + 0.5),
        # Derived from which of the scenario's surge_rp_yr/river_rp_yr are set.
        forcing_mode       = lambda wildcards: scenario_params(wildcards.scenario)["mode"],
        river_rp_yr        = lambda wildcards: scenario_params(wildcards.scenario)["river_rp_yr"],
        surge_rp_yr        = lambda wildcards: scenario_params(wildcards.scenario)["surge_rp_yr"],
        compound_lag_hr    = lambda wildcards: compound_lag_hr(wildcards.basin_id),
        event_after_peak_hr = lambda wildcards: event_after_peak_hr(wildcards.basin_id),
        # Real GloFAS flood event or synthetic wave; changing it only rebuilds scenarios.
        river_event        = config["boundary_forcings"]["river"]["event_hydrograph"],
        # Target global-mean SLR (m) × surge_forcing.nc's dimensionless
        # slr_fingerprint; per-scenario and applied here only.
        slr_enabled        = config["boundary_forcings"]["surge"]["slr"]["enabled"],
        slr_m              = lambda wildcards: scenario_params(wildcards.scenario)["slr_m"],
        # SLR ramps over the event's first hours from the spin-up's SLR to the scenario's.
        slr_ramp_hr        = config["sfincs"]["boundary_setup"]["slr_ramp_hr"],
        flat_boundary_point_spacing_m = config["sfincs"]["boundary_setup"]["flat_boundary_point_spacing_m"],
        water_level_buffer_m = config["sfincs"]["boundary_setup"]["water_level_buffer_m"],
    )


def map_figure_inputs():
    """Return the inputs every flood map figure draws on: domain, land mask, rivers."""
    return dict(
        spec_basins_meta    = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg         = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        land_mask_on_grid   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
        clean_river_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
    )


def run_event_params():
    """Return the params shared by rules run_event and adapt_run_event_pre."""
    return dict(
        basin_id                   = lambda wildcards: wildcards.basin_id,
        sfincs_exe                 = sfincs_exe_path,
        timeout_s                  = config["sfincs"]["simulation"]["timeout_s"],
        min_inundation_depth_m     = config["sfincs"]["sanity_checks"]["min_inundation_depth_m"],
        animation_fps              = config["sfincs"]["sanity_checks"]["animation_fps"],
    )
