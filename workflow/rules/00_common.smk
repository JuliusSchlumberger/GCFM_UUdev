import json
import os

from src.io import load_catalogue, catalogue_entry,read_geometry, raw_input_path, local_path
from src.io import general_path as _gen_path
from src.river_forcing import derive_forcing_mode
import geopandas as gpd

import re as _re
import yaml as _yaml

CATALOGUE = load_catalogue(config["data_catalogue"])

# ── local machine paths ──────────────────────────────────────────────────────
# results_dir, the raw-data catalogue root, and the SFINCS executable path
# are inherently machine-specific, so none of them is committed in config.yml
# / data_catalogue.yml (both git-tracked, shared files) -- they are read from
# three environment variables instead. Set them once in PowerShell (see
# CONTRIBUTING.md "Local machine paths"), then restart your terminal:
#   [Environment]::SetEnvironmentVariable("GCFM_RESULTS_DIR", "D:\your\results\path", "User")
#   [Environment]::SetEnvironmentVariable("GCFM_RAW_DATA_ROOT", "D:\your\raw_data\path", "User")
#   [Environment]::SetEnvironmentVariable("GCFM_SFINCS_EXE", "C:\path\to\sfincs.exe", "User")
# GCFM_RAW_DATA_ROOT is applied inside load_catalogue() itself (above), so
# scripts that load the catalogue on their own get the same root.
# Forward slashes: results_path() joins with "/", and a Windows-style value
# ("D:\results") would give mixed separators that Snakemake does not always
# match between one rule's output and another rule's input.
config["results_dir"] = Path(local_path("GCFM_RESULTS_DIR")).as_posix()

def sfincs_exe_path(wildcards=None):
    """SFINCS executable (GCFM_SFINCS_EXE) -- resolved lazily as a rule param
    so only the rules that actually run the solver require it; everything up
    to and including build_sfincs works without a SFINCS binary."""
    return local_path("GCFM_SFINCS_EXE")

def catalogue_path(name):
    return raw_input_path(CATALOGUE, name)

def general_path(name):
    return _gen_path(CATALOGUE, name)

def list_basins(name):
    """Discover available basisn from the delta polygons shapefile"""
    dataset = catalogue_entry(CATALOGUE, name)
    _delta_shp = catalogue_path(name)
    attribute = dataset["attributes"][0]["name"]

    _deltas = read_geometry(_delta_shp)
    _delta_ids = _deltas[attribute]
    return sorted(_delta_ids.astype(int).to_list())

BASINS = list_basins("delta_polygons")

# Optional CLI override to restrict a run to specific basin(s), e.g. for
# testing a single delta without re-running the full multi-basin fleet:
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
    return f"{RESULTS_DIR}/{pattern}"


# ── SFINCS main-grid resolution ───────────────────────────────────────────────
# One fixed dx (m) for every basin (sfincs.grid.resolution_m), optionally
# overridden per basin id (sfincs.grid.resolution_overrides_m) -- e.g. to
# keep a very large delta within the memory of the machine it is built on.
# The subgrid pixel size is dx / sfincs.subgrid.nr_subgrid_pixels.
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
    """-> main-grid dx (m) for one basin: its own override, else the default."""
    return float(_GRID_RES_OVERRIDES.get(int(basin_id), _GRID_CFG["resolution_m"]))

# Grid orientation: axis-aligned, or the minimum rotated rectangle around the
# delta polygon (sfincs.grid.rotated), optionally per basin
# (sfincs.grid.rotated_overrides). See src/grid.py.
_GRID_ROTATED_OVERRIDES = {
    int(_b): _r for _b, _r in (_GRID_CFG.get("rotated_overrides") or {}).items()
}
for _label, _r in [("rotated", _GRID_CFG.get("rotated", False))] + [
    (f"rotated_overrides[{_b}]", _r) for _b, _r in _GRID_ROTATED_OVERRIDES.items()
]:
    if not isinstance(_r, bool):
        raise ValueError(f"sfincs.grid.{_label} must be true or false, got {_r!r}")

def grid_rotated(basin_id):
    """-> whether one basin's grid is rotated: its own override, else the default."""
    return bool(_GRID_ROTATED_OVERRIDES.get(int(basin_id), _GRID_CFG.get("rotated", False)))


# ── storm-tide event window ───────────────────────────────────────────────────
# Half-length (h) of the storm-tide event around its peak
# (boundary_forcings.surge.hydrograph.window_hr): the length of the surge
# shapes (rule storm_tide_hydrographs) and of the event the boundary carries
# (rule get_boundary_forcings). One default, optionally overridden per basin
# id (window_overrides_hr) -- surges last hours on some coasts and days on
# others.
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
    """-> storm-tide event half-window (h) for one basin: its own override, else the default."""
    return float(_HG_WINDOW_OVERRIDES.get(int(basin_id), _HG_CFG["window_hr"]))


# ── scenario axis ─────────────────────────────────────────────────────────────
# Scenarios (incl. the reserved name "default", used when no target_scenarios
# is given) are defined by name in scenarios_file (config/scenarios.yml) and
# selected as the {scenario} wildcard. Each scenario's own forcing_mode is
# DERIVED from which of its RPs are set (see scenario_params below) rather
# than declared separately -- there is no standalone "mode" setting anywhere.

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
    """-> dict(mode, surge_rp, river_rp, discharge_multiplier, slr_m); mode
    via derive_forcing_mode. discharge_multiplier is per-scenario (default
    1.0) -- e.g. river_500/river_only_500/compound_500 can be scaled up to
    reach a genuinely flood-inducing river discharge without also inflating
    coast_500's own small RP=2 river component. slr_m is per-scenario
    (default 0.0 = no SLR) for the same reason: it's a single scalar read
    ONLY by rule build_sfincs/adapt_build_forcing_pre (never rule
    get_boundary_forcings), so keeping it per-scenario -- rather than one
    shared config.yml value -- means changing it only invalidates the
    scenario(s) whose own entry changed, not every already-built scenario's
    sfincs.inp."""
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
    """-> (river_scenario, coastal_scenario) the attribution mask compares
    `scenario` against: each keeps ONE driver at `scenario`'s own event level
    and puts the other at mean conditions (surge_rp Tide / river_rp Mean),
    with the same slr_m (SLR is part of the sea, not a separate driver):

      river side   : river_rp = scenario's river_rp, surge_rp = Tide
      coastal side : surge_rp = scenario's surge_rp, river_rp = Mean

    So coast_100_03SLR (100 / Mean) pairs with the mean-conditions run
    (Tide / Mean) and itself; river_100_03SLR (Tide / 100) with itself and
    the mean-conditions run; compound_100_03SLR (100 / 100) with
    river_100_03SLR and coast_100_03SLR. Cells flooded in both (class 3)
    are then flooded by either driver alone -- for a single-driver scenario,
    already under mean conditions. The other driver is never null: a null
    sea is a flat boundary far below sea level (13_build_sfincs.py,
    river_only), which removes the sea's backwater from the river run.
    `scenario` itself is preferred when it matches. Raises if scenarios.yml
    has no matching scenario rather than guessing a mismatched one."""

    s0 = SCENARIO_DEFS[scenario]
    slr = s0.get("slr_m", 0.0)

    def _norm(v):
        return v.strip().lower() if isinstance(v, str) else v

    def _find(want):
        def ok(s):
            return s.get("slr_m", 0.0) == slr and all(
                _norm(s.get(k, 1.0 if k == "discharge_multiplier" else None)) == _norm(v)
                for k, v in want.items()
            )
        if ok(s0):
            return scenario
        matches = [n for n, s in SCENARIO_DEFS.items() if ok(s)]
        if not matches:
            raise ValueError(
                f"No attribution counterpart {want} with slr_m={slr} for "
                f"{scenario!r} -- add one to config/scenarios.yml"
            )
        return matches[0]

    river = _find({"river_rp": s0["river_rp"], "surge_rp": "tide",
                   "discharge_multiplier": s0.get("discharge_multiplier", 1.0)})
    coastal = _find({"surge_rp": s0["surge_rp"], "river_rp": "mean"})
    return river, coastal


# What to run: CLI override, else just the "default" scenario.
#   snakemake build --config target_scenarios="['baseline','coast_100']"
SCENARIOS = list(config.get("target_scenarios", ["default"]))
_unknown = sorted(set(SCENARIOS) - set(SCENARIO_DEFS))
if _unknown:
    raise ValueError(f"target_scenarios {_unknown} not defined in {config['scenarios_file']}")

# Basin-level (not scenario-level) restart filename: run_spinup (14) always
# runs at a fixed RP=1 river + calm sea for spinup_days, entirely independent of any scenario's
# own RP, so its restart file -- and this filename -- is the SAME for every
# scenario of a basin. Computed here (00_common.smk, included first) rather
# than in 14_run_spinup.smk itself since rules build_sfincs (13, sets
# rstfile) and run_event (16, reads the restart file) both need it too, and
# Snakemake's include: shares one global namespace regardless of order --
# defining it centrally avoids a fragile "must be included after 14" dependency.
from datetime import datetime as _datetime, timedelta as _timedelta

_tref       = _datetime.strptime(config["sfincs"]["simulation"]["tref"], "%Y-%m-%d %H:%M:%S")
_spinup_end = _tref + _timedelta(days=config["sfincs"]["spinup"]["spinup_days"])
RST_FNAME   = f"sfincs.{_spinup_end.strftime('%Y%m%d.%H%M%S')}.rst"



# ---- Adding adaptation wildcard ----------------------------------------------- (KGL)
with open(config["adaptation"]["strategies_file"]) as _f:
    STRATEGY_DEFS_RAW = _yaml.safe_load(_f) or {}

# Strategy measure files live under the same raw-data root as the catalogue
# (GCFM_RAW_DATA_ROOT) -- adaptation_strategies.yml carries no root of its own.
ADAPT_CATALOGUE_ROOT = CATALOGUE["meta"]["root"]

with open(config["adaptation"]["measures_file"]) as _f:
    MEASURES_DEFS = (_yaml.safe_load(_f) or {}).get("adaptation_measures", {})


# Automating the adaptation strategy scales (KGL)
# adaptation_strategies.yml holds strategy FAMILIES written at scale 1; each
# family expands into one {strategy} per entry of `scales` (0.4 -> <family>_04,
# 0.9 -> <family>_09, 1.0 -> <family>_1), so strategy names, result folders and
# target_strategies stay exactly as before. A param is multiplied by the scale
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

# build strategy names
def _scale_tag(scale):
    """0.4 -> '04', 0.9 -> '09', 1.0 -> '1' (the existing strategy-name suffixes)."""
    return f"{scale:g}".replace(".", "")

# read strategy families which is a set of measures
STRATEGY_FAMILIES = {k: v for k, v in STRATEGY_DEFS_RAW.items() if k not in ("meta", "scales")}
if not STRATEGY_FAMILIES:
    raise ValueError(f"{config['adaptation']['strategies_file']} defines no strategies (besides 'meta'/'scales')")


# Each family (group fo measures) turns into one strategy per scale
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


# Resolve a single parameter value
def _resolve_strategy_param(measure, param, value, scale, event, where):
    """One raw strategy param -> its concrete value for this scale and event
    (or a {"from_forcing": ...} marker for src.adaptation_scaling)."""

    if isinstance(value, dict) and set(value) <= set(_STRATEGY_EVENTS) and value:
        if set(value) != set(_STRATEGY_EVENTS):
            raise ValueError(f"{where}: per-event map must set all of {_STRATEGY_EVENTS}, got {sorted(value)}")
        if event is None:
            raise ValueError(f"{where}: value depends on the event type, but the scenario has no "
                             f"`event:` in {config['scenarios_file']}")
        value = value[event]

    scalable = (measure, param) in _SCALABLE_PARAMS # check if the parameter is scalable
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
    """-> {"measures": {...}} for one strategy in one scenario: per-event values
    picked via the scenario's `event`, scale-flagged params multiplied by the
    strategy's scale, run-time-derived params left as {"from_forcing": ...}
    markers. Passed as params.strategy_def to rules 18a/18b."""
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
    """Resolve a strategy measure's locations/dep_subgrid string param
    against the raw-data root (GCFM_RAW_DATA_ROOT)."""
    return str(Path(ADAPT_CATALOGUE_ROOT) / rel)

def strategy_measure_input_paths(strategy):
    """Raw-data files a strategy's measures reference, as extra `input:`
    so editing e.g. adaptation/levee.geojson triggers a rerun."""
    paths = []
    for _measure_params in STRATEGY_DEFS[strategy]["measures"].values():
        for _k, _v in _measure_params.items():
            if _k in ("locations", "dep_subgrid", "inlet") and isinstance(_v, str):
                paths.append(adaptation_input_path(_v))
    return paths


def water_retention_excess_volume_input(wildcards):
    """rule adapt_apply_pre's own conditional input for baseline_excess_volume.json
    (rule attribution_mask's own output) -- ONLY when this strategy actually
    uses water_retention or its storage-volume sibling water_retention_greening
    (both sized against the same fixed reference). Unlike rule adapt_metrics_post
    (18b), where EVERY postprocessing measure already depends on
    attribution_mask.tif for its own class-based masking, no other preprocessing
    measure touches attribution at all, so this must stay strategy-conditional --
    an unconditional dependency would force every basin x scenario x strategy
    combination through rule attribution_mask (and its own river-only/
    coastal-only counterpart-scenario prerequisite, which not every scenario in
    scenarios.yml has) even when the strategy never uses either measure."""
    if {"water_retention", "water_retention_greening"} & set(STRATEGY_DEFS[wildcards.strategy]["measures"]):
        return results_path(f"{wildcards.basin_id}/runs/{wildcards.scenario}/baseline_excess_volume.json")
    return []


# Opt-in only -- no default strategy set (unlike scenario's "default").
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
