import json
import os

from src.io import load_catalogue, catalogue_entry,read_geometry, raw_input_path
from src.io import general_path as _gen_path
from src.river_forcing import derive_forcing_mode
import geopandas as gpd

import re as _re
import yaml as _yaml

CATALOGUE = load_catalogue(config["data_catalogue"])

# ── local machine overrides ──────────────────────────────────────────────────
# results_dir, the raw-data catalogue root, and the SFINCS executable path
# are inherently machine-specific -- hand-editing them in config.yml /
# data_catalogue.yml (both git-tracked, shared files) means every `git pull`
# either overwrites your own local paths with whoever committed last, or
# creates a merge conflict. Setting these three environment variables once
# (see CONTRIBUTING.md "Local machine paths") overrides the committed values
# without ever touching a tracked file again; unset, the committed defaults
# below are used as before.
if os.environ.get("GCFM_RESULTS_DIR"):
    config["results_dir"] = os.environ["GCFM_RESULTS_DIR"]
if os.environ.get("GCFM_RAW_DATA_ROOT"):
    CATALOGUE["meta"]["root"] = os.environ["GCFM_RAW_DATA_ROOT"]
if os.environ.get("GCFM_SFINCS_EXE"):
    config["sfincs"]["simulation"]["sfincs_exe"] = os.environ["GCFM_SFINCS_EXE"]

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
    if _srp is not None and _srp not in _SURGE_RPS:
        raise ValueError(f"{_name}: surge_rp must be a COAST-RP tabulated value {_SURGE_RPS}")
    if _rrp is not None and not _RIVER_RP_MIN <= _rrp <= _RIVER_RP_MAX:
        raise ValueError(f"{_name}: river_rp must be in [{_RIVER_RP_MIN}, {_RIVER_RP_MAX}] yr")

def scenario_params(name):
    """-> dict(mode, surge_rp, river_rp); mode via derive_forcing_mode."""
    s = SCENARIO_DEFS[name]
    river_rp, surge_rp = s.get("river_rp"), s.get("surge_rp")
    try:
        mode = derive_forcing_mode(river_rp, surge_rp)
    except ValueError as e:
        raise ValueError(f"scenario {name!r}: {e}") from e
    return {"mode": mode, "surge_rp": surge_rp, "river_rp": river_rp}

def attribution_counterparts(scenario):
    """-> (river_only_scenario, coastal_only_scenario): the two sibling
    single-driver scenario names whose own RP matches `scenario`'s own
    river_rp/surge_rp exactly (river_rp=X & surge_rp=None / surge_rp=Y &
    river_rp=None) -- e.g. coast_500 (surge=500, river=2) pairs with
    river_only (river=2) and coast_only_500 (surge=500); river_500
    (surge=2, river=500) pairs with river_only_500 and coast_only (surge=2).
    Raises if scenarios.yml has no such sibling defined yet (e.g. coast_100
    has no coast_only_100 counterpart) -- confirmed with the user rather
    than silently guessing a mismatched RP."""

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

# Basin-level (not scenario-level) restart filename: run_spinup (14) always
# runs at a fixed RP=1/spinup_days, entirely independent of any scenario's
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

ADAPT_CATALOGUE_ROOT = STRATEGY_DEFS_RAW.get("meta", {}).get("root", ".")
STRATEGY_DEFS = {k: v for k, v in STRATEGY_DEFS_RAW.items() if k != "meta"}
if not STRATEGY_DEFS:
    raise ValueError(f"{config['adaptation']['strategies_file']} defines no strategies (besides 'meta')")

for _name in STRATEGY_DEFS:
    if not _re.fullmatch(r"[A-Za-z0-9_-]+", _name):
        raise ValueError(f"strategy name {_name!r} invalid (use letters/digits/_/- only)")

with open(config["adaptation"]["measures_file"]) as _f:
    MEASURES_DEFS = (_yaml.safe_load(_f) or {}).get("adaptation_measures", {})

def adaptation_input_path(rel):
    """Resolve a strategy measure's locations/dep_subgrid string param
    against adaptation_strategies.yml's own meta.root."""
    return str(Path(ADAPT_CATALOGUE_ROOT) / rel)

def strategy_measure_input_paths(strategy):
    """Raw-data files a strategy's measures reference, as extra `input:`
    so editing e.g. adaptation/levee.geojson triggers a rerun."""
    paths = []
    for _measure_params in STRATEGY_DEFS[strategy]["measures"].values():
        for _k, _v in _measure_params.items():
            if _k in ("locations", "dep_subgrid") and isinstance(_v, str):
                paths.append(adaptation_input_path(_v))
    return paths


# Opt-in only -- no default strategy set (unlike scenario's "default"):
#   snakemake adapt --config target_strategies="['retreat']"
STRATEGIES = list(config.get("target_strategies", []))
_unknown = sorted(set(STRATEGIES) - set(STRATEGY_DEFS))
if _unknown:
    raise ValueError(f"target_strategies {_unknown} not defined in {config['adaptation']['strategies_file']}")

# Both methods run by default, unless a strategy is selected:
#   snakemake adapt --config target_strategies="['retreat']" target_methods="['pre']"
METHODS = list(config.get("target_methods", ["pre", "post"]))
_unknown_methods = sorted(set(METHODS) - {"pre", "post"})
if _unknown_methods:
    raise ValueError(f"target_methods {_unknown_methods} invalid -- only 'pre'/'post' supported")


wildcard_constraints:
    basin_id = r"\d+",
    scenario = r"|".join(sorted(SCENARIO_DEFS)),
    strategy = r"|".join(sorted(STRATEGY_DEFS)) if STRATEGY_DEFS else r"(?!)",
