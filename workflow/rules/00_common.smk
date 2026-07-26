import os

from src.io import load_catalogue, catalogue_entry,read_geometry, raw_input_path
from src.io import general_path as _gen_path
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


if config["sfincs"]["grid"]["quadtree"]["enabled"] and not config["sfincs"]["subgrid"]["enabled"]:
    raise ValueError(
        "sfincs.grid.quadtree.enabled requires sfincs.subgrid.enabled = true "
        "(quadtree postprocessing relies on the subgrid dep_subgrid.tif reference raster)"
    )

if (
    config["river_processing"]["burn_rivers"]["enabled"]
    and not config["river_processing"]["conditioning"]["enabled"]
):
    raise ValueError(
        "river_processing.burn_rivers.enabled requires "
        "river_processing.conditioning.enabled = true (burn_rivers burns the "
        "zbed_anchors.gpkg profile computed by rule burn_river_bed, which "
        "itself requires conditioning to be enabled)"
    )

# NOTE: KL removed
# _FORCING_MODES = ("compound", "coastal_only", "river_only")
# if config["boundary_setup"]["mode"] not in _FORCING_MODES:
#     raise ValueError(
#         f"boundary_setup.mode = {config['boundary_setup']['mode']!r} is not valid "
#         f"— must be one of {_FORCING_MODES}"
#     )

# ── scenario axis ─────────────────────────────────────────────────────────────
# Scenarios are defined by name in scenarios_file (config/scenarios.yml) and
# selected as the {scenario} wildcard. The reserved name "default" is not in
# that file: it replays config.yml's own boundary_setup settings (mode,
# design_rp_river_yr, surge.return_period) and is what plain `snakemake build`
# runs when no target_scenarios is given.

_SURGE_RPS = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000)    # COAST-RP tabulated; no interpolation
_RIVER_RP_MIN, _RIVER_RP_MAX = 2, 1000                     # log-interpolated from discharge_rp_table

with open(config["scenarios_file"]) as _f:
    SCENARIO_DEFS = _yaml.safe_load(_f) or {}

for _name, _s in SCENARIO_DEFS.items():
    if _name == "default" or not _re.fullmatch(r"[A-Za-z0-9_-]+", _name):
        raise ValueError(f"scenario name {_name!r} invalid ('default' is reserved; "
                         "use letters/digits/_/- only)")
    _srp, _rrp = _s.get("surge_rp"), _s.get("river_rp")
    if _srp is not None and _srp not in _SURGE_RPS:
        raise ValueError(f"{_name}: surge_rp must be a COAST-RP tabulated value {_SURGE_RPS}")
    if _rrp is not None and not _RIVER_RP_MIN <= _rrp <= _RIVER_RP_MAX:
        raise ValueError(f"{_name}: river_rp must be in [{_RIVER_RP_MIN}, {_RIVER_RP_MAX}] yr")

def scenario_params(name):
    """-> dict(mode, surge_rp, river_rp); None RP = mean conditions."""
    if name == "default":
        return {
            "mode":     config["boundary_setup"]["mode"],
            "surge_rp": config["boundary_forcings"]["surge"]["return_period"],
            "river_rp": config["boundary_setup"]["design_rp_river_yr"],
        }
    s = SCENARIO_DEFS[name]
    return {"mode": "compound", "surge_rp": s.get("surge_rp"), "river_rp": s.get("river_rp")}

# What to run: CLI override, else just the config.yml-driven default.
#   snakemake build --config target_scenarios="['baseline','coast_100']"
SCENARIOS = list(config.get("target_scenarios", ["default"]))
_unknown = sorted(set(SCENARIOS) - set(SCENARIO_DEFS) - {"default"})
if _unknown:
    raise ValueError(f"target_scenarios {_unknown} not defined in {config['scenarios_file']}")


wildcard_constraints:
    basin_id = r"\d+",
    scenario = r"|".join(sorted(set(SCENARIO_DEFS) | {"default"}))
