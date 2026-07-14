import os

from src.io import load_catalogue, catalogue_entry,read_geometry, raw_input_path
from src.io import general_path as _gen_path
import geopandas as gpd

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

wildcard_constraints:
    basin_id = r"\d+"

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

_FORCING_MODES = ("compound", "coastal_only", "river_only")
if config["boundary_setup"]["mode"] not in _FORCING_MODES:
    raise ValueError(
        f"boundary_setup.mode = {config['boundary_setup']['mode']!r} is not valid "
        f"— must be one of {_FORCING_MODES}"
    )
