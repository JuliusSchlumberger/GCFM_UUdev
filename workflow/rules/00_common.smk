from src.io import load_catalogue, catalogue_entry,read_geometry, raw_input_path
from src.io import general_path as _gen_path
import geopandas as gpd

CATALOGUE = load_catalogue(config["data_catalogue"])

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

_FORCING_MODES = ("compound", "coastal_only", "river_only")
if config["boundary_setup"]["mode"] not in _FORCING_MODES:
    raise ValueError(
        f"boundary_setup.mode = {config['boundary_setup']['mode']!r} is not valid "
        f"— must be one of {_FORCING_MODES}"
    )
