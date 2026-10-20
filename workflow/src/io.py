"""I/O utilities for loading the data catalogue and reading spatial datasets."""

import os
from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import yaml

# Machine-specific paths come from environment variables, never from config.yml
# or data_catalogue.yml (see CONTRIBUTING.md "Local machine paths"), e.g.:
#   [Environment]::SetEnvironmentVariable("GCFM_RESULTS_DIR", "D:\your\results\path", "User")
#   [Environment]::SetEnvironmentVariable("GCFM_RAW_DATA_ROOT", "D:\your\raw_data\path", "User")
#   [Environment]::SetEnvironmentVariable("GCFM_SFINCS_EXE", "C:\path\to\sfincs.exe", "User")
LOCAL_PATH_VARS = {
    "GCFM_RESULTS_DIR": "directory the pipeline outputs are written to",
    "GCFM_RAW_DATA_ROOT": "root directory of the raw input data (data catalogue root)",
    "GCFM_SFINCS_EXE": "path to your local SFINCS executable",
}


def local_path(var: str) -> str:
    """Return the machine-specific path held by environment variable ``var``."""
    value = os.environ.get(var)
    if not value:
        raise OSError(
            f"Environment variable {var} is not set ({LOCAL_PATH_VARS[var]}). "
            f"Set it once in PowerShell and restart your terminal:\n"
            f'  [Environment]::SetEnvironmentVariable("{var}", "<your path>", "User")\n'
            f'See CONTRIBUTING.md "Local machine paths".'
        )
    return value


def require_keys(cfg: dict, keys: Iterable[str], where: str) -> None:
    """Raise KeyError naming ``where.<key>`` for every key missing from cfg."""
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise KeyError(
            "missing config key(s): " + ", ".join(f"{where}.{k}" for k in missing)
        )


def load_catalogue(catalogue_path: Path) -> dict:
    """Load the YAML data catalogue from disk, rooted at GCFM_RAW_DATA_ROOT."""
    with open(catalogue_path, encoding="utf-8") as f:
        catalogue = yaml.safe_load(f)
    catalogue["meta"]["root"] = local_path("GCFM_RAW_DATA_ROOT")
    return catalogue


def catalogue_entry(catalogue, name):
    """Return the catalogue entry dict for the named dataset."""
    for ds in catalogue["datasets"]:
        if ds["name"] == name:
            return ds
    raise KeyError(f"Dataset {name} not found in catalogue.")


def raw_input_path(catalogue, name):
    """Return the absolute path to a raw input dataset."""
    entry = catalogue_entry(catalogue, name)
    return str(Path(catalogue["meta"]["root"]) / entry["file_path"])


def general_path(catalogue, name):
    """Return the relative path to a dataset using the catalogue root."""
    entry = catalogue_entry(catalogue, name)
    return str(Path(f"./{catalogue['meta']['root']}") / entry["file_path"])


def read_geometry(file_path: str) -> gpd.GeoDataFrame:
    """Read a vector geometry file into a GeoDataFrame."""
    return gpd.read_file(file_path)
