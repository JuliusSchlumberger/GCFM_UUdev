"""I/O utilities for loading the data catalogue and reading spatial datasets."""

import os
from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import yaml

SITE_FILE = Path(__file__).resolve().parents[2] / "config" / "site.yml"
SITE_ENV_VARS = {
    "results_dir": "GCFM_RESULTS_DIR",
    "raw_data_root": "GCFM_RAW_DATA_ROOT",
    "sfincs_exe": "GCFM_SFINCS_EXE",
}


def site_setting(key: str) -> str:
    """Return a site setting from its environment variable, else config/site.yml.

    Keys and variables are listed in config/site.example.yml. Empty values and
    unfilled ``<placeholder>`` values count as unset.
    """
    if key not in SITE_ENV_VARS:
        raise ValueError(
            f"Unknown site setting {key!r}; valid keys: {', '.join(SITE_ENV_VARS)}"
        )
    env_var = SITE_ENV_VARS[key]
    value = os.environ.get(env_var)
    if value:
        return value
    site = {}
    if SITE_FILE.exists():
        with open(SITE_FILE, encoding="utf-8") as f:
            site = yaml.safe_load(f) or {}
        unknown = sorted(set(site) - set(SITE_ENV_VARS))
        if unknown:
            raise ValueError(
                f"Unknown key(s) in {SITE_FILE}: {', '.join(unknown)}; "
                f"valid keys: {', '.join(SITE_ENV_VARS)}"
            )
    value = str(site.get(key) or "").strip()
    if not value or (value.startswith("<") and value.endswith(">")):
        raise OSError(
            f"Site setting {key!r} is not set: give it a value in {SITE_FILE} "
            f"or set the environment variable {env_var}. Create the file with:\n"
            f"  cp config/site.example.yml config/site.yml"
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
    """Load the YAML data catalogue, rooted at site setting raw_data_root."""
    with open(catalogue_path, encoding="utf-8") as f:
        catalogue = yaml.safe_load(f)
    catalogue["meta"]["root"] = site_setting("raw_data_root")
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
