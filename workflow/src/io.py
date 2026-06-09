"""I/O utilities for loading the data catalogue and reading spatial datasets."""

import yaml
from pathlib import Path
import geopandas as gpd


def load_catalogue(catalogue_path: Path) -> dict:
    """Load the YAML data catalogue from disk."""
    with open(catalogue_path, "r", encoding="utf-8") as f:
        catalogue = yaml.safe_load(f)
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
