"""Workflow entry point to derive Manning's N roughness from ESA WorldCover data."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from src.input_processing.utils.prepare_inputs_ut_roughness_from_landcover import (
    build_lut_array,
    parse_lookup,
    remap_landcover,
)
from src.utils.config_loader import load_config

_CONFIG_PATH = "../src/input_processing/config/decisions.yaml"
_CONFIG: Final[dict] = load_config(_CONFIG_PATH)  # type: ignore[type-arg]


def get_roughness_from_landcover(
    csv_path: str = _CONFIG["filepaths"]["ESA_roughness_mapping"],
    landcover_path: str = _CONFIG["filepaths"]["land_use"],
    max_code: int = 201,
    output_path: str = _CONFIG["filepaths"]["roughness_map"],
    chunk_size: int = 1024,
) -> None:
    """Remap an ESA WorldCover raster to Manning's N roughness values.

    Parses the lookup CSV, builds a numpy LUT array, and remaps the landcover
    raster in chunks. Any ESA codes absent from the lookup table are set to
    NaN in the output; the set of unknown codes is returned by
    :func:`remap_landcover` and logged as a warning by that function.

    Args:
        csv_path: Path to the CSV mapping ESA WorldCover codes to Manning's N
            values.
        landcover_path: Path to the input ESA WorldCover GeoTIFF.
        max_code: Size of the LUT array; ESA codes at or above this value are
            skipped. Defaults to 201 (covers all current ESA WorldCover codes).
        output_path: Destination path for the Manning's N output GeoTIFF.
        chunk_size: Number of raster rows processed per chunk.
    """
    lookup = parse_lookup(Path(csv_path))
    lut = build_lut_array(lookup, max_code)
    remap_landcover(Path(landcover_path), lut, Path(output_path), chunk_size)
