"""Tile selection and copying pipeline for the Delta-DTM dataset.

Identifies all Delta-DTM tiles that overlap the configured delta polygons
(plus a buffer) and copies them to a temporary working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from src.input_processing.utils.prepare_inputs_ut_deltadtm import (
    combine_tiles_all_deltas,
    copy_tiles,
)
from src.utils.config_loader import load_config

_CONFIG_PATH = "../src/input_processing/config/decisions.yaml"
_CONFIG: Final[dict] = load_config(_CONFIG_PATH)  # type: ignore[type-arg]


def select_deltadtm_tiles(
    tile_dir: str = _CONFIG["filepaths"]["DeltaDTM_original"],
    delta_polygons_path: str = _CONFIG["filepaths"]["new_domains"],
    buffer_deg: float = _CONFIG["Delta_masks"]["DTM_buffer"],
    output_dir: str = _CONFIG["filepaths"]["DeltaDTM_temp"],
    workers: int = 6,
    tile_extent: int | float = _CONFIG["filepaths"]["DeltaDTM_tile_extent"],
) -> None:
    """Select and copy Delta-DTM tiles overlapping the configured delta polygons.

    Combines :func:`combine_tiles_all_deltas` and :func:`copy_tiles` into a
    single convenience function. Tiles are selected by intersecting each delta
    polygon's buffered bounding box against the tile filenames in *tile_dir*,
    then copied in parallel to *output_dir*.

    Args:
        tile_dir: Root directory containing the Delta-DTM tile files.
        delta_polygons_path: Path to the GeoPackage with delta basin polygons
            used to determine which tiles are relevant.
        buffer_deg: Buffer in degrees added around each delta bounding box
            before tile selection.
        output_dir: Destination directory for the copied tiles.
        workers: Number of parallel copy threads.
        tile_extent: Width and height of each tile in degrees.
    """
    unique_selected = combine_tiles_all_deltas(
        Path(tile_dir),
        Path(delta_polygons_path),
        buffer_deg,
        _CONFIG,
        tile_extent,
    )

    copy_tiles(
        unique_selected,
        Path(output_dir),
        workers,
    )
