"""Workflow entry point to prepare inputs on bathymetry information."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from src.input_processing.utils.prepare_inputs_ut_gebco import (
    build_marine_buffer,
    load_land_polygons,
    merge_tiles,
    serialize_clipping,
)
from src.utils.config_loader import load_config

_CONFIG_PATH = "../src/input_processing/config/decisions.yaml"
_CONFIG: Final[dict] = load_config(_CONFIG_PATH)  # type: ignore[type-arg]


def clip_and_merge_gebco(
    land_path: str = _CONFIG["filepaths"]["OSM_land"],
    bbox: list[float] | None = None,
    buffer_deg: int = _CONFIG["Delta_masks"]["offshore_buffer"],
    buffer_crs: str = _CONFIG["CRS"]["standard"],
    buffer_fname: str = _CONFIG["filepaths"]["OSM_marine_buffer"],
    gebco_dir: str = _CONFIG["filepaths"]["GEBCO_original"],
    gebco_clipped: str = _CONFIG["filepaths"]["GEBCO_clipped"],
    depth_buffer: float = _CONFIG["Delta_masks"]["gebco_depth"],
    n_workers: int = 6,
    config_dict: dict = _CONFIG,  # type: ignore[assignment]
    delta_polygons_path: str = _CONFIG["filepaths"]["new_domains"],
) -> None:
    """Clip GEBCO bathymetry tiles to a marine buffer and merge into one GeoTIFF.

    Builds a marine-only buffer band from OSM land polygons, clips all GEBCO
    tiles in *gebco_dir* to that buffer (applying *depth_buffer* as a cutoff),
    merges the surviving tiles into a single output GeoTIFF, and removes the
    temporary clipped tile files.

    Args:
        land_path: Path to the OSM land polygons shapefile.
        bbox: Optional spatial filter as ``[minx, miny, maxx, maxy]`` applied
            before building the marine buffer.
        buffer_deg: Outward buffer distance for the marine band, in the units
            of *buffer_crs*.
        buffer_crs: CRS string (e.g. ``"EPSG:3857"``) used for metric
            buffering of the land polygons.
        buffer_fname: Output path for the intermediate marine buffer GeoPackage.
        gebco_dir: Directory containing the GEBCO GeoTIFF tiles to clip.
        gebco_clipped: Output path for the merged, clipped GEBCO GeoTIFF.
        depth_buffer: Depth cutoff in metres; pixels shallower than this value
            are masked during clipping.
        n_workers: Number of parallel worker processes for buffering and
            clipping.
        config_dict: Project config dict forwarded to helper functions.
        delta_polygons_path: Path to the delta basin GeoPackage used for
            pre-filtering land polygons.
    """
    land_gdf = load_land_polygons(land_path, config_dict, bbox)

    marine_gdf = build_marine_buffer(
        land_gdf,
        bbox,
        buffer_deg,
        buffer_crs,
        buffer_fname,
        n_workers,
        config_dict,
        delta_polygons_path,
    )

    clipped_paths = serialize_clipping(
        marine_gdf,
        Path(gebco_dir),
        depth_buffer,
        n_workers,
        config_dict,
    )

    merge_tiles(clipped_paths, Path(gebco_clipped))

    clipped_tmp = Path(gebco_dir) / "_tmp_clipped_tiles"
    for f in clipped_tmp.glob("*.tif"):
        f.unlink()
    clipped_tmp.rmdir()
