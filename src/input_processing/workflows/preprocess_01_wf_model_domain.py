"""Workflow entry point for building model domains at a chosen Pfafstetter level.

Thin wrapper around :func:`create_model_domains` that binds the configured
default file paths so the pipeline can be triggered with a single call and
no required arguments.

Example:
    >>> from src.input_processing.workflows.preprocess_01_wf_model_domain import (
    ...     create_domains_chosen_level,
    ... )
    >>> create_domains_chosen_level()
"""

from __future__ import annotations

from src.utils.config_loader import load_config
from src.input_processing.utils.preprocess_01_ut_model_domains import (
    create_model_domains,
)

from typing import Final

_CONFIG_PATH = "../src/input_processing/config/decisions.yaml"
_CONFIG: Final[dict] = load_config(_CONFIG_PATH)  # type: ignore[type-arg]


def create_domains_chosen_level(
    used_delta_polygons: str = _CONFIG["filepaths"]["hand_picked_deltas"],
    outpath_domains: str = _CONFIG["filepaths"]["new_domains"],
    outpath_mismatched: str = _CONFIG["filepaths"]["mismatched_polygons"],
    outpath_subset: str = _CONFIG["filepaths"]["delta_polygons_used"],
    pfaf_path: str = _CONFIG["filepaths"]["river_basins_applied"],
) -> None:
    """Build model domains at the configured Pfafstetter level.

    Thin wrapper around :func:`create_model_domains` that provides default
    file paths from the project config. All arguments can be overridden to
    run the pipeline against alternative inputs or write to different output
    locations, for example when processing a different Pfafstetter level or
    testing on a subset of deltas.

    Args:
        used_delta_polygons: Path to the input delta polygons GeoPackage,
            i.e. the hand-picked subset of Edmonds et al. (2020) polygons for
            the study area. Defaults to
            ``_CONFIG['filepaths']['hand_picked_deltas']``.
        outpath_domains: Output path for the basin domain polygons — large
            deltas with an intersecting river reach. Defaults to
            ``_CONFIG['filepaths']['new_domains']``.
        outpath_mismatched: Output path for large delta polygons that had no
            intersecting river reach. Defaults to
            ``_CONFIG['filepaths']['mismatched_polygons']``.
        outpath_subset: Output path for the subset of input delta polygons
            that passed the area filter and were used for domain building.
            Defaults to ``_CONFIG['filepaths']['delta_polygons_used']``.
        pfaf_path: Path to the Pfafstetter river-basin vector file used to
            build the domains. Defaults to
            ``_CONFIG['filepaths']['river_basins_applied']``.

    Returns:
        None. All results are written to the configured output paths.

    Raises:
        ValueError: If any required schema column is missing from the loaded
            delta or river GeoDataFrames (propagated from
            :func:`create_model_domains`).

    Example:
        >>> create_domains_chosen_level()

        Override paths for a different Pfafstetter level:

        >>> create_domains_chosen_level(
        ...     pfaf_path="data/river_basins_lvl05.gpkg",
        ...     outpath_domains="output/new_domains_lvl05.gpkg",
        ...     outpath_mismatched="output/mismatched_lvl05.gpkg",
        ...     outpath_subset="output/subset_lvl05.gpkg",
        ... )
    """
    create_model_domains(
        used_delta_polygons=used_delta_polygons,
        outpath_domains=outpath_domains,
        outpath_mismatched=outpath_mismatched,
        outpath_subset=outpath_subset,
        pfaf_path=pfaf_path,
    )
