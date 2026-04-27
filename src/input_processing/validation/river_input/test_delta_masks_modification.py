"""Workflow to modify coastal delta polygons from Edmonds et al. (2020).

Corrects delta polygons whose offshore edges do not fully reach the coastline
by iteratively expanding them outward until they are properly bounded.

Pipeline overview:
    1. Load the delta polygon shapefile and the ESA Copernicus land-use raster.
    2. Optionally produce a diagnostic vertex-position plot for a single test
       delta.
    3. For each delta above the minimum area threshold:

       a. Clip supporting data (coastline, rivers, land-use) to a local
          bounding box.
       b. Call :func:`modify_masks` to iteratively expand the polygon to the
          coast.
       c. Collect the result; log and skip any delta that raises an error.

    4. Write the corrected polygons to a GeoPackage.

Note:
    File paths that are still hardcoded are marked ``# TODO: move to datacatalog``.
    All CRS handling is explicit; intermediate objects are reprojected before
    use rather than assumed to already be in the correct CRS.

Example:
    >>> from src.input_processing.workflows.workflow_delta_masks import (
    ...     modify_test_delta_masks,
    ... )
    >>> modify_test_delta_masks(position_plot=False, debug_plot=True)
"""

from __future__ import annotations

import logging

import geopandas as gpd
import rioxarray
from geopandas import GeoDataFrame
from shapely.geometry import Polygon

from src.input_processing.config.loader import config
from src.input_processing.utils.validation.modify_delta_masks import (
    plot_polygons_with_vertices,
    unionize_coastal_data,
    process_delta,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def modify_test_delta_masks(
    position_plot: bool = False,
    debug_plot: bool = True,
) -> None:
    """Load, correct, and save all delta masks.

    Iterates over every delta polygon in the configured input file, expands
    any polygon whose offshore edges do not reach the coastline, and writes
    the corrected set to a GeoPackage. Deltas that fail processing are logged
    and skipped without aborting the loop.

    Args:
        position_plot: If True, generate a vertex-position diagnostic figure
            for the test-case delta defined by
            ``config['Testcase']['id_delta1']`` before running the main loop.
            Useful for verifying vertex classification on a known delta before
            processing the full dataset. Defaults to False.
        debug_plot: If True, generate step-by-step scaling figures for every
            delta processed. Figures are saved to the configured output
            directory. Defaults to True.

    Returns:
        None. The corrected polygons are written to
        ``config['filepaths']['output']`` as a GeoPackage. If no deltas were
        successfully processed, the file is not written and an error is logged.

    Raises:
        FileNotFoundError: If the delta polygon file or land-use raster cannot
            be found at the configured paths.

    Example:
        >>> modify_test_delta_masks(position_plot=True, debug_plot=False)
        INFO: === Modify & Test Delta Masks ===
        INFO: Loading delta polygons from ...
    """
    logger: logging.Logger = logging.getLogger(__name__)
    logger.info("=== Modify & Test Delta Masks ===")

    # --- Load input files ---
    logger.info(
        "Loading delta polygons from %s", config["filepaths"]["delta_polygons"]
    )
    delta_polygons: GeoDataFrame = gpd.read_file(
        config["filepaths"]["delta_polygons"]
    ).to_crs(epsg=config["CRS"]["for_distances"])

    # Open the land-use raster lazily; actual data is only read when clipped
    # inside load_delta_context, keeping peak memory usage low.
    logger.info(
        "Opening land-use raster from %s", config["filepaths"]["land_use"]
    )
    land_use = rioxarray.open_rasterio(
        config["filepaths"]["land_use"], masked=True, chunks=True
    )

    # --- Optional: vertex-position diagnostic plot for one test delta ---
    if position_plot:
        testcase_id: str = "id_delta1"  # TODO: move to function parameter
        test_polygon: GeoDataFrame = delta_polygons[
            delta_polygons["BasinID2"] == config["Testcase"][testcase_id]
        ]
        if test_polygon.empty:
            logger.warning(
                "Test-case delta '%s' not found in dataset.", testcase_id
            )
        else:
            coast_geom = unionize_coastal_data(
                test_polygon, config["filepaths"]["coastline"]
            )
            plot_polygons_with_vertices(test_polygon, coast_geom, testcase_id)

    # --- Main loop: process each delta ---
    # Results are keyed by the original DataFrame index so that skipped deltas
    # do not cause a length mismatch when writing back to the GeoDataFrame.
    results: dict[int, Polygon] = {}
    failed_deltas: list[int | str] = []

    for idx, row in delta_polygons.iterrows():
        try:
            modified_polygon: Polygon | None = process_delta(
                row, land_use, logger, debug_plot
            )
            if modified_polygon is not None:
                results[idx] = modified_polygon

        except Exception:
            logger.exception(
                "Failed to process delta %s (index %d). Skipping.",
                row.BasinID2,
                idx,
            )
            failed_deltas.append(row.BasinID2)

    # --- Assemble output GeoDataFrame and write to file ---
    if not results:
        logger.error(
            "No deltas were successfully processed. Output file not written."
        )
        return

    # Map results back by index; rows with no result retain the original geometry.
    new_delta_masks: GeoDataFrame = delta_polygons.copy()
    for idx, geom in results.items():
        new_delta_masks.at[idx, "geometry"] = geom

    logger.info(
        "Writing %d modified polygons to %s",
        len(results),
        config["filepaths"]["output"],
    )
    new_delta_masks.to_file(config["filepaths"]["output"], driver="GPKG")

    # --- Summary report ---
    n_total: int = len(delta_polygons)
    n_modified: int = len(results)
    n_skipped: int = n_total - n_modified - len(failed_deltas)

    logger.info(
        "Done. %d/%d deltas modified | %d skipped (size filter) | %d failed.",
        n_modified,
        n_total,
        n_skipped,
        len(failed_deltas),
    )
    if failed_deltas:
        logger.warning("Failed deltas: %s", failed_deltas)