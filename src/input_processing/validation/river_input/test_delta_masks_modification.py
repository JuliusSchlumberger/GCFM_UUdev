"""
workflow_delta_masks.py
=======================
Workflow to modify coastal delta polygons from Edmonds et al. (2020) so that
their offshore edges fully reach the coastline.

Pipeline overview
-----------------
1. Load the delta polygon shapefile and the ESA Copernicus land-use raster.
2. (Optional) Produce a diagnostic vertex-position plot for a single test delta.
3. For each delta above the minimum area threshold:
   a. Clip supporting data (coastline, rivers, land-use) to a local bounding box.
   b. Call :func:`modify_masks` to iteratively expand the polygon to the coast.
   c. Collect the result; log and skip any delta that raises an error.
4. Write the corrected polygons to a GeoPackage.

Notes
-----
* File paths that are still hardcoded are marked ``# TODO: move to datacatalog``.
* All CRS handling is explicit; intermediate objects are reprojected before use
  rather than assumed to already be in the correct CRS.
"""

import logging

import geopandas as gpd
import rioxarray

from src.input_processing.config.loader import config
from src.input_processing.utils.validation.modify_delta_masks import (
    plot_polygons_with_vertices,
    unionize_coastal_data,
    process_delta,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def modify_test_delta_masks(position_plot: bool = False, debug_plot: bool = True):
    """
    Main workflow: load, correct, and save all delta masks.

    Parameters
    ----------
    position_plot : bool, optional
        If ``True``, generate a vertex-position diagnostic plot for the
        test-case delta defined by ``config['Testcase']['id_delta1']`` before
        running the main loop. Default ``False``.
    debug_plot : bool, optional
        If ``True``, generate step-by-step scaling figures for every delta.
        Default ``True``.
    """

    logger = logging.getLogger(__name__)
    logger.info("=== Modify & Test Delta Masks ===")

    # ------------------------------------------------------------------
    # 1. Load input files
    # ------------------------------------------------------------------
    logger.info("Loading delta polygons from %s", config['filepaths']['delta_polygons'])
    delta_polygons = gpd.read_file(config['filepaths']['delta_polygons']).to_crs(
        epsg=config['CRS']['for_distances']
    )

    # Open the land-use raster lazily; actual data is only read when clipped
    # inside _load_delta_context, keeping memory usage low.
    logger.info("Opening land-use raster from %s", config['filepaths']['land_use'])
    land_use = rioxarray.open_rasterio(config['filepaths']['land_use'], masked=True, chunks=True)

    # ------------------------------------------------------------------
    # 2. Optional: vertex-position diagnostic plot for one test delta
    # ------------------------------------------------------------------
    if position_plot:
        testcase_id = 'id_delta1'  # TODO: make this a function parameter
        test_polygon = delta_polygons[
            delta_polygons["BasinID2"] == config['Testcase'][testcase_id]
        ]
        if test_polygon.empty:
            logger.warning("Test-case delta '%s' not found in dataset.", testcase_id)
        else:
            coast_geom = unionize_coastal_data(test_polygon, config['filepaths']['coastline'])
            plot_polygons_with_vertices(test_polygon, coast_geom, testcase_id)

    # ------------------------------------------------------------------
    # 3. Main loop: process each delta
    # ------------------------------------------------------------------
    # Store results keyed by the original dataframe index so that skipped
    # deltas do not cause a length mismatch when writing back to the GeoDataFrame.
    results: dict[int, object] = {}
    failed_deltas: list = []

    for idx, row in delta_polygons.iterrows():
        try:
            modified_polygon = process_delta(row, land_use, logger, debug_plot)
            if modified_polygon is not None:
                results[idx] = modified_polygon

        except Exception:
            logger.exception(
                "Failed to process delta %s (index %d). Skipping.",
                row.BasinID2, idx,
            )
            failed_deltas.append(row.BasinID2)

    # ------------------------------------------------------------------
    # 4. Assemble output GeoDataFrame and write to file
    # ------------------------------------------------------------------
    if not results:
        logger.error("No deltas were successfully processed. Output file not written.")
        return

    # Map results back by index; rows with no result retain the original geometry.
    new_delta_masks = delta_polygons.copy()
    for idx, geom in results.items():
        new_delta_masks.at[idx, 'geometry'] = geom

    logger.info("Writing %d modified polygons to %s", len(results), config['filepaths']['output'])
    new_delta_masks.to_file(config['filepaths']['output'], driver="GPKG")

    # ------------------------------------------------------------------
    # 5. Summary report
    # ------------------------------------------------------------------
    n_total    = len(delta_polygons)
    n_modified = len(results)
    n_skipped  = n_total - n_modified - len(failed_deltas)

    logger.info(
        "Done. %d/%d deltas modified | %d skipped (size filter) | %d failed.",
        n_modified, n_total, n_skipped, len(failed_deltas),
    )
    if failed_deltas:
        logger.warning("Failed deltas: %s", failed_deltas)