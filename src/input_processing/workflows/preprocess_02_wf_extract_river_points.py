"""Workflow for extracting river source points across all delta domain polygons.

Reads the pre-built basin domains and for each delta finds the most-downstream
GloFAS discharge cells that can serve as model inflow points.

Performance notes:
    - All heavy files (rivers, coastline, GloFAS) are loaded once via
      :func:`load_global_data` before the loop.
    - Per-delta subsets are derived via :func:`load_data_delta_domain` using
      coordinate indexing (``.cx``) and xarray slicing — no repeated file I/O.
    - Basin and delta lookups use pre-built dicts (O(1) per delta).
    - GloFAS is periodically re-opened every ``_GLOFAS_RELOAD_INTERVAL``
      iterations to prevent xarray lazy-graph accumulation from causing
      slowdowns or crashes.
    - Accumulated results are flushed to disk every ``_FLUSH_INTERVAL``
      successful iterations to cap peak memory usage.

Example:
    >>> from src.input_processing.workflows.preprocess_02_wf_extract_river_points import (
    ...     extract_points,
    ... )
    >>> extract_points()
"""

from __future__ import annotations

import gc
from collections.abc import Hashable
from pathlib import Path
from typing import Final, cast

import geopandas as gpd
import pandas as pd
import xarray as xr
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.geometry.base import BaseGeometry

from src.utils.config_loader import load_config
from src.input_processing.utils.loading_files import (
    GlobalData,
    load_data_delta_domain,
    load_global_data,
)
from src.input_processing.utils.preprocess_02_ut_extract_river_points import (
    clip_basin_boundary_from_coast,
    extract_cells_within_delta,
)
from src.input_processing.utils.util_unify_typing_and_schema import (
    BASIN_COL,
    CRS_STANDARD,
    GEOM_COL,
    ensure_valid_schema,
)
from src.input_processing.utils.validation.modify_delta_masks import classify_points
from src.utils.setup_logger import setup_logging

_LOG = setup_logging("preprocess_02_wf_extract_river_points")
_CONFIG_PATH = "../src/input_processing/config/decisions.yaml"
_CONFIG: Final[dict] = load_config(_CONFIG_PATH)  # type: ignore[type-arg]

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

_FLUSH_INTERVAL: int = 5
"""Write accumulated results to disk every this many successful iterations."""

_GLOFAS_RELOAD_INTERVAL: int = 5
"""Re-open the GloFAS dataset every this many iterations to clear the lazy graph."""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _is_empty(gdf: GeoDataFrame | None) -> bool:
    """Return True when *gdf* is None or contains no rows.

    Args:
        gdf: GeoDataFrame to check, or None.

    Returns:
        True if *gdf* is None or has no rows; False otherwise.

    Example:
        >>> _is_empty(None)
        True
        >>> _is_empty(gpd.GeoDataFrame())
        True
    """
    return gdf is None or gdf.empty


def _reload_glofas(global_data: GlobalData) -> GlobalData:
    """Close the current GloFAS dataset and re-open it with a fresh lazy graph.

    Returns a new :class:`GlobalData` instance reusing the already-loaded
    rivers and coastline GeoDataFrames. Closing and re-opening the xarray
    Dataset discards any accumulated lazy computation graph from previous
    ``.sel()`` calls, which prevents the process from slowing down or crashing
    after many iterations.

    Args:
        global_data: The current ``GlobalData`` instance whose GloFAS dataset
            will be replaced. Rivers and coastline are reused unchanged.

    Returns:
        A new frozen :class:`GlobalData` instance with a freshly opened GloFAS
        dataset and the same rivers and coastline as *global_data*.

    Example:
        >>> global_data = _reload_glofas(global_data)
    """
    _LOG.info("Re-opening GloFAS dataset to clear lazy graph ...")
    try:
        global_data.glofas.close()
    except Exception:  # noqa: BLE001
        pass  # already closed or not closeable — proceed

    fresh_glofas: xr.Dataset = xr.open_dataset(
        _CONFIG["filepaths"]["glofas"]
    ).rio.write_crs(CRS_STANDARD)

    return GlobalData(
        rivers=global_data.rivers,
        coastline=global_data.coastline,
        glofas=fresh_glofas,
    )


def _flush_to_disk(
    frames: list[GeoDataFrame],
    path: str,
    crs: int,
) -> None:
    """Concatenate *frames* and append to (or create) a GeoPackage at *path*.

    If the output file already exists the data is appended; if it does not
    exist a new file is created. Does nothing if *frames* is empty.

    Args:
        frames: List of GeoDataFrames to concatenate and write. If empty,
            the function returns immediately without touching the file.
        path: Output file path. Must be writable; the parent directory must
            already exist.
        crs: EPSG code applied to the concatenated GeoDataFrame before
            writing.

    Returns:
        None. Data is written directly to *path*.

    Example:
        >>> _flush_to_disk(all_unique_sources, "output/unique.gpkg", 4326)
        >>> _flush_to_disk([], "output/unique.gpkg", 4326)  # no-op
    """
    if not frames:
        return
    gdf: GeoDataFrame = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=crs)
    p: Path = Path(path)
    if p.exists():
        gdf.to_file(p, driver="GPKG", mode="a")
    else:
        gdf.to_file(p, driver="GPKG")


# ---------------------------------------------------------------------------
# Source-point extraction for a single delta
# ---------------------------------------------------------------------------


def find_river_source_points(
    delta_basins_gpd: GeoDataFrame,
    delta_polygon: Polygon,
    rivers_gpd: GeoDataFrame,
    glofas_min: xr.DataArray,
    coast_polygon: BaseGeometry,
    delta_edmonds: GeoDataFrame,
    basin_polygons: GeoDataFrame,
    basin_polygons_domain: GeoDataFrame,
    all_rivers: GeoDataFrame,
    use_basins: bool = True,
) -> tuple[GeoDataFrame | None, GeoDataFrame | None, GeoSeries | None]:
    """Find unique and possible river source points for a single delta domain.

    Clips the river network to the basin, derives the inland boundary either
    from the Pfafstetter watershed or from the classified Edmonds polygon
    vertices, then calls :func:`extract_cells_within_delta` to identify the
    most-downstream GloFAS cells.

    Args:
        delta_basins_gpd: Basin polygons for this delta in ``CRS_STANDARD``.
        delta_polygon: Edmonds delta polygon geometry. Only used when
            ``use_basins=False``; ignored otherwise.
        rivers_gpd: River network clipped to the delta bounding box.
        glofas_min: Per-cell minimum GloFAS discharge DataArray for the delta
            region, derived from the full time series.
        coast_polygon: Union of coastline polygons for the delta region.
            Used to subtract the seaward boundary from the basin outline.
        delta_edmonds: Edmonds delta polygon GeoDataFrame, passed through to
            :func:`extract_cells_within_delta` for plotting.
        basin_polygons: All basin polygons, passed through for plot context.
        basin_polygons_domain: Basin polygons for this delta only, passed
            through for plotting.
        all_rivers: Full river network for the region, passed through for
            plot context.
        use_basins: If True, derive the inland boundary by clipping the
            Pfafstetter basin outline against the coastline. If False, use the
            :func:`classify_points` fallback on the raw Edmonds polygon.
            Defaults to True.

    Returns:
        A tuple of ``(unique_sources, possible_sources, buffered_cells)`` in
        ``CRS_STANDARD``, or ``(None, None, None)`` if the inland boundary
        could not be derived.

    Example:
        >>> unique, possible, cells = find_river_source_points(
        ...     delta_basins_gpd, delta_polygon, rivers_gpd, glofas_min,
        ...     coast_polygon, delta_edmonds, basin_polygons,
        ...     basin_polygons_domain, all_rivers
        ... )
    """
    # --- Clip rivers to this basin ---
    relevant_rivers: GeoDataFrame = gpd.sjoin(rivers_gpd, delta_basins_gpd, how="inner")
    relevant_rivers = cast(GeoDataFrame, relevant_rivers[rivers_gpd.columns].copy())
    relevant_rivers[BASIN_COL] = delta_basins_gpd[BASIN_COL].iloc[0]

    # --- Inland boundary ---
    inland_boundary: LineString | MultiLineString
    try:
        if use_basins:
            inland_boundary = clip_basin_boundary_from_coast(
                delta_basins_gpd, coast_polygon
            )
        else:
            bw, s1, s2, _, _ = classify_points(delta_polygon, coast_polygon)
            inland_boundary = MultiLineString(
                [LineString([s1, bw]), LineString([bw, s2])]
            )
    except ValueError as e:
        delta_id: int | str = delta_basins_gpd[BASIN_COL].iloc[0]
        _LOG.warning("Delta %s — could not derive inland boundary: %s", delta_id, e)
        return None, None, None

    # --- Extract GloFAS source cells ---
    return extract_cells_within_delta(
        glofas_min,
        inland_boundary,
        relevant_rivers,
        delta_edmonds,
        basin_polygons,
        basin_polygons_domain,
        all_rivers,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def extract_all_river_source_points(
    out_unique_sources: str = _CONFIG["filepaths"]["unique_sources"],
    out_possible_sources: str = _CONFIG["filepaths"]["possible_sources"],
    out_deltas_no_rivers: str = _CONFIG["filepaths"]["out_deltas_no_rivers"],
    debug_plots: bool = True,
) -> None:
    """Iterate over all delta domains, extract river source points, and save results.

    All heavy files are loaded once before the loop. Per-delta subsets are
    derived via coordinate-index slicing with no repeated file I/O. The GloFAS
    dataset is re-opened every ``_GLOFAS_RELOAD_INTERVAL`` iterations to
    prevent xarray lazy-graph accumulation. Accumulated results are flushed to
    disk every ``_FLUSH_INTERVAL`` successful iterations to cap peak memory.

    Args:
        out_unique_sources: Output path for the most-downstream unique inflow
            points per delta. Defaults to
            ``_CONFIG['filepaths']['unique_sources']``.
        out_possible_sources: Output path for all candidate inflow points per
            delta. Defaults to ``_CONFIG['filepaths']['possible_sources']``.
        out_deltas_no_rivers: Output path for delta polygons for which no
            river sources were found. Defaults to
            ``_CONFIG['filepaths']['out_deltas_no_rivers']``.
        debug_plots: If True, pass ``debug_plots=True`` through to
            :func:`find_river_source_points` so diagnostic figures are saved
            for each delta. Defaults to True.

    Returns:
        None. Three GeoPackage files are written to the configured output
        paths. Each file is built incrementally via :func:`_flush_to_disk` so
        partial results are preserved even if the pipeline is interrupted.

    Raises:
        FileNotFoundError: If the delta domains or basin domains GeoPackage
            cannot be found at the configured paths.

    Example:
        >>> extract_all_river_source_points(debug_plots=False)
    """
    # --- Load heavy datasets once ---
    global_data: GlobalData = load_global_data()

    delta_domains: GeoDataFrame = ensure_valid_schema(
        gpd.read_file(_CONFIG["filepaths"]["delta_polygons_used"]),
        excluded=[],
    )
    river_basins_gpd: GeoDataFrame = ensure_valid_schema(
        gpd.read_file(_CONFIG["filepaths"]["new_domains"]),
        excluded=[],
    )
    all_rivers: GeoDataFrame = global_data.rivers

    _LOG.info(
        "Loaded %d delta domain(s) and %d basin polygon(s).",
        len(delta_domains),
        len(river_basins_gpd),
    )

    # --- Pre-build O(1) lookups by basin ID ---
    basin_lookup: dict[int | str, GeoDataFrame] = {
        cast(int | str, k): cast(GeoDataFrame, v.copy())
        for k, v in river_basins_gpd.groupby(BASIN_COL)
    }
    delta_lookup: dict[int | str, GeoDataFrame] = {
        cast(int | str, k): cast(GeoDataFrame, v.copy())
        for k, v in delta_domains.groupby(BASIN_COL)
    }
    basin_polygons_lookup: dict[int | str, GeoDataFrame] = basin_lookup

    # --- Accumulators ---
    all_unique_sources: list[GeoDataFrame] = []
    all_possible_sources: list[GeoDataFrame] = []
    deltas_without_sources: list[pd.Series] = []

    success_count: int = 0
    fail_count: int = 0

    # --- Per-delta loop ---
    idx: Hashable
    row: pd.Series
    for idx, row in delta_domains.iterrows():
        idx_int: int = cast(int, idx)

        if idx_int % 10 == 0:
            _LOG.info(
                "Processing delta %d / %d (success=%d, fail=%d) ...",
                idx_int,
                len(delta_domains),
                success_count,
                fail_count,
            )

        # Periodic GloFAS reload — clears the xarray lazy computation graph.
        if idx_int > 0 and idx_int % _GLOFAS_RELOAD_INTERVAL == 0:
            global_data = _reload_glofas(global_data)
            gc.collect()

        polygon_id: int | str = cast(int | str, row[BASIN_COL])
        delta_polygon: Polygon = cast(Polygon, row[GEOM_COL])

        if polygon_id not in basin_lookup:
            _LOG.warning("Missing basin for delta %s — skipping.", polygon_id)
            deltas_without_sources.append(row.copy())
            fail_count += 1
            continue

        delta_basins_gpd: GeoDataFrame = basin_lookup[polygon_id]
        delta_edmonds: GeoDataFrame = delta_lookup[polygon_id]
        basin_polygons_domain: GeoDataFrame = basin_polygons_lookup[polygon_id]

        try:
            rivers_gpd, coast_polygon, coastline_gpd, glofas_min = (
                load_data_delta_domain(delta_basins_gpd, global_data)
            )
        except Exception as e:  # noqa: BLE001
            _LOG.error("Failed subsetting data for delta %s: %s", polygon_id, e)
            deltas_without_sources.append(row.copy())
            fail_count += 1
            continue

        unique_sources, possible_sources, _ = find_river_source_points(
            delta_basins_gpd=delta_basins_gpd,
            delta_polygon=delta_polygon,
            rivers_gpd=rivers_gpd,
            glofas_min=glofas_min,
            coast_polygon=coast_polygon,
            delta_edmonds=delta_edmonds,
            basin_polygons=river_basins_gpd,
            basin_polygons_domain=basin_polygons_domain,
            all_rivers=all_rivers,
        )

        del rivers_gpd, coast_polygon, coastline_gpd, glofas_min

        if _is_empty(unique_sources) or _is_empty(possible_sources):
            _LOG.debug("Delta %s: no sources found.", polygon_id)
            deltas_without_sources.append(row.copy())
            fail_count += 1
            continue

        assert unique_sources is not None and possible_sources is not None

        unique_sources[BASIN_COL] = polygon_id
        possible_sources[BASIN_COL] = polygon_id

        all_unique_sources.append(unique_sources)
        all_possible_sources.append(possible_sources)
        success_count += 1

        # Periodic flush to disk — caps peak memory by clearing the accumulators.
        if success_count % _FLUSH_INTERVAL == 0:
            _LOG.info(
                "Flushing %d result(s) to disk (success=%d, fail=%d) ...",
                len(all_unique_sources),
                success_count,
                fail_count,
            )
            _flush_to_disk(all_unique_sources, out_unique_sources, CRS_STANDARD)
            _flush_to_disk(all_possible_sources, out_possible_sources, CRS_STANDARD)
            all_unique_sources.clear()
            all_possible_sources.clear()
            gc.collect()
            _LOG.debug("Flush complete.")

    _LOG.info(
        "Pipeline complete — %d domain(s) processed, %d without sources.",
        success_count,
        fail_count,
    )

    # Final flush of any remaining results.
    _flush_to_disk(all_unique_sources, out_unique_sources, CRS_STANDARD)
    _flush_to_disk(all_possible_sources, out_possible_sources, CRS_STANDARD)

    if deltas_without_sources:
        gpd.GeoDataFrame(
            pd.DataFrame(deltas_without_sources),
            geometry=GEOM_COL,
            crs=CRS_STANDARD,
        ).to_file(Path(out_deltas_no_rivers), driver="GPKG")
        _LOG.info(
            "%d delta(s) without sources written to: %s",
            len(deltas_without_sources),
            out_deltas_no_rivers,
        )


def extract_points() -> None:
    """Run the full river source point extraction pipeline with default settings.

    Convenience entry point that calls
    :func:`extract_all_river_source_points` with all arguments at their
    configured defaults. Intended for use as a script entry point or a
    one-line pipeline trigger.

    Returns:
        None.

    Example:
        >>> extract_points()
    """
    extract_all_river_source_points()
