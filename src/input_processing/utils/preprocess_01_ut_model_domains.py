"""Pipeline for building model domains from Edmonds et al. (2020) delta polygons.

Transforms the original delta polygons into a set of relevant model domains
by applying two sequential filters:

1. **Area filter** — removes polygons smaller than ``MIN_DELTA_AREA_KM2``.
2. **River intersection filter** — removes polygons with no overlapping river
   reach in the SWORD dataset.

For the surviving polygons, the overlapping Pfafstetter river-basin polygons
are found via spatial join and written to a GeoPackage file.

Example:
    >>> from src.input_processing.utils.preprocess_01_ut_model_domains import (
    ...     create_model_domains,
    ... )
    >>> create_model_domains(
    ...     used_delta_polygons="data/delta_polygons_used.gpkg",
    ...     outpath_domains="output/new_domains.gpkg",
    ...     outpath_mismatched="output/mismatched_polygons.gpkg",
    ...     outpath_subset="output/delta_polygons_subset.gpkg",
    ...     pfaf_path="data/river_basins_pfaf.gpkg",
    ... )
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from src.input_processing.config.loader import config
from src.input_processing.utils.util_unify_typing_and_schema import (
    ensure_valid_schema,
    compute_area_km2,
    build_spatial_index,
    find_intersections,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CRS_STANDARD: Final[int] = config["CRS"]["standard"]
CRS_PROJECTED: Final[int] = config["CRS"]["for_distances"]

GEOM_COL: Final[str] = config["DomainSchema"]["geometry_lbl"]
BASIN_COL: Final[str] = config["DomainSchema"]["delta_id_lbl"]
AREA_COL: Final[str] = config["DomainSchema"]["area_lbl"]
MIN_DELTA_AREA_KM2: Final[float] = config["Delta_masks"]["min_delta_area"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_relevant_basins(
    delta_gdf: GeoDataFrame,
    river_basins: GeoDataFrame,
) -> GeoDataFrame:
    """Return the subset of *river_basins* that spatially overlaps *delta_gdf*.

    Performs an inner spatial join and returns only the columns that were
    present in the original *river_basins* GeoDataFrame, discarding any join
    artifacts added by geopandas.

    Args:
        delta_gdf: Single-row GeoDataFrame for one delta polygon, used as the
            right-hand side of the spatial join.
        river_basins: Pre-loaded river-basin GeoDataFrame to query against.
            Must be in the same CRS as *delta_gdf*.

    Returns:
        A copy of the spatially overlapping rows from *river_basins*, retaining
        only the original columns. Returns an empty GeoDataFrame with the same
        schema if no basins overlap.

    Example:
        >>> basins = _get_relevant_basins(single_delta_gdf, river_basins_gdf)
        >>> print(len(basins))
        3
    """
    joined: GeoDataFrame = gpd.sjoin(river_basins, delta_gdf, how="inner")
    return cast(GeoDataFrame, joined[river_basins.columns]).copy()


def _create_new_domain(
    single_domain: GeoDataFrame,
    river_basins: GeoDataFrame,
) -> GeoDataFrame:
    """Return the river-basin polygons that overlap *single_domain*.

    Thin wrapper around :func:`_get_relevant_basins` that makes the intent
    explicit at the call site inside :func:`build_domains`.

    Args:
        single_domain: Single-row GeoDataFrame for one delta polygon in
            ``CRS_STANDARD``.
        river_basins: Pre-loaded and pre-projected river-basin GeoDataFrame.
            The caller is responsible for loading this once and passing it in
            to avoid repeated disk reads inside the loop.

    Returns:
        GeoDataFrame of river-basin polygons that overlap *single_domain*,
        with the same schema as *river_basins*.

    Example:
        >>> domain = _create_new_domain(single_gdf, river_basins_gdf)
        >>> print(domain.crs)
        EPSG:4326
    """
    return _get_relevant_basins(single_domain, river_basins)


# ---------------------------------------------------------------------------
# Public domain builder
# ---------------------------------------------------------------------------


def build_domains(
    valid_polygons: GeoDataFrame,
    pfaf_path: str,
) -> GeoDataFrame:
    """Build a GeoDataFrame of model domains from the supplied delta polygons.

    For each delta polygon the overlapping Pfafstetter river-basin polygons
    are found via spatial join and collected into a single GeoDataFrame. The
    river-basin file is read once before the loop to avoid repeated I/O.

    Args:
        valid_polygons: GeoDataFrame of delta polygons that have already passed
            the area and river-intersection filters. Must contain at minimum
            the columns ``GEOM_COL``, ``BASIN_COL``, and ``AREA_COL``.
        pfaf_path: Path to the Pfafstetter river-basin vector file (any
            OGR-readable format, e.g. GeoPackage or Shapefile).

    Returns:
        GeoDataFrame with columns ``[GEOM_COL, BASIN_COL, AREA_COL]`` in
        ``CRS_STANDARD``, with one row per river-basin polygon that overlaps
        any of the input delta polygons. Returns an empty GeoDataFrame with
        the same column schema if no domains could be built.

    Raises:
        ValueError: If *valid_polygons* is missing any required schema columns,
            as enforced by :func:`ensure_valid_schema`.

    Example:
        >>> domains = build_domains(valid_polygons_gdf, "data/pfaf_basins.gpkg")
        >>> print(domains.columns.tolist())
        ['geometry', 'BasinID2', 'Area']
    """
    ensure_valid_schema(valid_polygons)

    # Read the basin file ONCE here — not inside the loop.
    river_basins: GeoDataFrame = gpd.read_file(pfaf_path).to_crs(epsg=CRS_STANDARD)

    domains: list[GeoDataFrame] = []
    geom: BaseGeometry
    basin_id: int | str

    for row in valid_polygons.itertuples(index=False):
        geom = getattr(row, GEOM_COL)
        basin_id = getattr(row, BASIN_COL)

        single: GeoDataFrame = gpd.GeoDataFrame(
            [{GEOM_COL: geom}],
            crs=CRS_STANDARD,
            geometry=GEOM_COL,
        )

        new_domain: GeoDataFrame = _create_new_domain(single, river_basins)
        new_domain[BASIN_COL] = basin_id
        domains.append(new_domain)

    if not domains:
        empty_df = pd.DataFrame(
            data=None,
            index=pd.RangeIndex(0),
            columns=pd.Index([GEOM_COL, BASIN_COL, AREA_COL]),
        )
        empty: gpd.GeoDataFrame = gpd.GeoDataFrame(empty_df)
        empty = empty.set_crs(epsg=CRS_STANDARD)
        return cast(GeoDataFrame, empty)

    result: GeoDataFrame = gpd.GeoDataFrame(
        pd.concat(domains, ignore_index=True),
        crs=CRS_STANDARD,
        geometry=GEOM_COL,
    )
    result[AREA_COL] = compute_area_km2(result)

    return result


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def create_model_domains(
    used_delta_polygons: str,
    outpath_domains: str,
    outpath_mismatched: str,
    outpath_subset: str,
    pfaf_path: str,
) -> None:
    """Run the full domain-building pipeline: load, filter, intersect, and save.

    Applies two sequential filters to the input delta polygons — area and
    river intersection — then builds Pfafstetter basin domains for the
    surviving polygons and writes three output GeoPackage files.

    Args:
        used_delta_polygons: Path to the input delta polygons GeoPackage
            (Edmonds et al. 2020 dataset, pre-filtered to the study area).
        outpath_domains: Output path for the basin domain polygons (large
            deltas with an intersecting river reach).
        outpath_mismatched: Output path for large delta polygons that had no
            intersecting river reach in the SWORD dataset.
        outpath_subset: Output path for the subset of input delta polygons
            that passed the area filter and were used for domain building.
        pfaf_path: Path to the Pfafstetter river-basin vector file used in
            :func:`build_domains`.

    Returns:
        None. Results are written directly to the three output paths.

    Raises:
        ValueError: If any required schema columns are missing from the loaded
            delta or river GeoDataFrames.
        fiona.errors.DriverError: If any output path is not writable or the
            directory does not exist.

    Example:
        >>> create_model_domains(
        ...     used_delta_polygons="data/delta_polygons_used.gpkg",
        ...     outpath_domains="output/new_domains.gpkg",
        ...     outpath_mismatched="output/mismatched.gpkg",
        ...     outpath_subset="output/subset.gpkg",
        ...     pfaf_path="data/pfaf_basins.gpkg",
        ... )
    """
    # --- Load and validate ---
    deltas: GeoDataFrame = ensure_valid_schema(
        gpd.read_file(used_delta_polygons), excluded=[AREA_COL]
    )
    rivers: GeoDataFrame = ensure_valid_schema(
        gpd.read_file(config["filepaths"]["river_sword"]),
        excluded=[AREA_COL, BASIN_COL],
    )

    # --- Area filter (vectorised) ---
    # compute_area_km2 returns integer km²; MIN_DELTA_AREA_KM2 is also in km².
    deltas[AREA_COL] = compute_area_km2(deltas)
    large_mask: pd.Series = deltas[AREA_COL] > MIN_DELTA_AREA_KM2

    large_deltas: GeoDataFrame = deltas.loc[large_mask].copy()
    small_deltas: GeoDataFrame = deltas.loc[~large_mask].copy()

    # --- Spatial index and intersection classification ---
    river_tree: STRtree = build_spatial_index(rivers)
    intersects_mask: np.ndarray = find_intersections(large_deltas, river_tree)

    valid_polygons: GeoDataFrame = large_deltas.loc[intersects_mask].copy()
    mismatched_polygons: GeoDataFrame = large_deltas.loc[~intersects_mask].copy()

    # --- Build domains ---
    domains: GeoDataFrame = build_domains(valid_polygons, pfaf_path)

    # --- Save outputs ---
    domains.to_file(Path(outpath_domains), driver="GPKG")
    mismatched_polygons.to_file(Path(outpath_mismatched), driver="GPKG")
    valid_polygons.to_file(Path(outpath_subset), driver="GPKG")

    print(
        f"Processed: {len(valid_polygons)} domains | "
        f"{len(mismatched_polygons)} no river | "
        f"{len(small_deltas)} too small"
    )
