"""
This file contains all functions needed to get from the original Polygons provided by Edmonds et al. (2020) to the set
of relevant delta domain (large enough, intersects with river system) and the corresponding relevant water-shed levels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

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
# Domain construction (private helpers)
# ---------------------------------------------------------------------------


def _get_relevant_basins(
    delta_gdf: GeoDataFrame,
    river_basins: GeoDataFrame,
) -> GeoDataFrame:
    """
    Spatial-join to return the subset of *river_basins* that overlaps
    *delta_gdf*, retaining only the original river-basin columns.
    """
    joined: GeoDataFrame = gpd.sjoin(river_basins, delta_gdf, how="inner")
    return joined[river_basins.columns].copy()


def _create_new_domain(
    single_domain: GeoDataFrame,
    river_basins: GeoDataFrame,
) -> GeoDataFrame:
    """
    Return the river-basin polygons that overlap *single_domain*.

    Args:
        single_domain: Single-row GeoDataFrame for one delta polygon.
        river_basins:  Pre-loaded and pre-projected river-basin GeoDataFrame.
                       Caller is responsible for loading once and passing in
                       to avoid repeated disk reads inside a loop.
    """
    return _get_relevant_basins(single_domain, river_basins)


# ---------------------------------------------------------------------------
# Public domain builder
# ---------------------------------------------------------------------------


def build_domains(
    valid_polygons: GeoDataFrame,
    pfaf_path: str,
) -> GeoDataFrame:
    """
    Build and return a GeoDataFrame of model domains for *valid_polygons*.

    For each delta polygon the overlapping Pfafstetter river-basin polygons
    are found via spatial join and collected into a single GeoDataFrame.
    The river-basin file is read once before the loop to avoid repeated I/O.

    Args:
        valid_polygons: GeoDataFrame of delta polygons that passed the area
                        and river-intersection filters.
        pfaf_path:      Path to the Pfafstetter river-basin file.

    Returns:
        GeoDataFrame with columns [GEOM_COL, BASIN_COL, AREA_COL] in
        CRS_STANDARD, or an empty GeoDataFrame if no domains could be built.
    """
    ensure_valid_schema(valid_polygons)

    # Read the basin file ONCE here — not inside the loop.
    river_basins: GeoDataFrame = gpd.read_file(pfaf_path).to_crs(epsg=CRS_STANDARD)

    domains: list[GeoDataFrame] = []

    geom: BaseGeometry
    basin_id: int | str  # adjust to match your actual BASIN_COL dtype

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
        return gpd.GeoDataFrame(
            columns=[GEOM_COL, BASIN_COL, AREA_COL],
            crs=CRS_STANDARD,
            geometry=GEOM_COL,
        )

    result: GeoDataFrame = gpd.GeoDataFrame(
        pd.concat(domains, ignore_index=True),
        crs=CRS_STANDARD,
        geometry=GEOM_COL,
    )
    result[AREA_COL] = compute_area_km2(result)  # already returns int Series

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
    """
    Full pipeline: load → validate → filter → intersect → build → save.

    Outputs three GeoPackage files:
    - *outpath_domains*:    Basin domains (polygons with a matching river).
    - *outpath_mismatched*: Large polygons with no intersecting river.
    - *outpath_subset*:     Subset of original delta polygons that were used.
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
