"""Shared schema validation, CRS enforcement, and spatial index utilities.

Provides the canonical column name constants and core utility functions used
throughout the input-processing pipeline. Importing from this module ensures
that column names, CRS codes, and validation logic stay consistent across all
scripts.

Constants exported:
    - ``CRS_STANDARD``: EPSG code for the project geographic CRS (WGS-84).
    - ``CRS_PROJECTED``: EPSG code for the project metric CRS (for area/distance).
    - ``GEOM_COL``: Column name for geometry.
    - ``BASIN_COL``: Column name for the basin/delta identifier.
    - ``AREA_COL``: Column name for area in km².
    - ``MIN_DELTA_AREA_KM2``: Minimum delta area threshold in km².

Example:
    >>> from src.input_processing.utils.util_unify_typing_and_schema import (
    ...     ensure_valid_schema, compute_area_km2, CRS_STANDARD
    ... )
    >>> gdf = ensure_valid_schema(raw_gdf, excluded=["Area"])
    >>> gdf["Area"] = compute_area_km2(gdf)
"""

from __future__ import annotations

from typing import Final, cast

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from shapely.strtree import STRtree

from src.utils.config_loader import load_config

_CONFIG_PATH = "../src/input_processing/config/decisions.yaml"
_CONFIG: dict = load_config(_CONFIG_PATH)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CRS_STANDARD: Final[int] = _CONFIG["CRS"]["standard"]
CRS_PROJECTED: Final[int] = _CONFIG["CRS"]["for_distances"]

GEOM_COL: Final[str] = _CONFIG["DomainSchema"]["geometry_lbl"]
BASIN_COL: Final[str] = _CONFIG["DomainSchema"]["delta_id_lbl"]
AREA_COL: Final[str] = _CONFIG["DomainSchema"]["area_lbl"]
MIN_DELTA_AREA_KM2: Final[float] = _CONFIG["Delta_masks"]["min_delta_area"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_schema(gdf: GeoDataFrame, excluded: list[str]) -> None:
    """Raise ValueError if any required column is absent from *gdf*.

    Reads the required column names from ``_CONFIG['DomainSchema']`` and checks
    that each one is present in *gdf*, unless it appears in *excluded*.

    Args:
        gdf: The GeoDataFrame to validate.
        excluded: Column names to skip during the check. Pass an empty list
            to require all schema columns.

    Raises:
        ValueError: If one or more required columns are absent from *gdf*,
            listing the missing names in the error message.

    Example:
        >>> _validate_schema(gdf, excluded=["Area"])  # skips the Area column
    """
    required: list[str] = [
        _CONFIG["DomainSchema"][k]
        for k in _CONFIG["DomainSchema"]
        if _CONFIG["DomainSchema"][k] not in excluded
    ]
    missing: list[str] = [col for col in required if col not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Public utilities
# ---------------------------------------------------------------------------


def ensure_valid_schema(
    gdf: GeoDataFrame,
    excluded: list[str] | None = None,
) -> GeoDataFrame:
    """Validate schema and return a normalised copy of *gdf*.

    Applies three normalisation steps in order:

    1. **Schema check** — raises ``ValueError`` if any required column is
       absent (respecting *excluded*).
    2. **CRS enforcement** — sets the CRS to ``CRS_STANDARD`` if it is
       missing, or reprojects if it differs.
    3. **Null geometry removal** — drops rows with null geometry and logs the
       count.

    Args:
        gdf: Input GeoDataFrame to validate and normalise. The original is
            not modified — a copy is always returned.
        excluded: Column names to skip during the schema check, e.g. columns
            that will be computed later in the pipeline such as ``AREA_COL``.
            Defaults to no exclusions (``None`` is treated as ``[]``).

    Returns:
        A normalised copy of *gdf* in ``CRS_STANDARD`` with no null
        geometries.

    Raises:
        ValueError: If any required schema column is missing and not in
            *excluded*.

    Example:
        >>> clean = ensure_valid_schema(raw_gdf, excluded=["Area"])
        >>> print(clean.crs.to_epsg())
        4326
    """
    excluded = excluded or []
    _validate_schema(gdf, excluded)

    gdf = gdf.copy()

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=CRS_STANDARD)
    elif gdf.crs.to_epsg() != CRS_STANDARD:
        gdf = gdf.to_crs(epsg=CRS_STANDARD)

    null_geom_count: int = int(gdf.geometry.isna().sum())
    if null_geom_count > 0:
        print(
            f"[ensure_valid_schema] Dropping {null_geom_count} row(s) "
            f"with null geometry."
        )
        gdf = cast(GeoDataFrame, gdf[gdf.geometry.notna()].copy())

    return gdf


def compute_area_km2(gdf: GeoDataFrame) -> pd.Series:
    """Return a Series of integer area values in km² for each row of *gdf*.

    Reprojects *gdf* from ``CRS_STANDARD`` (geographic, degrees) to
    ``CRS_PROJECTED`` (metric) before computing areas, so the result is
    geometrically correct regardless of latitude.

    Args:
        gdf: GeoDataFrame in ``CRS_STANDARD`` whose geometry areas will be
            computed. The original CRS is not modified.

    Returns:
        Integer ``pd.Series`` of area values in km², aligned to the index of
        *gdf*. Values are truncated (not rounded) to integers via
        ``astype(int)``.

    Example:
        >>> gdf["Area"] = compute_area_km2(gdf)
        >>> print(gdf["Area"].dtype)
        int64
    """
    projected: GeoDataFrame = gdf.to_crs(epsg=CRS_PROJECTED)
    areas: pd.Series = projected.geometry.area / 1e6
    return areas.astype(int)


def build_spatial_index(gdf: GeoDataFrame) -> STRtree:
    """Return an STRtree spatial index built from the geometries of *gdf*.

    The returned tree can be queried with a single geometry or an array of
    geometries using the ``predicate`` argument for fast spatial lookups
    without loading the full GeoDataFrame.

    Args:
        gdf: GeoDataFrame whose geometries will be indexed. All rows are
            included; filter *gdf* before calling if a subset is needed.

    Returns:
        An :class:`~shapely.strtree.STRtree` over the geometry values of
        *gdf*, suitable for ``query`` calls with spatial predicates such as
        ``'intersects'`` or ``'dwithin'``.

    Example:
        >>> tree = build_spatial_index(rivers_gdf)
        >>> hits = tree.query(delta_polygon, predicate="intersects")
    """
    return STRtree(gdf.geometry.values)


def find_intersections(polygons: GeoDataFrame, river_tree: STRtree) -> np.ndarray:
    """Return a boolean mask where True means the polygon intersects a river.

    Uses a vectorised bulk ``STRtree.query`` call instead of a Python-level
    loop. The query returns a ``(2, N)`` array of hit pairs; row 0 contains
    indices into *polygons* and row 1 contains indices into the tree.

    Args:
        polygons: GeoDataFrame of polygons to test for river intersection.
            Must be in the same CRS as the geometries used to build
            *river_tree*.
        river_tree: Pre-built STRtree of river geometries, typically the
            output of :func:`build_spatial_index`.

    Returns:
        Boolean ``np.ndarray`` of shape ``(len(polygons),)`` where ``True``
        means the polygon at that position intersects at least one river
        geometry in *river_tree*.

    Example:
        >>> river_tree = build_spatial_index(rivers_gdf)
        >>> mask = find_intersections(delta_polygons_gdf, river_tree)
        >>> valid = delta_polygons_gdf.loc[mask]
    """
    # hits shape is (2, n_hits):
    #   row 0 → index into polygons.geometry
    #   row 1 → index into the STRtree
    hits: np.ndarray = river_tree.query(polygons.geometry, predicate="intersects")

    mask: np.ndarray = np.zeros(len(polygons), dtype=bool)
    if hits.size > 0:
        mask[np.unique(hits[0])] = True
    return mask
