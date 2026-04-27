"""Utilities for identifying GloFAS discharge cells as river inflow points.

Implements a multi-step pipeline to find the most-downstream unique inflow
points for a single delta domain:

1. **Threshold filter** — keep only GloFAS cells above a minimum discharge.
2. **Boundary filter** — keep only cells whose buffer intersects the inland
   domain boundary.
3. **River proximity filter** — keep only cells within half a grid cell of
   the river network.
4. **Downstream selection** — trace the river network upstream to discard
   redundant candidates and return only the most-downstream points.

Example:
    >>> from src.input_processing.utils.preprocess_02_ut_extract_river_points import (
    ...     extract_cells_within_delta,
    ... )
    >>> unique, possible, buffered = extract_cells_within_delta(
    ...     glofas_min, inland_boundary, rivers_gdf,
    ...     delta_edmonds, basin_polygons, basin_polygons_domain, all_rivers
    ... )
"""

from __future__ import annotations

from collections import defaultdict
from typing import Final

import pandas as pd
import geopandas as gpd
import numpy as np
from typing import cast
from shapely import get_parts
import xarray as xr
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import MultiLineString, LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, unary_union
from shapely.strtree import STRtree
from geopandas.array import GeometryArray

from src.input_processing.config.loader import config
from src.input_processing.utils.plotting import plot_river_locations
from src.input_processing.utils.util_unify_typing_and_schema import (
    ensure_valid_schema,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CRS_STANDARD: Final[int] = config["CRS"]["standard"]
GEOM_COL: Final[str] = config["DomainSchema"]["geometry_lbl"]
BASIN_COL: Final[str] = config["DomainSchema"]["delta_id_lbl"]
AREA_COL: Final[str] = config["DomainSchema"]["area_lbl"]
_LON: Final[str] = config["Rivers"]["glofas_lon_name"]
_LAT: Final[str] = config["Rivers"]["glofas_lat_name"]
_VAR: Final[str] = config["Rivers"]["glofas_discharge_parameter"]
_THRESHOLD: Final[float] = config["Rivers"]["glofas_min_discharge"]
_SNAP_TOL: Final[float] = config["Delta_masks"]["tolerance_deg"]


# ---------------------------------------------------------------------------
# Candidate cell filtering
# ---------------------------------------------------------------------------


def _filter_cells_by_threshold(
    q_values_nc: xr.DataArray,
    threshold: float,
    var_name: str,
    longitude: str,
    latitude: str,
) -> GeoDataFrame:
    """Return a GeoDataFrame of grid cells exceeding a discharge threshold.

    Filters the DataArray to retain only cells with discharge above
    *threshold*, converts the result to a DataFrame, and attaches point
    geometries from the longitude and latitude coordinates.

    Args:
        q_values_nc: GloFAS discharge DataArray with named longitude and
            latitude dimensions.
        threshold: Minimum discharge value in m³/s. Cells at or below this
            value are excluded.
        var_name: Name of the discharge variable in *q_values_nc*, used to
            drop NaN rows after conversion to a DataFrame.
        longitude: Name of the longitude dimension in *q_values_nc*.
        latitude: Name of the latitude dimension in *q_values_nc*.

    Returns:
        GeoDataFrame of point geometries in ``CRS_STANDARD``, one row per
        grid cell that exceeds *threshold*.

    Example:
        >>> cells = _filter_cells_by_threshold(glofas_da, 100.0, "dis24", "longitude", "latitude")
        >>> print(cells.geom_type.unique())
        ['Point']
    """
    q_thresh = q_values_nc.where(q_values_nc > threshold, drop=True)
    df = q_thresh.to_dataframe().dropna(subset=[var_name]).reset_index()
    print(f"Cells above threshold: {len(df)} / {q_values_nc.size}")
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[longitude], df[latitude]),
        crs=CRS_STANDARD,
    )


def _compute_half_cell(
    q_values_nc: xr.DataArray,
    longitude: str,
    latitude: str,
) -> float:
    """Return half the maximum grid-cell size in degrees.

    Computes the median resolution along each axis and returns half of the
    larger value. Used as a buffer radius when checking cell proximity to
    boundaries and rivers.

    Args:
        q_values_nc: GloFAS discharge DataArray with named longitude and
            latitude dimensions.
        longitude: Name of the longitude dimension.
        latitude: Name of the latitude dimension.

    Returns:
        Half the maximum grid spacing in degrees (the same unit as the
        coordinate values).

    Example:
        >>> half = _compute_half_cell(glofas_da, "longitude", "latitude")
        >>> print(round(half, 4))
        0.025
    """
    lon_res: float = float(q_values_nc[longitude].diff(longitude).median())
    lat_res: float = float(q_values_nc[latitude].diff(latitude).median())
    return max(abs(lon_res), abs(lat_res)) / 2.0


def _filter_cells_near_boundary(
    cells_gdf: GeoDataFrame,
    boundary: BaseGeometry,
    half_cell: float,
) -> GeoDataFrame:
    """Keep only cells whose buffer intersects the inland domain boundary.

    Each cell is buffered by *half_cell* before the intersection test, so a
    cell is retained even when its centre sits just outside the boundary line.
    The STRtree is built from the buffered cells and queried with the single
    boundary geometry.

    Args:
        cells_gdf: GeoDataFrame of point geometries (GloFAS grid cells).
        boundary: Single inland boundary geometry to test against. Typically
            the output of :func:`clip_basin_boundary_from_coast`.
        half_cell: Buffer radius in the same units as the CRS (degrees for
            ``CRS_STANDARD``).

    Returns:
        Filtered GeoDataFrame containing only cells whose buffer intersects
        *boundary*. Original row indices and columns are preserved.

    Example:
        >>> near = _filter_cells_near_boundary(cells_gdf, inland_boundary, 0.025)
        >>> print(f"{len(near)} / {len(cells_gdf)} cells on boundary")
    """
    buffered: GeometryArray = cells_gdf.geometry.buffer(half_cell).values
    tree: STRtree = STRtree(buffered)
    hits: np.ndarray = tree.query(boundary, predicate="intersects")
    mask: np.ndarray = np.zeros(len(cells_gdf), dtype=bool)
    if hits.size > 0:
        mask[hits] = True
    return cells_gdf.loc[mask].copy()


def _filter_cells_near_rivers(
    cells_gdf: GeoDataFrame,
    rivers_union: BaseGeometry,
    half_cell: float,
) -> GeoDataFrame:
    """Keep only cells within *half_cell* degrees of the river union geometry.

    Builds an STRtree from the cell geometries and queries it with the river
    union using a ``dwithin`` predicate, which is more efficient than buffering
    all cells individually.

    Args:
        cells_gdf: GeoDataFrame of point geometries (GloFAS grid cells),
            typically already filtered by :func:`_filter_cells_near_boundary`.
        rivers_union: Union of all river geometries in the delta region.
        half_cell: Maximum distance in degrees from a river for a cell to be
            retained.

    Returns:
        Filtered GeoDataFrame containing only cells within *half_cell* of the
        river union. Original row indices and columns are preserved.

    Example:
        >>> rivers_union = unary_union(rivers_gdf.geometry)
        >>> near = _filter_cells_near_rivers(boundary_cells, rivers_union, 0.025)
    """
    tree: STRtree = STRtree(cells_gdf.geometry.values)
    near: np.ndarray = tree.query(rivers_union, predicate="dwithin", distance=half_cell)
    mask: np.ndarray = np.zeros(len(cells_gdf), dtype=bool)
    if near.size > 0:
        mask[near] = True
    result: GeoDataFrame = cells_gdf.loc[mask].copy()
    print(f"Cells near rivers: {mask.sum()} / {len(cells_gdf)}")
    return result


# ---------------------------------------------------------------------------
# Basin boundary clipping
# ---------------------------------------------------------------------------


def clip_basin_boundary_from_coast(
    relevant_basins: GeoDataFrame,
    coast_polygon: BaseGeometry,
    simplify_tolerance: float = 0.01,
) -> LineString | MultiLineString:
    """Return the full inland basin boundary after removing coastal parts.

    Dissolves the basin polygons, computes the outer boundary, subtracts a
    buffered and simplified coastal zone, and returns all remaining line
    segments reunited as a single (Multi)LineString.

    Args:
        relevant_basins: Basin polygons for one delta in ``CRS_STANDARD``.
        coast_polygon: Coastal geometry used to mask the seaward boundary.
            Typically the output of ``coastline_gdf.geometry.union_all()``.
        simplify_tolerance: Simplification tolerance in degrees applied to the
            buffered coast before ``.difference()`` to prevent hangs on highly
            fragmented coastlines. Defaults to 0.01° (≈ 1 km at the equator).

    Returns:
        The complete inland boundary as a ``LineString`` (single contiguous
        segment) or ``MultiLineString`` (multiple disconnected segments).

    Raises:
        ValueError: If no ``LineString`` segments remain after subtracting the
            buffered coast, or if the reunited result is not a line geometry.

    Example:
        >>> boundary = clip_basin_boundary_from_coast(basin_gdf, coast_geom)
        >>> print(boundary.geom_type)
        MultiLineString
    """
    basin_boundary: BaseGeometry = relevant_basins.union_all().boundary
    buffered_coast: BaseGeometry = coast_polygon.buffer(
        config["Delta_masks"]["tolerance_deg"]
    )
    buffered_coast = buffered_coast.simplify(simplify_tolerance, preserve_topology=True)
    clipped: BaseGeometry = basin_boundary.difference(buffered_coast)

    # get_parts() handles any geometry type — unlike .geoms which raises
    # AttributeError on a bare LineString.
    lines: list[LineString] = [
        g for g in get_parts(clipped) if g.geom_type == "LineString"
    ]

    if not lines:
        raise ValueError(
            f"[clip_basin_boundary_from_coast] No inland boundary segments remain "
            f"after coastal subtraction. Check tolerance_deg "
            f"({config['Delta_masks']['tolerance_deg']}) or basin/coast alignment."
        )

    result: BaseGeometry = unary_union(lines)

    if not isinstance(result, (LineString, MultiLineString)):
        raise ValueError(
            f"[clip_basin_boundary_from_coast] Unexpected geometry type after "
            f"union: {result.geom_type}. Expected LineString or MultiLineString."
        )

    return result


# ---------------------------------------------------------------------------
# Downstream ID parsing
# ---------------------------------------------------------------------------


def _parse_downstream_ids(rch_id_dn: str | None) -> list[int]:
    """Parse a downstream reach ID field that may encode bifurcations.

    Bifurcations are encoded as space-separated integer IDs, e.g.
    ``'74223100024 74221000031'``. Returns an empty list when the value
    signals no downstream reach.

    Args:
        rch_id_dn: Raw downstream reach ID string from the SWORD dataset, or
            None if the field is missing.

    Returns:
        A list of integer downstream reach IDs, or an empty list if the value
        is absent, empty, ``'None'``, or ``'nan'``.

    Example:
        >>> _parse_downstream_ids("74223100024 74221000031")
        [74223100024, 74221000031]
        >>> _parse_downstream_ids(None)
        []
    """
    if rch_id_dn is None or str(rch_id_dn).strip() in ("", "None", "nan"):
        return []
    return [int(x) for x in str(rch_id_dn).split()]


# ---------------------------------------------------------------------------
# Downstream input-point logic
# ---------------------------------------------------------------------------


def _find_intersecting_reaches(
    rivers: GeoDataFrame,
    source_candidates: GeoDataFrame,
    snap_tolerance: float,
) -> GeoDataFrame:
    """Return river segments within *snap_tolerance* of any source candidate.

    Unions all candidate geometries into a single geometry and queries the
    river STRtree with a ``dwithin`` predicate for efficiency.

    Args:
        rivers: Full river network GeoDataFrame for the delta region.
        source_candidates: GeoDataFrame of candidate discharge cells (points).
        snap_tolerance: Maximum distance in degrees between a candidate cell
            and a river segment for the segment to be included.

    Returns:
        GeoDataFrame of river segments within *snap_tolerance* of any
        candidate cell, copied from *rivers* with original columns.

    Example:
        >>> reaches = _find_intersecting_reaches(rivers_gdf, cells_gdf, snap_tol)
        >>> print(f"{len(reaches)} candidate reaches found")
    """
    tree: STRtree = STRtree(rivers.geometry.values)
    union_geom: BaseGeometry = source_candidates.geometry.union_all()
    idx: np.ndarray = tree.query(
        union_geom, predicate="dwithin", distance=snap_tolerance
    )
    result: GeoDataFrame = rivers.iloc[idx].copy()
    print(f"River segments intersecting candidates: {len(result)}")
    return result


def _attach_candidate_points(
    candidate_reaches: GeoDataFrame,
    source_candidates: GeoDataFrame,
    snap_tolerance: float,
) -> GeoDataFrame:
    """Add a ``candidate_point`` column with the nearest point on each reach.

    For each candidate reach, finds the nearest source candidate within
    *snap_tolerance* and computes the closest point on the river geometry to
    that candidate. Uses an STRtree bulk query to avoid a Python-level loop.

    Args:
        candidate_reaches: GeoDataFrame of river segments near the source
            candidates, typically the output of
            :func:`_find_intersecting_reaches`.
        source_candidates: GeoDataFrame of candidate discharge cells (points).
        snap_tolerance: Maximum distance in degrees to search for a matching
            candidate point.

    Returns:
        Copy of *candidate_reaches* with an additional ``candidate_point``
        column containing a :class:`~shapely.geometry.Point` for each reach
        that matched a candidate, or ``None`` for unmatched reaches.

    Example:
        >>> reaches = _attach_candidate_points(reaches_gdf, cells_gdf, snap_tol)
        >>> print(reaches["candidate_point"].notna().sum())
        4
    """
    candidate_reaches = candidate_reaches.copy()
    src_tree: STRtree = STRtree(source_candidates.geometry.values)

    hits: np.ndarray = src_tree.query(
        candidate_reaches.geometry.values,
        predicate="dwithin",
        distance=snap_tolerance,
    )

    if hits.size == 0:
        candidate_reaches["candidate_point"] = None
        return candidate_reaches

    # hits shape: (2, n) — row 0: index into candidate_reaches,
    #                       row 1: index into source_candidates.
    # Keep only the first matching source for each reach.
    mapping: dict[int, int] = {}
    for reach_idx, src_idx in zip(hits[0], hits[1]):
        if reach_idx not in mapping:
            mapping[reach_idx] = src_idx

    points: list[Point | None] = []
    for i, geom in enumerate(candidate_reaches.geometry.values):
        src_idx = mapping.get(i)
        if src_idx is None:
            points.append(None)
            continue
        candidate_geom = source_candidates.geometry.iloc[src_idx]
        p_on_river, _ = nearest_points(geom, candidate_geom)
        points.append(p_on_river)

    candidate_reaches["candidate_point"] = points
    return candidate_reaches


def _deduplicate_by_point(candidate_reaches: GeoDataFrame) -> GeoDataFrame:
    """Keep only the reach with the largest ``dist_out`` per unique candidate point.

    When multiple reaches snap to the same candidate point, retaining the one
    with the largest ``dist_out`` value ensures the most-downstream reach is
    kept.

    Args:
        candidate_reaches: GeoDataFrame of candidate reaches with a
            ``candidate_point`` column and a ``dist_out`` column.

    Returns:
        Deduplicated copy of *candidate_reaches* with at most one row per
        unique ``candidate_point`` value.

    Example:
        >>> deduped = _deduplicate_by_point(candidate_reaches_gdf)
        >>> print(len(deduped) <= len(candidate_reaches_gdf))
        True
    """
    return cast(
        GeoDataFrame,
        (
            candidate_reaches.sort_values("dist_out", ascending=False).drop_duplicates(
                subset=["candidate_point"], keep="first"
            )
        ),
    ).copy()


def _build_upstream_map(rivers: GeoDataFrame) -> dict[int, list[int]]:
    """Map each ``reach_id`` to the list of ``reach_id`` values directly upstream.

    Parses the ``rch_id_dn`` (downstream reach ID) column from *rivers* to
    invert the flow direction: each reach stores which reaches drain into it,
    enabling upstream traversal.

    Args:
        rivers: River network GeoDataFrame containing ``reach_id`` and
            ``rch_id_dn`` columns.

    Returns:
        Dictionary mapping each reach ID to a list of IDs of the reaches
        directly upstream of it. Reaches with no upstream tributaries are not
        present as keys.

    Example:
        >>> upstream_map = _build_upstream_map(rivers_gdf)
        >>> print(upstream_map[74223100024])
        [74223100025, 74223100026]
    """
    upstream_map: dict[int, list[int]] = defaultdict(list)
    row: pd.Series
    for _, row in rivers.iterrows():
        for dn_id in _parse_downstream_ids(str(row["rch_id_dn"])):
            upstream_map[dn_id].append(int(row["reach_id"]))
    return upstream_map


def _trace_upstream(
    start_id: int,
    upstream_map: dict[int, list[int]],
    max_steps: int = 100,
) -> set[int]:
    """Return all reach IDs reachable upstream of *start_id* within *max_steps*.

    Uses an iterative depth-first traversal to avoid recursion-depth issues on
    large networks. The *max_steps* guard prevents infinite loops on malformed
    or cyclic river topologies.

    Args:
        start_id: The reach ID to trace upstream from.
        upstream_map: Mapping from reach ID to upstream reach IDs, as returned
            by :func:`_build_upstream_map`.
        max_steps: Maximum number of traversal steps before stopping. Defaults
            to 100. Increase for very large catchments.

    Returns:
        Set of reach IDs that are upstream of *start_id*, not including
        *start_id* itself.

    Example:
        >>> upstream = _trace_upstream(74223100024, upstream_map)
        >>> print(74223100025 in upstream)
        True
    """
    visited: set[int] = set()
    stack: list[int] = [start_id]
    steps: int = 0

    while stack and steps < max_steps:
        current: int = stack.pop()
        steps += 1
        for upstream_id in upstream_map.get(current, []):
            if upstream_id not in visited:
                visited.add(upstream_id)
                stack.append(upstream_id)

    return visited


def _remove_upstream_candidates(
    candidate_reaches: GeoDataFrame,
    upstream_map: dict[int, list[int]],
) -> GeoDataFrame:
    """Discard candidates that appear in the upstream network of another candidate.

    If candidate A's ``reach_id`` is found in the upstream set of candidate B,
    then A is upstream of B and is redundant — retaining B is sufficient. A is
    therefore discarded.

    Args:
        candidate_reaches: GeoDataFrame of candidate reaches, each with a
            ``reach_id`` column.
        upstream_map: Mapping from reach ID to upstream reach IDs, as returned
            by :func:`_build_upstream_map`.

    Returns:
        Filtered copy of *candidate_reaches* containing only the
        most-downstream (non-redundant) candidates.

    Example:
        >>> pruned = _remove_upstream_candidates(candidate_reaches_gdf, upstream_map)
        >>> print(len(pruned) <= len(candidate_reaches_gdf))
        True
    """
    candidate_ids: set[int] = set(candidate_reaches["reach_id"])

    upstream_sets: dict[int, set[int]] = {
        rid: _trace_upstream(rid, upstream_map) for rid in candidate_ids
    }

    to_discard: set[int] = {
        cid
        for cid in candidate_ids
        for other_id in candidate_ids
        if other_id != cid and cid in upstream_sets[other_id]
    }

    return cast(
        GeoDataFrame,
        candidate_reaches[~candidate_reaches["reach_id"].isin(list(to_discard))],
    ).copy()


def find_downstream_input_points(
    rivers: GeoDataFrame,
    source_candidates: GeoDataFrame,
    buffer: float,
) -> GeoDataFrame:
    """Identify the most-downstream unique inflow points from *source_candidates*.

    Runs the full downstream-selection pipeline: find intersecting reaches,
    attach candidate points, deduplicate, build the upstream map, and prune
    upstream-redundant candidates.

    Args:
        rivers: River network GeoDataFrame with ``reach_id`` and ``rch_id_dn``
            columns. Duplicate ``reach_id`` values are dropped before
            processing.
        source_candidates: GeoDataFrame of candidate discharge cells (points)
            near the domain boundary.
        buffer: Snap tolerance in degrees used in both
            :func:`_find_intersecting_reaches` and
            :func:`_attach_candidate_points`.

    Returns:
        GeoDataFrame of :class:`~shapely.geometry.Point` geometries in
        ``CRS_STANDARD``, one per unique most-downstream inflow location.
        Returns an empty GeoDataFrame if no valid candidates remain.

    Example:
        >>> sources = find_downstream_input_points(rivers_gdf, cells_gdf, 0.025)
        >>> print(f"{len(sources)} inflow points identified")
    """
    rivers = cast(
        GeoDataFrame, rivers.drop_duplicates(subset=["reach_id"], keep="last")
    )

    candidate_reaches: GeoDataFrame = _find_intersecting_reaches(
        rivers, source_candidates, buffer
    )
    candidate_reaches = _attach_candidate_points(
        candidate_reaches, source_candidates, buffer
    )
    candidate_reaches = _deduplicate_by_point(candidate_reaches)

    upstream_map: dict[int, list[int]] = _build_upstream_map(rivers)
    candidate_reaches = _remove_upstream_candidates(candidate_reaches, upstream_map)

    valid_points = candidate_reaches["candidate_point"].dropna()
    result: GeoDataFrame = cast(
        GeoDataFrame,
        gpd.GeoDataFrame(
            geometry=[Point(p) for p in valid_points],
            crs=CRS_STANDARD,
        ),
    )

    print(f"Final downstream points: {len(result)}")
    return result


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_cells_within_delta(
    q_values_nc: xr.DataArray,
    model_domain_mls: MultiLineString | LineString,
    rivers: GeoDataFrame,
    delta_edmonds: GeoDataFrame,
    basin_polygons: GeoDataFrame,
    basin_polygons_domain: GeoDataFrame,
    all_rivers: GeoDataFrame,
    threshold: float = _THRESHOLD,
    var_name: str = _VAR,
) -> tuple[GeoDataFrame, GeoDataFrame, GeoSeries]:
    """Extract GloFAS cells within the delta domain and identify river inflow points.

    Applies three sequential spatial filters to the GloFAS discharge grid —
    threshold, boundary proximity, and river proximity — then identifies the
    most-downstream unique inflow points using the SWORD river network. A
    diagnostic or summary plot is always saved.

    Args:
        q_values_nc: GloFAS discharge DataArray with ``longitude`` and
            ``latitude`` dimensions and a ``valid_time`` dimension already
            reduced to a per-cell minimum before calling this function.
        model_domain_mls: Inland boundary geometry of the delta domain, used
            to filter cells by proximity. Typically the output of
            :func:`clip_basin_boundary_from_coast`.
        rivers: River network GeoDataFrame clipped to the delta basin. Must
            contain ``reach_id``, ``rch_id_dn``, and ``dist_out`` columns.
        delta_edmonds: Single-row GeoDataFrame with the Edmonds delta polygon,
            passed through to :func:`plot_river_locations`.
        basin_polygons: Full set of basin polygons, passed through to
            :func:`plot_river_locations` for background context.
        basin_polygons_domain: Basin polygons for the current delta only,
            passed through to :func:`plot_river_locations`.
        all_rivers: Full river network for the region, passed through to
            :func:`plot_river_locations` for context.
        threshold: Minimum discharge in m³/s for a GloFAS cell to be
            considered. Defaults to ``config['Rivers']['glofas_min_discharge']``.
        var_name: Name of the discharge variable in *q_values_nc*. Defaults to
            ``config['Rivers']['glofas_discharge_parameter']``.

    Returns:
        A tuple of ``(unique_sources, possible_sources, buffered_cells)`` where:

        - *unique_sources*: GeoDataFrame of the most-downstream unique inflow
          points in ``CRS_STANDARD``.
        - *possible_sources*: GeoDataFrame of all candidate cells that passed
          the boundary and river proximity filters.
        - *buffered_cells*: GeoSeries of buffered geometries for all
          above-threshold cells, useful for visualisation.

    Example:
        >>> unique, possible, buffered = extract_cells_within_delta(
        ...     glofas_min, inland_boundary, rivers_gdf,
        ...     delta_edmonds, basin_polygons, basin_polygons_domain, all_rivers
        ... )
        >>> print(f"{len(unique)} unique inflow points found")
    """
    rivers = ensure_valid_schema(rivers, excluded=[AREA_COL, BASIN_COL])

    # --- Threshold filter ---
    all_cells: GeoDataFrame = _filter_cells_by_threshold(
        q_values_nc, threshold, var_name, _LON, _LAT
    )

    # --- Grid resolution ---
    half_cell: float = _compute_half_cell(q_values_nc, _LON, _LAT)
    buffered_cells: GeoSeries = all_cells.geometry.buffer(half_cell)

    # --- Boundary filter ---
    cells_boundary: GeoDataFrame = _filter_cells_near_boundary(
        all_cells, model_domain_mls, half_cell
    )
    print(f"Cells on boundary: {len(cells_boundary)}")

    # Always save a plot — debug mode if no boundary cells were found.
    plot_river_locations(
        all_cells,
        model_domain_mls,
        rivers,
        delta_edmonds,
        basin_polygons,
        basin_polygons_domain,
        all_rivers,
        debugging=(len(cells_boundary) == 0),
    )

    # --- River proximity filter ---
    rivers_union: BaseGeometry = unary_union(rivers.geometry)
    cells_boundary_rivers: GeoDataFrame = _filter_cells_near_rivers(
        cells_boundary, rivers_union, half_cell
    )

    # --- Downstream point selection ---
    unique_sources: GeoDataFrame = find_downstream_input_points(
        rivers, cells_boundary_rivers, half_cell
    )

    return unique_sources, cells_boundary_rivers, buffered_cells
