from __future__ import annotations

from collections import defaultdict
from typing import Final

import geopandas as gpd
import numpy as np
from shapely import get_parts
import xarray as xr
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import MultiLineString, LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, unary_union
from shapely.strtree import STRtree

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
    threshshold: float,
    var_name: str,
    longitude: str,
    latitude: str,
) -> GeoDataFrame:
    """Return GeoDataFrame of cells exceeding discharge threshold."""
    q_thresh = q_values_nc.where(q_values_nc > threshshold, drop=True)
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
    """Return half the maximum grid-cell size."""
    lon_res = float(q_values_nc[longitude].diff(longitude).median())
    lat_res = float(q_values_nc[latitude].diff(latitude).median())
    return max(abs(lon_res), abs(lat_res)) / 2.0


def _filter_cells_near_boundary(
    cells_gdf: GeoDataFrame,
    boundary: BaseGeometry,
    half_cell: float,
) -> GeoDataFrame:
    """Keep cells whose buffer intersects the boundary."""
    buffered: np.ndarray = cells_gdf.geometry.buffer(half_cell).values
    tree = STRtree(buffered)

    hits = tree.query(boundary, predicate="intersects")

    mask = np.zeros(len(cells_gdf), dtype=bool)
    if len(hits) > 0:
        mask[hits] = True

    return cells_gdf.loc[mask].copy()


def _filter_cells_near_rivers(
    cells_gdf: GeoDataFrame,
    rivers_union: BaseGeometry,
    half_cell: float,
) -> GeoDataFrame:
    """Keep only cells within half_cell distance of rivers."""
    tree = STRtree(cells_gdf.geometry.values)

    near = tree.query(rivers_union, predicate="dwithin", distance=half_cell)

    mask = np.zeros(len(cells_gdf), dtype=bool)
    if len(near) > 0:
        mask[near] = True

    result = cells_gdf.loc[mask].copy()

    print(f"Cells near rivers: {mask.sum()} / {len(cells_gdf)}")
    return result


def clip_basin_boundary_from_coast(
    relevant_basins: GeoDataFrame,
    coast_polygon: BaseGeometry,
    simplify_tolerance: float = 0.01,
) -> LineString | MultiLineString:
    """
    Return the full inland basin boundary after removing coastal parts.

    Dissolves the basin polygons, computes the outer boundary, subtracts a
    buffered and simplified coastal zone, and returns all remaining line
    segments reunited as a single (Multi)LineString.

    Args:
        relevant_basins:    Basin polygons for one delta, in CRS_STANDARD.
        coast_polygon:      Coastal geometry used to mask the seaward boundary.
        simplify_tolerance: Simplification tolerance in degrees applied to the
                            buffered coast before .difference() to prevent
                            hangs on highly fragmented coastlines (default
                            0.01° ≈ 1 km at the equator).

    Returns:
        The complete inland boundary as a LineString (single segment) or
        MultiLineString (multiple disconnected segments).

    Raises:
        ValueError: If no inland boundary segments remain after coastal
                    subtraction, or if the result is not a line geometry.
    """
    basin_boundary: BaseGeometry = relevant_basins.union_all().boundary

    buffered_coast: BaseGeometry = coast_polygon.buffer(
        config["Delta_masks"]["tolerance_deg"]
    )

    # Simplify before .difference() — complex/fragmented coastlines make
    # .difference() scale very poorly without this step.
    buffered_coast = buffered_coast.simplify(simplify_tolerance, preserve_topology=True)

    clipped: BaseGeometry = basin_boundary.difference(buffered_coast)

    # get_parts() handles any geometry type uniformly — unlike .geoms which
    # raises AttributeError on a bare LineString.
    lines: list[LineString] = [
        g for g in get_parts(clipped) if g.geom_type == "LineString"
    ]

    if not lines:
        raise ValueError(
            f"[clip_basin_boundary_from_coast] No inland boundary segments remain "
            f"after coastal subtraction. Check tolerance_deg "
            f"({config['Delta_masks']['tolerance_deg']}) or basin/coast alignment."
        )

    # unary_union on lines returns LineString (one segment) or MultiLineString
    # (multiple disconnected segments) — both are valid return values.
    result: BaseGeometry = unary_union(lines)

    if not isinstance(result, (LineString, MultiLineString)):
        raise ValueError(
            f"[clip_basin_boundary_from_coast] Unexpected geometry type after "
            f"union: {result.geom_type}. Expected LineString or MultiLineString."
        )

    return result


# ---------------------------------------------------------------------------
# Downstream ID handling
# ---------------------------------------------------------------------------


def _parse_downstream_ids(rch_id_dn: str | None) -> list[int]:
    """Parse downstream IDs (handles bifurcations)."""
    if rch_id_dn is None or str(rch_id_dn).strip() in ("", "None", "nan"):
        return []
    return [int(x) for x in str(rch_id_dn).split()]


# ---------------------------------------------------------------------------
# Downstream input-point logic
# ---------------------------------------------------------------------------


def _find_intersecting_reaches(
    rivers: GeoDataFrame, source_candidates: GeoDataFrame, snap_tolerance: float
) -> GeoDataFrame:
    """Find river segments near candidate cells."""
    tree = STRtree(rivers.geometry.values)

    union_geom = source_candidates.geometry.union_all()

    idx = tree.query(union_geom, predicate="dwithin", distance=snap_tolerance)

    result = rivers.iloc[idx].copy()
    print(f"River segments intersecting candidates: {len(result)}")

    return result


def _attach_candidate_points(
    candidate_reaches: GeoDataFrame,
    source_candidates: GeoDataFrame,
    snap_tolerance: float,
) -> GeoDataFrame:
    """Attach nearest candidate point to each reach."""
    candidate_reaches = candidate_reaches.copy()

    src_tree = STRtree(source_candidates.geometry.values)

    hits = src_tree.query(
        candidate_reaches.geometry.values, predicate="dwithin", distance=snap_tolerance
    )

    if hits.size == 0:
        candidate_reaches["candidate_point"] = None
        return candidate_reaches

    mapping = {}
    for reach_idx, src_idx in zip(hits[0], hits[1]):
        if reach_idx not in mapping:
            mapping[reach_idx] = src_idx

    points = []
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
    """Keep reach with largest dist_out per point."""
    return (
        candidate_reaches.sort_values("dist_out", ascending=False)
        .drop_duplicates(subset=["candidate_point"], keep="first")
        .copy()
    )


def _build_upstream_map(rivers: GeoDataFrame) -> dict[int, list[int]]:
    """Map reach_id → upstream reach_ids."""
    upstream_map: dict[int, list[int]] = defaultdict(list)

    for row in rivers.itertuples(index=False):
        for dn_id in _parse_downstream_ids(row.rch_id_dn):
            upstream_map[dn_id].append(int(row.reach_id))

    return upstream_map


def _trace_upstream(
    start_id: int,
    upstream_map: dict[int, list[int]],
    max_steps: int = 100,
) -> set[int]:
    """Trace upstream network."""
    visited: set[int] = set()
    stack: list[int] = [start_id]

    steps: int = 0
    while stack:
        current = stack.pop()
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
    """Remove upstream-redundant candidates."""
    candidate_ids = set(candidate_reaches["reach_id"])

    upstream_sets = {rid: _trace_upstream(rid, upstream_map) for rid in candidate_ids}

    to_discard: set[int] = {
        cid
        for cid in candidate_ids
        for other_id in candidate_ids
        if other_id != cid and cid in upstream_sets[other_id]
    }

    return candidate_reaches[~candidate_reaches["reach_id"].isin(to_discard)].copy()


def find_downstream_input_points(
    rivers: GeoDataFrame,
    source_candidates: GeoDataFrame,
    buffer: float,
) -> GeoDataFrame:
    """Return most-downstream unique input points."""
    rivers = rivers.drop_duplicates(subset=["reach_id"], keep="last")

    candidate_reaches = _find_intersecting_reaches(rivers, source_candidates, buffer)

    candidate_reaches = _attach_candidate_points(
        candidate_reaches, source_candidates, buffer
    )

    candidate_reaches = _deduplicate_by_point(candidate_reaches)

    upstream_map = _build_upstream_map(rivers)

    candidate_reaches = _remove_upstream_candidates(candidate_reaches, upstream_map)

    valid_points = candidate_reaches["candidate_point"].dropna()

    result = gpd.GeoDataFrame(
        geometry=[Point(p) for p in valid_points],
        crs=CRS_STANDARD,
    )

    print(f"Final downstream points: {len(result)}")
    return result


# ---------------------------------------------------------------------------
# Main function
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
    """
    Extract GloFAS cells within delta domain and identify river input points.
    """

    rivers = ensure_valid_schema(rivers, excluded=[AREA_COL, BASIN_COL])

    # Threshold filter
    all_cells = _filter_cells_by_threshold(q_values_nc, threshold, var_name, _LON, _LAT)

    # Grid resolution
    half_cell = _compute_half_cell(q_values_nc, _LON, _LAT)
    buffered_cells = all_cells.geometry.buffer(half_cell)

    # Boundary filter
    cells_boundary = _filter_cells_near_boundary(all_cells, model_domain_mls, half_cell)

    print(f"Cells on boundary: {len(cells_boundary)}")

    if len(cells_boundary) == 0:
        plot_river_locations(
            all_cells,
            model_domain_mls,
            rivers,
            delta_edmonds,
            basin_polygons,
            basin_polygons_domain,
            all_rivers,
            debugging=True,
        )
    else:
        plot_river_locations(
            all_cells,
            model_domain_mls,
            rivers,
            delta_edmonds,
            basin_polygons,
            basin_polygons_domain,
            all_rivers,
            debugging=False,
        )

    # River filter
    rivers_union = unary_union(rivers.geometry)

    cells_boundary_rivers = _filter_cells_near_rivers(
        cells_boundary, rivers_union, half_cell
    )

    # Downstream points
    unique_sources = find_downstream_input_points(
        rivers, cells_boundary_rivers, half_cell
    )

    return unique_sources, cells_boundary_rivers, buffered_cells
