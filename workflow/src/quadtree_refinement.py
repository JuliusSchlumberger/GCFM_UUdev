"""Quadtree refinement-zone polygons for SFINCS grids: river and coastal buffers."""

from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd

from src.geometry import pick_utm_crs

log = logging.getLogger(__name__)


def build_refinement_polygons(
    river_network_path: str,
    land_polygons_path: str,
    river_refinement_level: int,
    river_buffer_factor: float,
    coastal_refinement_enabled: bool,
    coastal_refinement_level: int,
    coastal_buffer_m: float,
    target_crs: str,
) -> gpd.GeoDataFrame:
    """
    Build quadtree refinement-zone polygons from the river network (and,
    optionally, land polygons), reprojected to ``target_crs``.

    ``target_crs`` must be the SFINCS grid's exact target CRS:
    ``quadtree_grid.create_from_region`` does not reproject refinement
    polygons internally, so they must already match.

    River reaches are buffered by ``width * river_buffer_factor`` (``width``
    is whichever SWORD attribute river_processing.width_column selected as
    canonical -- see src.river_network.normalize_channel_widths -- so this
    buffer always matches the SFINCS rivwth burning width); if
    ``coastal_refinement_enabled``, the coastline
    (land polygon boundary) is also buffered by ``coastal_buffer_m``. Overlap
    between the two zones needs no explicit resolution: hydromt_sfincs
    refines each polygon's footprint independently to its own level, so a
    cell covered by both ends up at the higher of the two levels regardless
    of which polygon is processed first.

    Args:
        river_network_path:      Path to river_network_processed.gpkg (must
                                  have a 'width' column).
        land_polygons_path:      Path to land_polygons.gpkg (OSM land).
        river_refinement_level:  Refinement level for the river buffer zone.
        river_buffer_factor:     River buffer distance = width * this factor.
        coastal_refinement_enabled: If False, no coastal zone is built (the
                                  coastal buffer covers the entire coastline
                                  perimeter and can make builds far slower).
        coastal_refinement_level: Refinement level for the coastal buffer zone.
        coastal_buffer_m:        Coastal buffer distance (m).
        target_crs:               Exact target CRS to reproject the result to.

    Returns:
        GeoDataFrame with columns [geometry, refinement_level] in ``target_crs``.
    """
    zones = []

    rivers = gpd.read_file(river_network_path)
    if not rivers.empty:
        metric_crs = pick_utm_crs(rivers) if rivers.crs.is_geographic else rivers.crs
        rivers_m = rivers.to_crs(metric_crs)
        buffer_dist = (
            rivers_m["width"].fillna(0.0).clip(lower=0.0) * river_buffer_factor
        )
        river_zone = gpd.GeoDataFrame(
            geometry=[rivers_m.geometry.buffer(buffer_dist).union_all()],
            crs=metric_crs,
        )
        river_zone["refinement_level"] = river_refinement_level
        zones.append(river_zone)
    else:
        log.warning(
            "build_refinement_polygons: river network is empty, skipping river refinement"
        )

    if coastal_refinement_enabled:
        land = gpd.read_file(land_polygons_path)
        if not land.empty:
            metric_crs = pick_utm_crs(land) if land.crs.is_geographic else land.crs
            land_m = land.to_crs(metric_crs)
            coastline = land_m.geometry.union_all().boundary
            coastal_zone = gpd.GeoDataFrame(
                geometry=[coastline.buffer(coastal_buffer_m)],
                crs=metric_crs,
            )
            coastal_zone["refinement_level"] = coastal_refinement_level
            zones.append(coastal_zone)
        else:
            log.warning(
                "build_refinement_polygons: land polygons are empty, skipping coastal refinement"
            )

    if not zones:
        log.warning(
            "build_refinement_polygons: no river or coastal refinement zones found"
        )
        return gpd.GeoDataFrame(
            geometry=[], columns=["refinement_level"], crs=target_crs
        )

    refinement_gdf = gpd.GeoDataFrame(
        pd.concat([z.to_crs(target_crs) for z in zones], ignore_index=True),
        crs=target_crs,
    )
    # quadtree_builder.refine_in_polygon() expects a simple Polygon per row
    # (it reads geometry.exterior directly) — union_all()/coastline buffering
    # can produce MultiPolygons when a buffer zone has disjoint parts.
    refinement_gdf = refinement_gdf.explode(index_parts=False).reset_index(drop=True)
    return refinement_gdf
