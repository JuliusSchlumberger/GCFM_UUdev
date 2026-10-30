"""Geometry helpers: UTM zone choice, buffered bounding boxes, point snapping."""

import math

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box
from shapely.ops import nearest_points


def pick_utm_crs(gdf: gpd.GeoDataFrame) -> str:
    """Return the UTM EPSG code (326xx north, 327xx south) at the GeoDataFrame centroid."""
    centroid = gdf.to_crs("EPSG:4326").geometry.union_all().centroid
    lon, lat = centroid.x, centroid.y
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def buffered_bbox(
    geom_gdf: gpd.GeoDataFrame, buffer_m: float, target_crs: str, source_crs=None
):
    """Return the bbox of geom_gdf in target_crs, grown by buffer_m, as (gdf, bounds).

    The box is axis-aligned in target_crs; source_crs is only applied when
    geom_gdf has no CRS.
    """
    # TODO: ensure that buffered box is oriented alongside the shoreline, not just a simple lat/lon aligned box
    if source_crs is not None and geom_gdf.crs is None:
        geom_gdf = geom_gdf.set_crs(source_crs)

    projected = geom_gdf.to_crs(target_crs)
    minx, miny, maxx, maxy = projected.total_bounds

    minx -= buffer_m
    miny -= buffer_m
    maxx += buffer_m
    maxy += buffer_m

    bbox_geom = gpd.GeoDataFrame(
        geometry=[box(minx, miny, maxx, maxy)],
        crs=target_crs,
    )
    return bbox_geom, (minx, miny, maxx, maxy)


def snap_points_into_region(
    points: gpd.GeoDataFrame,
    region_wgs84,
    buffer_deg: float,
    nudge_m: float = 1.0,
    always_keep: bool = False,
) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """Keep points within region_wgs84 + buffer_deg and nudge outside ones just inside.

    points and region_wgs84 are in EPSG:4326; buffer_deg (typically grid
    resolution / 111_000) only decides which points are kept. always_keep=True
    skips that filter and snaps every point in, for callers that need one point
    per row (e.g. one calibration point per reach). Returns (filtered, keep_mask)
    with keep_mask aligned to the input rows.

    hydromt_sfincs clips locations with an unbuffered 'intersects' check against
    the model region and raises when none survive; its own buffer= does not
    widen that check (region.boundary.buffer(buffer).clip(region)). A point
    snapped with nearest_points() can still land a hair outside due to GEOS
    floating point, so it is moved nudge_m (m) further towards
    representative_point() to sit safely inside.
    """
    if always_keep:
        keep_mask = np.ones(len(points), dtype=bool)
        filtered = points.copy()
    else:
        keep_mask = points.geometry.within(region_wgs84.buffer(buffer_deg)).to_numpy()
        filtered = points[keep_mask].copy()
    if filtered.empty:
        return filtered, keep_mask

    nudge_deg = nudge_m / 111_000.0
    interior_anchor = region_wgs84.representative_point()

    def _snap(g):
        if region_wgs84.contains(g):
            return g
        boundary_pt = nearest_points(region_wgs84, g)[0]
        dx = interior_anchor.x - boundary_pt.x
        dy = interior_anchor.y - boundary_pt.y
        dist = math.hypot(dx, dy)
        if dist == 0:
            return boundary_pt
        frac = min(nudge_deg / dist, 1.0)
        return Point(boundary_pt.x + dx * frac, boundary_pt.y + dy * frac)

    filtered["geometry"] = filtered.geometry.apply(_snap)
    return filtered, keep_mask
