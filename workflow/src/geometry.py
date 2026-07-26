import math

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box
from shapely.ops import nearest_points


def pick_utm_crs(gdf: gpd.GeoDataFrame) -> str:
    """Pick an appropriate UTM CRS from the centroid of a (lat/lon) GeoDataFrame."""
    # Project to centroid in geographic coords, then derive UTM zone
    centroid = gdf.to_crs("EPSG:4326").geometry.unary_union.centroid
    lon, lat = centroid.x, centroid.y
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def buffered_bbox(
    geom_gdf: gpd.GeoDataFrame, buffer_m: float, target_crs: str, source_crs=None
):
    """
    Reproject to `target_crs`, compute the bbox, buffer it by `buffer_m`,
    and return (bbox_gdf_in_target_crs, bounds_tuple).
    """
    # TODO: ensure that buffered box is oriented alongside the shoreline, not just a simple lat/lon aligned box
    if source_crs is not None and geom_gdf.crs is None:
        geom_gdf = geom_gdf.set_crs(source_crs)

    projected = geom_gdf.to_crs(target_crs)
    minx, miny, maxx, maxy = projected.total_bounds

    # Buffer the bbox itself (simple and predictable)
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
    """
    Filter points to those within ``region_wgs84`` (buffered by
    ``buffer_deg``), then nudge any kept point that falls just outside the
    *exact* (unbuffered) region back inside by ``nudge_m``.

    ``always_keep=True`` skips the buffered-membership filter entirely and
    snaps EVERY point into the region regardless of how far outside it
    originally was (``keep_mask`` is then all-``True``) -- for callers that
    need exactly one point per input row no matter what (e.g. one
    calibration observation point per river reach, where dropping a point
    means that reach silently gets no calibrated depth), as opposed to the
    default "drop anything too far outside" behavior appropriate for
    points that may legitimately not belong to this basin at all (e.g.
    discharge crossings). The underlying nudge-to-boundary mechanism below
    already works at any distance -- ``always_keep`` only changes whether
    far-outside points are dropped before it runs.

    hydromt_sfincs's discharge_points.create() (and similar) clip locations
    against the model's own region using an UNBUFFERED 'intersects' check,
    raising when none survive. Its own `buffer=` parameter does NOT expand
    that check outward -- internally it computes
    `region.boundary.buffer(buffer).clip(self.model.region)`, which stays
    clipped to the unbuffered region regardless of buffer size, so it can
    never rescue a point that falls just outside. A small buffer is used
    here only to decide which points to keep; any kept point that falls just
    outside the exact (unbuffered) region is then snapped onto its boundary
    (nearest_points; a no-op for points already inside) so every kept point
    is guaranteed to satisfy hydromt's own 'intersects' check regardless of
    its own buffer setting.

    ``nearest_points()`` alone is not enough: it returns a point that is
    mathematically ON the region boundary, but GEOS's floating-point
    arithmetic can place it a hair outside, which can still fail hydromt's
    'intersects' check. Nudging ``nudge_m`` further from the boundary point,
    towards a point guaranteed to be inside the region
    (``representative_point()``), lands safely interior rather than balanced
    on a numerically fragile edge.

    Args:
        points:      GeoDataFrame of points, in EPSG:4326.
        region_wgs84: The model's active region (shapely geometry), in EPSG:4326.
        buffer_deg:  Buffer (degrees) around region_wgs84 used only to decide
                     which points to keep (typically ~grid resolution / 111_000).
        nudge_m:     Distance (m) to nudge a boundary-snapped point further
                     inward, past GEOS's floating-point fragility margin.
        always_keep: Skip the buffered-membership filter and snap every
                     point in, regardless of distance (see above).

    Returns:
        (filtered, keep_mask): ``filtered`` is the kept points (copy, geometry
        replaced with the snapped-if-needed version); ``keep_mask`` is a
        boolean array (len == len(points)) so callers can filter companion
        arrays (e.g. a discharge column) consistently with which points
        were kept.
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
