"""Domain bounding-box utilities shared across workflow steps."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from pyproj import CRS, Transformer
from shapely.geometry import box, Polygon


def bounds_to_wgs84(
    bounds: dict[str, float],
    crs_str: str,
) -> tuple[float, float, float, float]:
    """
    Convert a bounding-box dict from any CRS to WGS84.

    Args:
        bounds: Dict with keys 'xmin', 'ymin', 'xmax', 'ymax' in `crs_str` units.
        crs_str: Any CRS string accepted by pyproj (e.g. 'EPSG:32651').

    Returns:
        (lon_min, lat_min, lon_max, lat_max) in EPSG:4326.
    """
    src = CRS.from_string(crs_str)
    xmin, ymin, xmax, ymax = (
        bounds["xmin"],
        bounds["ymin"],
        bounds["xmax"],
        bounds["ymax"],
    )
    if src.equals(CRS.from_epsg(4326)):
        return xmin, ymin, xmax, ymax
    t = Transformer.from_crs(src, CRS.from_epsg(4326), always_xy=True)
    corners = [
        t.transform(x, y)
        for x, y in [(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)]
    ]
    lons, lats = zip(*corners)
    return min(lons), min(lats), max(lons), max(lats)


def load_domain(
    meta_path: str | Path,
    domain_gpkg_path: str | Path | None = None,
) -> tuple[tuple[float, float, float, float], str, Polygon]:
    """
    Load domain metadata and return the WGS84 clipping bounds, CRS, and domain polygon.

    The metadata JSON (domain_bbox.json) holds the clipping bbox bounds. The
    actual domain polygon (the delta polygon) is read from domain.gpkg when
    ``domain_gpkg_path`` is provided; otherwise the bbox polygon derived from
    the JSON bounds is used as a fallback.

    Args:
        meta_path:        Path to domain_bbox.json.
        domain_gpkg_path: Optional path to domain.gpkg.  When supplied, its geometry
                          is returned as the domain polygon (in WGS84).

    Returns:
        wgs84_bounds: (lon_min, lat_min, lon_max, lat_max) clipping extent in WGS84.
        crs_str:      UTM CRS string for the domain (e.g. 'EPSG:32651').
        domain_poly:  Shapely Polygon of the domain in WGS84.
                      From domain.gpkg if provided, else bbox derived from JSON bounds.
    """
    with open(meta_path) as f:
        meta = json.load(f)
    wgs84 = bounds_to_wgs84(meta["bounds"], meta["crs"])

    if domain_gpkg_path is not None:
        gdf = gpd.read_file(domain_gpkg_path).to_crs("EPSG:4326")
        domain_poly = gdf.geometry.unary_union
    else:
        domain_poly = box(*wgs84)

    return wgs84, meta["crs"], domain_poly
