"""Domain bounding-box utilities shared across workflow steps."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box


def bounds_to_wgs84(
    bounds: dict[str, float],
    crs_str: str,
) -> tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max) in EPSG:4326 for an xmin/ymin/xmax/ymax dict.

    Reprojects all four corners, so the result encloses the rotated source box.
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
    """Return the WGS84 clipping bounds, UTM CRS string and WGS84 domain polygon.

    Bounds and CRS come from domain_bbox.json. The polygon is the delta polygon
    from domain.gpkg when given, else the bbox of the JSON bounds.
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
