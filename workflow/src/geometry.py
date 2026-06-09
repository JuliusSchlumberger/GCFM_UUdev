import geopandas as gpd
from shapely.geometry import box


def select_intersecting(
    candidates: gpd.GeoDataFrame, reference: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Return rows of `candidates` whose geometry intersects `reference`."""
    if candidates.crs != reference.crs:
        reference = reference.to_crs(candidates.crs)
    # spatial join is faster than a naive intersects loop on large datasets
    joined = gpd.sjoin(
        candidates,
        reference[["geometry"]],
        how="inner",
        predicate="intersects",
    )
    return candidates.loc[joined.index.unique()].copy()


def merge_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve all features into a single geometry, preserving CRS."""
    dissolved = gdf.dissolve()
    return dissolved[["geometry"]]


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
