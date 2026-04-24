from shapely.geometry import Polygon
import geopandas as gpd
from shapely.geometry.multipolygon import MultiPolygon

from src.input_processing.config.loader import config
from src.input_processing.utils.validation.modify_delta_masks import build_local_bbox
from geopandas import GeoDataFrame
import xarray as xr

from dataclasses import dataclass


@dataclass(frozen=True)
class GlobalData:
    rivers: GeoDataFrame
    coastline: GeoDataFrame
    glofas: xr.Dataset


def load_global_data() -> GlobalData:
    """Load heavy datasets once."""
    rivers = gpd.read_file(config["filepaths"]["river_sword"]).to_crs(
        epsg=config["CRS"]["standard"]
    )

    coastline = gpd.read_file(config["filepaths"]["coastline"]).to_crs(
        epsg=config["CRS"]["standard"]
    )

    glofas = xr.open_dataset(config["filepaths"]["glofas"])
    glofas = glofas.rio.write_crs(config["CRS"]["standard"])

    return GlobalData(
        rivers=rivers,
        coastline=coastline,
        glofas=glofas,
    )


from shapely.geometry import MultiPolygon
from shapely.geometry.base import BaseGeometry


def load_data_delta_domain(
    delta_basin_mask: GeoDataFrame | Polygon | MultiPolygon,
    global_data: GlobalData,
) -> tuple[GeoDataFrame, Polygon, GeoDataFrame, xr.DataArray]:
    """
    Clip global datasets to a delta-specific bounding box.

    Returns:
        rivers_gpd, coast_polygon, coastline_gpd, glofas_min
    """

    # --- Normalize geometry ---
    if isinstance(delta_basin_mask, (Polygon, MultiPolygon)):
        delta_polygon: BaseGeometry = delta_basin_mask
    else:
        delta_polygon = delta_basin_mask.geometry.union_all()

    # --- Bounding box ---
    bbox_4326 = build_local_bbox(delta_polygon)
    minx, miny, maxx, maxy = bbox_4326.bounds

    # --- Clip rivers (FAST, no I/O) ---
    rivers_gpd = global_data.rivers.cx[minx:maxx, miny:maxy].copy()

    # --- Clip coastline ---
    coastline_gpd = global_data.coastline.cx[minx:maxx, miny:maxy].copy()
    coast_polygon: BaseGeometry = coastline_gpd.geometry.union_all().simplify(
        config["Delta_masks"]["tolerance_deg"], preserve_topology=True
    )

    # --- Clip GloFAS (NO re-open) ---
    glofas_clip = global_data.glofas.sel(
        longitude=slice(minx, maxx),
        latitude=slice(maxy, miny),
    )

    glofas_min: xr.DataArray = glofas_clip.dis24.min(dim="valid_time")

    return rivers_gpd, coast_polygon, coastline_gpd, glofas_min
