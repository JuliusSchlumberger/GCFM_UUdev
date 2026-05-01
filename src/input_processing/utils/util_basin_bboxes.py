"""Utility to convert basin polygons to buffered bounding boxes per delta."""

from __future__ import annotations

from typing import Final

import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import box

from src.utils.config_loader import load_config

_CONFIG_PATH = "../src/input_processing/config/decisions.yaml"
_CONFIG: Final[dict] = load_config(_CONFIG_PATH)  # type: ignore[type-arg]

BASIN_COL: Final[str] = _CONFIG["DomainSchema"]["delta_id_lbl"]


def basins_to_buffered_bboxes(
    basin_gdf: GeoDataFrame,
    buffer_deg: float,
    basin_col: str = BASIN_COL,
) -> GeoDataFrame:
    """Convert basin polygons to one buffered bounding box per unique delta.

    Dissolves all polygons sharing a ``basin_col`` value into a single
    bounding box, expands it by ``buffer_deg`` on all sides, and returns a
    GeoDataFrame with one row per unique basin ID.

    Args:
        basin_gdf: Input GeoDataFrame containing basin polygons or
            multipolygons, potentially with multiple rows per basin ID.
        buffer_deg: Buffer distance in degrees (or CRS units) added to each
            bounding box on all sides.
        basin_col: Column name identifying unique delta basins. Defaults to
            the project-configured delta ID label.

    Returns:
        GeoDataFrame with one row per unique basin ID, a ``box`` geometry
        column containing the buffered bounding box, and the ``basin_col``
        column. The CRS matches the input.

    Example:
        >>> bboxes = basins_to_buffered_bboxes(basin_gdf, buffer_deg=0.1)
        >>> print(bboxes.columns.tolist())
        ['BasinID', 'geometry']
        >>> print(bboxes.geom_type.unique())
        ['Polygon']
    """

    def _bbox_with_buffer(geoms):  # type: ignore[no-untyped-def]
        total = geoms.union_all()
        minx, miny, maxx, maxy = total.bounds
        return box(
            minx - buffer_deg,
            miny - buffer_deg,
            maxx + buffer_deg,
            maxy + buffer_deg,
        )

    bboxes = (
        basin_gdf.groupby(basin_col)["geometry"].apply(_bbox_with_buffer).reset_index()
    )

    return gpd.GeoDataFrame(bboxes, geometry="geometry", crs=basin_gdf.crs)
