"""
river_preburn.py — Compute river bed points for SFINCS subgrid burning.

For each pixel along every river centerline:
    rivbed = DEM_conditioned[pixel] - rivdph

where rivdph is the hydraulic depth of the reach (constant per SWORD reach,
derived from discharge/width via the power-law or estuarine depth model).

The returned GeoDataFrame is passed to HydroMT-SFINCS's burn_river_rect via
    sf.subgrid.create(river_list=[{"centerlines": rivers, "gdf_zb": points}])

burn_river_rect interpolates along merged river centerlines and lowers the
subgrid DEM where rivbed < DEM — the DEM file itself is never modified.
rivdph in the centerline GDF is completely bypassed when gdf_zb is provided.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Point

from src.river_network import _as_linestring, _sample_line_cells, normalize_reach_id

log = logging.getLogger(__name__)


def compute_river_bed_points(
    rivers: gpd.GeoDataFrame,
    elevation_path: str | Path,
    depth_column: str = "rivdph",
) -> gpd.GeoDataFrame:
    """
    Sample absolute river bed elevations along all river centerlines.

    For every DEM-pixel-spaced point along each reach centerline:
        rivbed = DEM_conditioned[pixel] - rivdph

    rivdph is taken from the reach attribute (constant per SWORD reach).
    No monotonic enforcement is applied — the conditioned DEM is already
    monotonically non-increasing along the centerline; rivdph reflects the
    hydraulic geometry and can vary naturally between reaches.

    The output is intended as the gdf_zb argument to burn_river_rect inside
    sf.subgrid.create().  HydroMT-SFINCS will interpolate these points along
    the merged river centerlines and burn the bed into the subgrid DEM.

    Args:
        rivers:         River network GeoDataFrame (river_network_estuarine.gpkg).
                        Must have reach_id and depth_column attributes.
        elevation_path: Conditioned DEM (elevation_conditioned.tif).
        depth_column:   Column name for hydraulic depth (default "rivdph").

    Returns:
        GeoDataFrame of Point features in the DEM CRS with columns:
        geometry (Point), rivbed (float), reach_id, and all other columns
        from rivers.
    """
    with rasterio.open(elevation_path) as src:
        elevation_arr = src.read(1).astype(np.float32)
        transform = src.transform
        nodata = src.nodata
        raster_crs = src.crs

    step_m = abs(transform.a)
    raster_h, raster_w = elevation_arr.shape

    rivers_proj = (
        rivers.to_crs(raster_crs) if rivers.crs != raster_crs else rivers.copy()
    )

    records: list[dict] = []
    n_skipped = 0

    for row in rivers_proj.itertuples(index=False):
        rid = normalize_reach_id(row.reach_id)
        if rid is None:
            continue

        g = _as_linestring(row.geometry)
        if g is None or g.length == 0:
            continue

        d = getattr(row, depth_column, None)
        rivdph = float(d) if d is not None and np.isfinite(float(d)) else 0.0

        base_attrs = row._asdict()

        for c in _sample_line_cells(g, transform, (raster_h, raster_w), step_m):
            v = float(elevation_arr[c["row"], c["col"]])
            if (nodata is not None and v == nodata) or not np.isfinite(v):
                n_skipped += 1
                continue

            # World-coordinate centre of this pixel
            px = transform.c + (c["col"] + 0.5) * transform.a
            py = transform.f + (c["row"] + 0.5) * transform.e

            rec = dict(base_attrs)
            rec["geometry"] = Point(px, py)
            rec["rivbed"] = float(v - rivdph)
            records.append(rec)

    if not records:
        log.warning("compute_river_bed_points: no points produced")
        return gpd.GeoDataFrame(columns=["geometry", "rivbed"], crs=raster_crs)

    gdf = gpd.GeoDataFrame(records, crs=raster_crs)
    log.info(
        f"compute_river_bed_points: {len(gdf)} points across "
        f"{rivers_proj['reach_id'].nunique()} reach(es) "
        f"({n_skipped} nodata pixels skipped)"
    )
    return gdf
