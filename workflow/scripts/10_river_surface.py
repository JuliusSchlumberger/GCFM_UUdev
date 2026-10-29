"""Build the river water surface and its slope, and condition the DEM to it; rule river_surface."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from rasterio.crs import CRS as RasterioCRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from src.grid import load_grid_def
from src.log import setup_logging
from src.plots import plot_water_surface_profiles
from src.profiling import ScriptProfiler
from src.raster import FLOAT_NODATA
from src.river_network import normalize_reach_id
from src.surge import read_calm_sea_level_m
from src.water_surface import (
    build_water_surface,
    condition_channels_to_surface,
    node_sampler,
    sample_centrelines,
    write_water_surface,
)

log = setup_logging(snakemake.log[0])
profiler = ScriptProfiler(snakemake)
build_water_surface = profiler.wrap(build_water_surface)
condition_channels_to_surface = profiler.wrap(condition_channels_to_surface)

merged_path = snakemake.input.elevation_merged
landuse_path = snakemake.input.landuse
with rasterio.open(merged_path) as _src:
    dem_crs = _src.crs
rivers = gpd.read_file(snakemake.input.river_network).to_crs(dem_crs)
log.info(f"Loaded {len(rivers)} reaches from river_network_clean.gpkg")

with xr.open_dataset(snakemake.input.surge_forcing, decode_times=False) as _surge_ds:
    calm_sea_level_m = read_calm_sea_level_m(_surge_ds)
surface, rivers_oriented = build_water_surface(
    rivers,
    lambda lines: sample_centrelines(
        lines, merged_path, landuse_path, int(snakemake.params.water_code)
    ),
    dict(snakemake.params.water_surface),
    sea_level_m=calm_sea_level_m,
    sample_nodes=node_sampler(merged_path, snakemake.input.landuse_channels, dem_crs),
    gebco_floor_m=calm_sea_level_m - float(snakemake.params.gebco_max_depth_m),
)
write_water_surface(surface, snakemake.output.water_surface, dem_crs)
log.info(f"Written: {snakemake.output.water_surface}")
condition_channels_to_surface(
    surface,
    rivers_oriented,
    merged_path,
    landuse_path,
    snakemake.output.elevation_conditioned,
)
log.info(f"Written: {snakemake.output.elevation_conditioned}")

if hasattr(snakemake.output, "plot_water_surface"):
    seeds = sorted(
        normalize_reach_id(r)
        for r, s in zip(rivers_oriented["reach_id"], rivers_oriented.get("is_seed", []))
        if bool(s)
    )
    plot_water_surface_profiles(
        surface,
        rivers_oriented,
        seeds,
        snakemake.output.plot_water_surface,
        f"Basin {snakemake.wildcards.basin_id}: river water surface (rule 10) along each seed's widest arm",
    )

# Elevation ceiling for the active-cell mask in rules 11/12a
# (plus sfincs.grid.active_mask.elevation_buffer_m): the highest surface level.
centerline_max_m = float(np.nanmax(surface.level_m))
with open(snakemake.output.river_elevation_max, "w") as f:
    json.dump({"river_elevation_max_m": centerline_max_m}, f, indent=2)
log.info(f"Written: {snakemake.output.river_elevation_max} ({centerline_max_m:.2f} m)")

# Resample once onto the shared SFINCS grid so rules 11 and 12a use the
# identical coarse base layer for sf.elevation.create().
grid_def = load_grid_def(snakemake.input.sfincs_grid)
grid_transform = grid_def["transform"]
dst_crs = RasterioCRS.from_string(grid_def["crs"])

with rasterio.open(snakemake.output.elevation_conditioned) as src:
    src_arr = src.read(1)
    src_transform = src.transform
    src_crs = src.crs
    src_nodata = src.nodata if src.nodata is not None else FLOAT_NODATA
    src_bounds = rasterio.transform.array_bounds(src.height, src.width, src_transform)

# Cover the full native elevation extent, snapped to the grid phase, not just
# the grid bounds: HydroMT's elevation.create() reads a margin beyond the grid,
# and a nodata margin triggers "Dataset ... does not fully cover bbox ..." and
# spurious flooding. Rule 05a's buffer exceeds HydroMT's margin.
inv_grid = ~grid_transform
corners = [
    (src_bounds[0], src_bounds[1]),
    (src_bounds[2], src_bounds[1]),
    (src_bounds[0], src_bounds[3]),
    (src_bounds[2], src_bounds[3]),
]
cols, rows = zip(*(inv_grid * c for c in corners))
col_off = int(np.floor(min(cols)))
row_off = int(np.floor(min(rows)))
col_stop = int(np.ceil(max(cols)))
row_stop = int(np.ceil(max(rows)))
dst_transform = grid_transform * Affine.translation(col_off, row_off)
dst_shape = (row_stop - row_off, col_stop - col_off)

dst_arr = np.full(dst_shape, src_nodata, dtype=src_arr.dtype)
reproject(
    source=src_arr,
    destination=dst_arr,
    src_transform=src_transform,
    src_crs=src_crs,
    src_nodata=src_nodata,
    dst_transform=dst_transform,
    dst_crs=dst_crs,
    dst_nodata=src_nodata,
    resampling=Resampling.average,
)

dst_meta = dict(
    driver="GTiff",
    dtype=src_arr.dtype,
    width=dst_shape[1],
    height=dst_shape[0],
    count=1,
    crs=dst_crs,
    transform=dst_transform,
    nodata=src_nodata,
    compress="deflate",
)
Path(snakemake.output.elevation_conditioned_sfincs_grid).parent.mkdir(
    parents=True, exist_ok=True
)
with rasterio.open(
    snakemake.output.elevation_conditioned_sfincs_grid, "w", **dst_meta
) as dst:
    dst.write(dst_arr, 1)
log.info(
    f"Written: {snakemake.output.elevation_conditioned_sfincs_grid} "
    f"({dst_shape[0]}x{dst_shape[1]} px @ {grid_def['resolution']} m)"
)

profiler.stop()
log.info("Done")
