"""Land use and Manning's n with the river channel, on the elevation and model grids.

Run by rule grid_align_landuse. Writes:

- landuse_channels.tif: landuse.tif (elevation grid) with every non-sea pixel of
  the river channel (src.river_burn.channel_polygons) set to landuse.water_code;
- roughness.tif: Manning's n on the elevation grid subdivided to about the
  land-use source's pixel size (~10 m under ESA WorldCover, the unchanged ~31 m
  grid under LC100), area-averaged from the source, channel at
  channel_manning_n. hydromt samples it per subgrid pixel and averages by
  conveyance;
- landuse_on_grid.tif, roughness_on_grid.tif, land_mask_on_grid.gpkg: the same on
  the SFINCS grid (rule build_sfincs_grid), in SfincsModel orientation (row 0 on
  the origin side), so every model assigns them to sf.grid.data cell for cell.
  landuse_on_grid.tif is the sea/land classification downstream (weir tracing,
  zsini sea cells, the water-level boundary).

Land use is categorical, so it is resampled by mode, never averaged. Manning's n
is area-averaged from the native source, not reclassified from a resampled land
use: the dominant class would give a 60 % water / 40 % built-up cell water's n.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from src.domain import load_domain
from src.grid import load_grid_def
from src.landuse import write_channel_landuse, write_roughness_raster
from src.log import setup_logging
from src.profiling import ScriptProfiler
from src.raster import vectorize_land_mask_on_grid
from src.river_burn import channel_polygons

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
write_roughness_raster = profiler.wrap(write_roughness_raster)

wgs84_bounds, _, _ = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}")
water_code = int(snakemake.params.water_code)
channel_n = float(snakemake.params.channel_manning_n)
method = str(snakemake.params.roughness_aggregation)
rivers = gpd.read_file(snakemake.input.river_network)

# Channel land use on the elevation grid
with rasterio.open(snakemake.input.landuse) as _src:
    elev_crs = _src.crs
converted = write_channel_landuse(
    snakemake.input.landuse,
    channel_polygons(rivers.to_crs(elev_crs)),
    elev_crs,
    water_code,
    snakemake.output.landuse_channels,
)
log.info(
    f"Written: {snakemake.output.landuse_channels} ({sum(converted.values()):,} channel pixel(s) "
    f"set to {water_code}; from class: {converted})"
)

# Roughness on the elevation grid, subdivided to the source's pixel size
with rasterio.open(snakemake.input.elevation_merged) as ref:
    ref_meta = ref.meta.copy()
unmapped, info = write_roughness_raster(
    snakemake.input.landuse_source,
    snakemake.input.matching_lu_roughness,
    ref_meta,
    snakemake.output.spec_roughness,
    method=method,
    refine_to_source=True,
    channel=(
        channel_polygons(rivers.to_crs(ref_meta["crs"])),
        ref_meta["crs"],
        channel_n,
    ),
)
if unmapped:
    log.warning(
        f"Land-use codes without roughness mapping (NaN, excluded from the average): "
        f"{sorted(unmapped)}"
    )
log.info(
    f"Roughness ({method}) from a {info['source_resolution_m']} m source -> "
    f"{info['shape'][1]}x{info['shape'][0]} px @ {info['resolution_m']} m "
    f"(elevation grid subdivided by {info['refine_factor']}). "
    f"Written: {snakemake.output.spec_roughness}"
)

# Land use and roughness on the SFINCS grid
resolution_m = float(snakemake.params.resolution_m)
grid_def = load_grid_def(snakemake.input.sfincs_grid)
grid_transform = grid_def["transform"]
grid_shape = grid_def["shape"]
log.info(
    f"Shared SFINCS grid: {grid_shape[0]}x{grid_shape[1]} cells @ {resolution_m} m, "
    f"rotation {grid_def['rotation']:.3f} deg"
)

with rasterio.open(snakemake.output.landuse_channels) as src:
    lu_arr = src.read(1)
    src_transform, src_crs, src_nodata = src.transform, src.crs, src.nodata

lu_on_grid = np.full(grid_shape, src_nodata, dtype=lu_arr.dtype)
# Mode: landuse (~30 m) is finer than this grid (~70 m), so nearest would
# keep one arbitrary source pixel per cell; classes are categorical.
reproject(
    source=lu_arr,
    destination=lu_on_grid,
    src_transform=src_transform,
    src_crs=src_crs,
    dst_transform=grid_transform,
    dst_crs=src_crs,
    src_nodata=src_nodata,
    dst_nodata=src_nodata,
    resampling=Resampling.mode,
)
landuse_on_grid_meta = {
    "driver": "GTiff",
    "dtype": lu_arr.dtype,
    "count": 1,
    "height": grid_shape[0],
    "width": grid_shape[1],
    "crs": src_crs,
    "transform": grid_transform,
    "nodata": src_nodata,
    "compress": "deflate",
}
Path(snakemake.output.landuse_on_grid).parent.mkdir(parents=True, exist_ok=True)
with rasterio.open(
    snakemake.output.landuse_on_grid, "w", **landuse_on_grid_meta
) as dst:
    dst.write(lu_on_grid, 1)
log.info(
    f"landuse resampled onto the SFINCS grid (mode, single pass): "
    f"{grid_shape[0]}x{grid_shape[1]} cells. Written: {snakemake.output.landuse_on_grid}"
)

unmapped, _info = write_roughness_raster(
    snakemake.input.landuse_source,
    snakemake.input.matching_lu_roughness,
    {
        "height": grid_shape[0],
        "width": grid_shape[1],
        "transform": grid_transform,
        "crs": src_crs,
    },
    snakemake.output.roughness_on_grid,
    method=method,
    # The model's own "manning" field: stays on the model grid.
    refine_to_source=False,
    channel=(channel_polygons(rivers.to_crs(src_crs)), src_crs, channel_n),
)
if unmapped:
    log.warning(
        f"Land-use codes without roughness mapping (NaN, excluded from the average): "
        f"{sorted(unmapped)}"
    )
log.info(
    f"Roughness ({method}) area-averaged from the land-use source onto the model grid: "
    f"{snakemake.output.roughness_on_grid}"
)

# Land mask (landuse != 200), cell-aligned with the model: the background of
# every figure of model output.
land_mask = vectorize_land_mask_on_grid(snakemake.output.landuse_on_grid)
land_mask.to_file(snakemake.output.land_mask_on_grid, driver="GPKG")
log.info(
    f"Land mask on the SFINCS grid: {len(land_mask)} polygon(s). Written: {snakemake.output.land_mask_on_grid}"
)
profiler.stop()
log.info("Done")
