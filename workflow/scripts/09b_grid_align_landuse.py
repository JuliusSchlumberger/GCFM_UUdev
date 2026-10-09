"""
09b_grid_align_landuse.py -- resample landuse.tif (rule get_landuse, 05b,
native FathomDEM resolution) onto the shared SFINCS regular grid EXACTLY
ONCE, then derive roughness directly from that single coarse raster.

The grid is the shared grid definition written by rule build_sfincs_grid
(08c, {basin_id}_sfincs_grid.json -- see src/grid.py): axis-aligned or
rotated, stored in SfincsModel's own orientation (row 0 on the origin side,
row index = SFINCS n). Every SfincsModel in this pipeline (rule
modelled_depth_estimation's calibration model, rule build_sfincs_skeleton,
rule build_sfincs) is created from that same definition, so every
downstream consumer can assign this rule's landuse_on_grid.tif /
roughness_on_grid.tif onto its own sf.grid.data cell for cell, with zero
reprojection.

landuse_on_grid.tif becomes the ONLY sea/land/roughness classification any
downstream consumer resamples from ever again -- coastal protection weir
tracing (rule modelled_depth_estimation, 10),
zsini's sea-cell classification (same rules), 13_build_sfincs_skeleton.py's
own roughness section and weir diagnostics. Previously each of those
consumers independently reprojected native landuse.tif/roughness.tif a
second time (src.protection_weir.GridArrays.sample_landuse, HydroMT's own
roughness.create() reproject) -- several independent nearest-neighbor
passes over the same source raster, at different points in the pipeline,
each capable of disagreeing slightly with the others at the coastline
fringe. One raster, one resampling pass, one source of truth from here on
-- matches the same principle already applied to zsini
(zsini_sea_cells_on_grid.tif, rule modelled_depth_estimation).

Landuse is a categorical field (Copernicus LC100 codes) -- resampled with
nearest-neighbor, never averaged (averaging two different land-use codes
produces a meaningless third code). Roughness is then built directly from
this coarse classification via the same lookup-table reclassification
05c_get_roughness.py uses on the native raster, rather than resampling a
continuous roughness raster with a DIFFERENT interpolation method -- so
"this cell's roughness" and "this cell's land-use code" always agree by
construction.

DEM/subgrid resolution is unaffected by any of this -- elevation stays
native resolution throughout (subgrid genuinely needs finer-than-grid-cell
detail); only the categorical landuse/roughness/sea classification chain
is coarse-grid-aligned here.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from src.domain import load_domain
from src.grid import load_grid_def
from src.landuse import write_roughness_raster
from src.log import setup_logging
from src.plots import plot_landuse, plot_roughness
from src.raster import vectorize_land_mask_on_grid

log = setup_logging(snakemake.log[0])

wgs84_bounds, domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)

resolution = float(snakemake.params.resolution)

# The shared grid definition (rule build_sfincs_grid, 08c) -- already in
# SfincsModel's own orientation (row 0 on the origin side), so these rasters
# line up cell for cell with every model's sf.grid.data.
grid_def = load_grid_def(snakemake.input.sfincs_grid)
grid_transform = grid_def["transform"]
grid_shape = grid_def["shape"]
log.info(
    f"Shared SFINCS grid: {grid_shape[0]}x{grid_shape[1]} cells @ {resolution} m, "
    f"rotation {grid_def['rotation']:.3f} deg"
)

with rasterio.open(snakemake.input.landuse) as src:
    lu_arr = src.read(1)
    src_transform, src_crs, src_nodata = src.transform, src.crs, src.nodata

lu_on_grid = np.full(grid_shape, src_nodata, dtype=lu_arr.dtype)
# Resampling.mode: landuse.tif (~30 m) is finer than this grid (~70 m), so
# nearest would keep one arbitrary source pixel per model cell. Classes are
# categorical, so mode is the only meaningful upscaling; Manning's n, being
# continuous, is area-averaged from the source instead (below).
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
    "driver": "GTiff", "dtype": lu_arr.dtype, "count": 1,
    "height": grid_shape[0], "width": grid_shape[1],
    "crs": src_crs, "transform": grid_transform, "nodata": src_nodata,
    "compress": "deflate",
}
Path(snakemake.output.landuse_on_grid).parent.mkdir(parents=True, exist_ok=True)
with rasterio.open(snakemake.output.landuse_on_grid, "w", **landuse_on_grid_meta) as dst:
    dst.write(lu_on_grid, 1)
log.info(
    f"landuse resampled onto the SFINCS grid (nearest-neighbor, single pass): "
    f"{grid_shape[0]}x{grid_shape[1]} cells. Written: {snakemake.output.landuse_on_grid}"
)

# Manning's n area-averaged from the land-use SOURCE at its own resolution
# onto this grid -- NOT reclassified from landuse_on_grid, whose dominant
# class would hand a cell that is 60% water and 40% built-up water's own
# 0.02 (see src.landuse.aggregate_manning / rule get_roughness).
_method = str(snakemake.params.roughness_aggregation)
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
    method=_method,
    # This raster IS the model's own "manning" field, so it stays exactly on
    # the model grid -- unlike rule 05c's, which is subdivided for the subgrid.
    refine_to_source=False,
)
if unmapped:
    log.warning(
        f"Land-use codes without roughness mapping (NaN, excluded from the average): "
        f"{sorted(unmapped)}"
    )
log.info(
    f"Roughness ({_method}) area-averaged from the land-use source onto the model grid: "
    f"{snakemake.output.roughness_on_grid}"
)

# Land mask traced from landuse_on_grid (land use != 200), cell-aligned with
# the model -- THE land background of every figure of model output from here
# on (see src.plots' module docstring).
land_mask = vectorize_land_mask_on_grid(snakemake.output.landuse_on_grid)
land_mask.to_file(snakemake.output.land_mask_on_grid, driver="GPKG")
log.info(f"Land mask on the SFINCS grid: {len(land_mask)} polygon(s). Written: {snakemake.output.land_mask_on_grid}")

plot_landuse(
    snakemake.output.landuse_on_grid, domain_poly,
    snakemake.output.land_mask_on_grid, snakemake.output.plot_landuse_on_grid,
)
plot_roughness(
    snakemake.output.roughness_on_grid, domain_poly,
    snakemake.output.land_mask_on_grid, snakemake.output.plot_roughness_on_grid,
)
log.info("Done")
