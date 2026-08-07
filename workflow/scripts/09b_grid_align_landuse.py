"""
09b_grid_align_landuse.py -- resample landuse.tif (rule get_landuse, 05b,
native FathomDEM resolution) onto the shared SFINCS regular grid EXACTLY
ONCE, then derive roughness directly from that single coarse raster.

Builds its own grid via hydromt.model.processes.create_grid_from_region
(region=domain_gpkg, res=grid_resolution.json) -- the SAME deterministic
call rule build_sfincs_grid (08c) uses for sfincs_grid.json -- rather than
reading sfincs_grid.json's own transform directly. This matters: rule 08c
deliberately calls the standalone function directly (not via a SfincsModel)
to avoid leaving behind an empty scratch model directory, but that means
sfincs_grid.json is stored in the function's own RAW orientation (y
DESCENDING, row 0 = north, standard GDAL convention). Every actual
SfincsModel built in this pipeline (rule modelled_depth_estimation's
calibration model, rule build_sfincs_skeleton, rule build_sfincs) goes
through SfincsModel.grid.create_from_region(), which explicitly flips to y
ASCENDING (row 0 = south) right after calling that same function --
required by SFINCS's own m/n indexing convention (row index increases
northward). Reading sfincs_grid.json's raw (unflipped) transform and then
directly assigning an actual SfincsModel's own sf.grid.data["dep"] coords
onto it -- exactly what every downstream consumer of this rule's own
landuse_on_grid.tif/roughness_on_grid.tif does, by design, for zero-
reprojection reuse -- silently mismatches row order otherwise: same pixel
grid, same shape, but row 0 means opposite geographic edges. 2026-08-07e:
replicate the SAME create_grid_from_region + flipud check
SfincsModel.grid.create_from_region() does internally (see hydromt_sfincs's
own regulargrid.py), so this rule's own outputs land in the identical,
SfincsModel-native (y-ascending) orientation every consumer already
assumes -- without needing to instantiate an actual SfincsModel (which
would leave behind its own empty scratch directory, the exact thing rule
08c's own design avoids).

landuse_on_grid.tif becomes the ONLY sea/land/roughness classification any
downstream consumer resamples from ever again -- coastal protection weir
tracing (rule modelled_depth_estimation/empirical_depth_estimation, 10),
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

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from hydromt.model.processes.grid import create_grid_from_region
from rasterio.warp import Resampling, reproject

from src.domain import load_domain
from src.log import setup_logging
from src.plots import plot_landuse, plot_roughness
from src.raster import build_roughness_raster

log = setup_logging(snakemake.log[0])

wgs84_bounds, domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)

with open(snakemake.input.grid_resolution) as f:
    resolution = float(json.load(f)["resolution"])
delta_domain = gpd.read_file(snakemake.input.domain_gpkg)

_grid_ds = create_grid_from_region(
    region={"geom": delta_domain}, res=resolution, crs="utm", region_crs=4326,
    rotated=False, add_mask=False, align=True,
)
# SfincsModel.grid.create_from_region()'s own flip check (regulargrid.py) --
# SFINCS's own m/n indexing needs y ascending (row 0 = south), opposite of
# create_grid_from_region's own raw (GDAL-standard, row 0 = north) output.
if _grid_ds.raster.res[1] < 0:
    _grid_ds = _grid_ds.raster.flipud()
grid_transform = _grid_ds.raster.transform
grid_shape = _grid_ds.raster.shape
log.info(
    f"Grid built via create_grid_from_region + SfincsModel-native y-ascending flip: "
    f"{grid_shape[0]}x{grid_shape[1]} cells @ {resolution} m"
)

with rasterio.open(snakemake.input.landuse) as src:
    lu_arr = src.read(1)
    src_transform, src_crs, src_nodata = src.transform, src.crs, src.nodata

lu_on_grid = np.full(grid_shape, src_nodata, dtype=lu_arr.dtype)
reproject(
    source=lu_arr,
    destination=lu_on_grid,
    src_transform=src_transform,
    src_crs=src_crs,
    dst_transform=grid_transform,
    dst_crs=src_crs,
    src_nodata=src_nodata,
    dst_nodata=src_nodata,
    resampling=Resampling.nearest,
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

unmapped = build_roughness_raster(
    snakemake.output.landuse_on_grid,
    snakemake.input.matching_lu_roughness,
    snakemake.output.roughness_on_grid,
)
if unmapped:
    log.warning(f"Land-use codes without roughness mapping: {unmapped}")
log.info(f"Roughness built directly from landuse_on_grid.tif: {snakemake.output.roughness_on_grid}")

plot_landuse(
    snakemake.output.landuse_on_grid, domain_poly,
    snakemake.input.land_polygons, snakemake.output.plot_landuse_on_grid,
    water_bodies_path=snakemake.output.landuse_on_grid,
)
plot_roughness(
    snakemake.output.roughness_on_grid, domain_poly,
    snakemake.input.land_polygons, snakemake.output.plot_roughness_on_grid,
    water_bodies_path=snakemake.output.landuse_on_grid,
)
log.info("Done")
