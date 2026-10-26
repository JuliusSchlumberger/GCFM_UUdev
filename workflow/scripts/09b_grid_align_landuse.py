"""Resample land use and roughness onto the shared SFINCS grid; run by rule grid_align_landuse.

Grid from rule build_sfincs_grid (08c), in SfincsModel orientation (row 0 on the
origin side, row index = SFINCS n), so every model assigns landuse_on_grid.tif /
roughness_on_grid.tif to sf.grid.data cell for cell. landuse_on_grid.tif is the
single sea/land/roughness classification downstream: weir tracing and zsini sea
cells (rule 10), rule 13's roughness and weir diagnostics.

Landuse is a categorical field (Copernicus LC100 codes) -- resampled by mode
(the most frequent source class per cell), never averaged (averaging two
different land-use codes produces a meaningless third code). Manning's n is
area-averaged onto the grid from the native land-use source via the
05c_get_roughness.py lookup table, so a mixed cell gets a mixed roughness
rather than that of its dominant class.

Elevation stays at native resolution (the subgrid needs finer-than-cell
detail); only the landuse/roughness/sea classification is grid-aligned.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from src.domain import load_domain
from src.grid import load_grid_def
from src.landuse import write_roughness_raster
from src.log import setup_logging
from src.plots import plot_landuse, plot_roughness
from src.raster import vectorize_land_mask_on_grid
from src.river_burn import channel_polygons

log = setup_logging(snakemake.log[0])

wgs84_bounds, domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)

resolution_m = float(snakemake.params.resolution_m)

grid_def = load_grid_def(snakemake.input.sfincs_grid)
grid_transform = grid_def["transform"]
grid_shape = grid_def["shape"]
log.info(
    f"Shared SFINCS grid: {grid_shape[0]}x{grid_shape[1]} cells @ {resolution_m} m, "
    f"rotation {grid_def['rotation']:.3f} deg"
)

with rasterio.open(snakemake.input.landuse) as src:
    lu_arr = src.read(1)
    src_transform, src_crs, src_nodata = src.transform, src.crs, src.nodata

lu_on_grid = np.full(grid_shape, src_nodata, dtype=lu_arr.dtype)
# Mode: landuse.tif (~30 m) is finer than this grid (~70 m), so nearest would
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
    "driver": "GTiff", "dtype": lu_arr.dtype, "count": 1,
    "height": grid_shape[0], "width": grid_shape[1],
    "crs": src_crs, "transform": grid_transform, "nodata": src_nodata,
    "compress": "deflate",
}
Path(snakemake.output.landuse_on_grid).parent.mkdir(parents=True, exist_ok=True)
with rasterio.open(snakemake.output.landuse_on_grid, "w", **landuse_on_grid_meta) as dst:
    dst.write(lu_on_grid, 1)
log.info(
    f"landuse resampled onto the SFINCS grid (mode, single pass): "
    f"{grid_shape[0]}x{grid_shape[1]} cells. Written: {snakemake.output.landuse_on_grid}"
)

# Manning's n area-averaged from the land-use source, not reclassified from
# landuse_on_grid: the dominant class would give a 60% water / 40% built-up
# cell water's 0.02.
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
    # The model's own "manning" field: stays on the model grid, unlike rule
    # 05c's, which is subdivided for the subgrid.
    refine_to_source=False,
    channel=(
        channel_polygons(gpd.read_file(snakemake.input.river_network).to_crs(src_crs)),
        src_crs,
        float(snakemake.params.channel_manning_n),
    ),
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

# Land mask (landuse != 200), cell-aligned with the model: the background of
# every figure of model output.
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
