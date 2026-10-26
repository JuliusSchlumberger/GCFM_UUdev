"""Warp land use onto the elevation grid and write the sea mask; run by rule get_landuse."""

import numpy as np
import rasterio

from src.domain import load_domain
from src.landuse import LANDUSE_SEA_CODE, warp_landuse_to_grid
from src.log import setup_logging
from src.plots import plot_landuse, plot_sea_mask
from src.profiling import ScriptProfiler
from src.raster import FLOAT_GTIFF_COMPRESSION, FLOAT_NODATA

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
warp_landuse_to_grid = profiler.wrap(warp_landuse_to_grid)

wgs84_bounds, domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}")

# elevation_merged.tif's UTM grid, so landuse/roughness share one pixel grid
# with elevation/zsini.
with rasterio.open(snakemake.input.elevation_merged) as ref:
    ref_meta = ref.meta.copy()
    elevation_arr = ref.read(1).astype(np.float32)
    elevation_nodata = ref.nodata

# Mode, not nearest: ESA WorldCover (10 m) is finer than this grid (~30 m), so
# nearest would keep one arbitrary source pixel per cell. With 100 m LC100 this
# is upsampling, where mode and nearest agree. Manning's n is area-averaged
# instead (rule get_roughness). Streamed in row blocks: a 10 m source window
# reaches 2.0 Gpx on the largest delta.
histogram = warp_landuse_to_grid(
    snakemake.input.landuse_source, ref_meta, snakemake.output.spec_landuse
)
_total = sum(histogram.values())
log.info(
    "landuse.tif class histogram (% of grid): "
    + ", ".join(f"{c}={100 * n / _total:.2f}" for c, n in sorted(histogram.items()))
)
log.info(f"Written: {snakemake.output.spec_landuse}")

with rasterio.open(snakemake.output.spec_landuse) as lu_src:
    data = lu_src.read(1)

# Sea mask: landuse==200 only -> 1.0 (sea), else nodata; cells without
# elevation data are excluded. Classification only, no water level: zsini
# (sea cells = calm_sea_level_m) is built in rule build_sfincs (13). Landuse-only so
# it matches src.protection_weir's ocean_mask (landuse==200); landuse 80
# (inland water) at tidal flats/lagoons is not sea.
lu_arr = data[0] if data.ndim == 3 else data
sea_mask = lu_arr == LANDUSE_SEA_CODE
sea_mask_arr = np.where(sea_mask, np.float32(1.0), FLOAT_NODATA).astype(np.float32)

if elevation_nodata is not None:
    sea_mask_arr[elevation_arr == np.float32(elevation_nodata)] = FLOAT_NODATA
sea_mask_arr[~np.isfinite(elevation_arr)] = FLOAT_NODATA

sea_mask_meta = ref_meta.copy()
sea_mask_meta.update(nodata=FLOAT_NODATA, **FLOAT_GTIFF_COMPRESSION)

n_water = int((sea_mask_arr == 1.0).sum())
log.info(f"sea_mask: {n_water:,} sea px (landuse==200 only, no OSM land-polygon input)")

with rasterio.open(snakemake.output.sea_mask, "w", **sea_mask_meta) as zdst:
    zdst.write(sea_mask_arr, 1)
log.info(f"Written: {snakemake.output.sea_mask}")

# Plots; background is rule 03's land mask (the SFINCS grid does not exist yet).
plot_landuse(
    snakemake.output.spec_landuse, domain_poly,
    snakemake.input.land_polygons, snakemake.output.plot_landuse,
)
plot_sea_mask(
    sea_mask_path=snakemake.output.sea_mask,
    bbox_poly=domain_poly,
    land_polygons_path=snakemake.input.land_polygons,
    output_path=snakemake.output.plot_sea_mask,
)
profiler.stop()
log.info("Done")
