"""
05c_get_roughness.py -- Manning's n on elevation_merged.tif's grid.

Area-averaged from the land-use SOURCE at its own resolution (rule
prepare_landuse, 02b), never reclassified from the already-upscaled
landuse.tif: n is a continuous quantity, and taking the dominant class'
value hands a cell that is 60% water and 40% built-up water's 0.02.

Written on the elevation grid SUBDIVIDED to about the source's own pixel
size (refine_to_source): ~10 m under ESA WorldCover, the unchanged ~31 m
elevation grid under LC100, whose 100 m pixels carry no finer detail to
keep. hydromt samples this raster per SUBGRID pixel (7 m at
nr_subgrid_pixels=10) and averages it by conveyance itself, so at 10 m
those 100 samples per model cell see ~49 distinct values instead of ~5.
Staying an exact subdivision of the elevation grid keeps CRS and alignment
with elevation/subgrid. See src.landuse.write_roughness_raster.
"""

import rasterio

from src.domain import load_domain
from src.landuse import write_roughness_raster
from src.log import setup_logging
from src.plots import plot_roughness
from src.profiling import ScriptProfiler

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
write_roughness_raster = profiler.wrap(write_roughness_raster)

wgs84_bounds, domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}")

with rasterio.open(snakemake.input.elevation_merged) as ref:
    ref_meta = ref.meta.copy()

method = str(snakemake.params.roughness_aggregation)
unmapped, info = write_roughness_raster(
    snakemake.input.landuse_source,
    snakemake.input.matching_lu_roughness,
    ref_meta,
    snakemake.output.spec_roughness,
    method=method,
    refine_to_source=True,
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

plot_roughness(
    snakemake.output.spec_roughness, domain_poly,
    snakemake.input.land_polygons, snakemake.output.plot_roughness,
)
profiler.stop()
log.info("Done")
