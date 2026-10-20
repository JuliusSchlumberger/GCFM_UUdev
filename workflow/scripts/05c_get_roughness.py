"""Write Manning's n on the elevation grid; run by rule get_roughness.

Area-averaged from the land-use source at its own resolution, not reclassified
from the upscaled landuse.tif: n is continuous, and the dominant class would give
a 60% water / 40% built-up cell water's 0.02.

The grid is the elevation grid subdivided to about the source pixel size: ~10 m
under ESA WorldCover, the unchanged ~31 m grid under LC100 (100 m pixels). hydromt
samples it per subgrid pixel (7 m at nr_subgrid_pixels=10) and averages by
conveyance, so at 10 m the 100 samples per cell see ~49 distinct values, not ~5.
An exact subdivision keeps CRS and alignment with elevation/subgrid.
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
