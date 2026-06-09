from pathlib import Path

from src.domain import load_domain
from src.log import setup_logging
from src.plots import plot_landuse
from src.profiling import ScriptProfiler
from src.raster import clip_raster

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
clip_raster = profiler.wrap(clip_raster)

wgs84_bounds, domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}")

Path(snakemake.output.spec_landuse).parent.mkdir(parents=True, exist_ok=True)
clip_raster(snakemake.input.global_landuse, wgs84_bounds, snakemake.output.spec_landuse)
log.info(f"Written: {snakemake.output.spec_landuse}")

plot_landuse(
    snakemake.output.spec_landuse, domain_poly,
    snakemake.input.land_polygons, snakemake.output.plot_landuse,
)
profiler.stop()
log.info("Done")
