from pathlib import Path

from src.domain import load_domain
from src.log import setup_logging
from src.plots import plot_roughness
from src.profiling import ScriptProfiler
from src.raster import build_roughness_raster

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
build_roughness_raster = profiler.wrap(build_roughness_raster)

wgs84_bounds, domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}")

Path(snakemake.output.spec_roughness).parent.mkdir(parents=True, exist_ok=True)
unmapped = build_roughness_raster(
    snakemake.input.spec_landuse,
    snakemake.input.matching_lu_roughness,
    snakemake.output.spec_roughness,
)
if unmapped:
    log.warning(f"Land-use codes without roughness mapping: {unmapped}")
log.info(f"Written: {snakemake.output.spec_roughness}")

plot_roughness(
    snakemake.output.spec_roughness, domain_poly,
    snakemake.input.land_polygons, snakemake.output.plot_roughness,
    water_bodies_path=snakemake.input.spec_landuse,
)
profiler.stop()
log.info("Done")
