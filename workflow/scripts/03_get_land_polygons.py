"""Vectorize land (landuse != 200) as a mask polygon; run by rule get_land_polygons.

WGS84 at the source's native resolution. Figure backgrounds only, for figures
made before the SFINCS grid exists (rules 04-09, 11b); nothing computational
reads it. Same landuse==200 sea criterion as sea_mask.tif (rule 05b) and
src.protection_weir's ocean_mask.
"""

from pathlib import Path

from src.domain import load_domain
from src.log import setup_logging
from src.profiling import ScriptProfiler
from src.raster import vectorize_land_from_landuse

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
vectorize_land_from_landuse = profiler.wrap(vectorize_land_from_landuse)

# Domain bounds
wgs84_bounds, _, _ = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}")

# Vectorize landuse != 200
land = vectorize_land_from_landuse(snakemake.input.landuse_source, wgs84_bounds)
log.info(f"Landuse land polygons: {len(land)} polygon(s) (landuse != 200)")

# Write
out_path = Path(snakemake.output.land_polygons)
out_path.parent.mkdir(parents=True, exist_ok=True)
land.to_file(out_path, driver="GPKG")
profiler.stop()
log.info(f"Written: {out_path}")
