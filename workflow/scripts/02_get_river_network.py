"""Clip the global SWORD river network to the domain; run by rule get_river_network.

It runs before rule prepare_landuse (05), whose open-sea flood fill from the
domain border uses it as a barrier. analysis/plot_river_network.py maps it.
"""

from pathlib import Path

from src.domain import load_domain
from src.log import setup_logging
from src.profiling import ScriptProfiler
from src.sword import apply_width_corrections, read_sword_reaches, read_width_corrections

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
read_sword_reaches = profiler.wrap(read_sword_reaches)

wgs84_bounds, domain_crs, _domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}")

# Load, clip, manual widths

clipped_rivers = apply_width_corrections(
    read_sword_reaches(list(snakemake.input.sword_regions), wgs84_bounds),
    read_width_corrections(snakemake.input.width_corrections),
)
log.info(f"Clipped to {len(clipped_rivers)} reach(es)")


# Write output

Path(snakemake.output.spec_river_network).parent.mkdir(parents=True, exist_ok=True)
clipped_rivers.to_file(snakemake.output.spec_river_network, driver="GPKG")
log.info(f"Written: {snakemake.output.spec_river_network} ({len(clipped_rivers)} reaches)")
profiler.stop()
log.info("Done")
