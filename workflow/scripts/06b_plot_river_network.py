"""
06b_plot_river_network.py -- the clipped river network's diagnostic map.

Split out of 06_get_river_network.py so that the clip itself can run BEFORE
rule prepare_landuse (02b), which uses the network as a barrier when
deriving open sea. This plot needs rule get_land_polygons' (03) land mask
for its background, and 03 reads 02b's output -- so only the plot, not the
clip, may sit downstream of 03.
"""

from src.domain import load_domain
from src.log import setup_logging
from src.plots import plot_river_network
from src.profiling import ScriptProfiler

log = setup_logging(snakemake.log[0])
profiler = ScriptProfiler(snakemake)

_, _, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
plot_river_network(
    snakemake.input.spec_river_network, domain_poly,
    snakemake.input.land_polygons, snakemake.output.plot_river_network,
)
profiler.stop()
log.info(f"Written: {snakemake.output.plot_river_network}")
