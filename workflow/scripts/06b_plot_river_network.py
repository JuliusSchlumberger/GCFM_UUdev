"""Plot the clipped river network; run by rule plot_river_network.

Separate from rule 06 because the background needs rule 03's land mask, which
depends on rule 02b, which reads the clipped network.
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
