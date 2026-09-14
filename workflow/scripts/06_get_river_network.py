"""
06_get_river_network.py -- clip the global SWORD network to the domain.

Clip only: the network's own diagnostic map is rule plot_river_network
(06b). The split exists because this output is an INPUT of rule
prepare_landuse (02b) -- which derives open sea by flood-filling
WorldCover's water class from the domain border with the river network as a
barrier -- while the plot needs rule get_land_polygons' (03) own mask for
its background, and 03 in turn reads 02b's output. Keeping the clip free of
that dependency is what breaks the cycle.
"""

from pathlib import Path

import geopandas as gpd
from shapely.geometry import box as shapely_box

from src.domain import load_domain
from src.log import setup_logging
from src.profiling import ScriptProfiler

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
gpd_read_file = profiler.wrap(gpd.read_file)
gpd_clip      = profiler.wrap(gpd.clip)

wgs84_bounds, domain_crs, _domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}")

# ── 1. Load and clip river network ────────────────────────────────────────────

clip_gdf = gpd.GeoDataFrame(geometry=[shapely_box(*wgs84_bounds)], crs="EPSG:4326")

probe_crs = gpd.read_file(snakemake.input.global_river_network, rows=0).crs
if probe_crs is not None and probe_crs != clip_gdf.crs:
    bbox = clip_gdf.to_crs(probe_crs).total_bounds
else:
    bbox = wgs84_bounds

river_gdf = gpd_read_file(
    snakemake.input.global_river_network, bbox=tuple(bbox), engine="pyogrio"
)
clip_src = clip_gdf if river_gdf.crs is None or river_gdf.crs == clip_gdf.crs else clip_gdf.to_crs(river_gdf.crs)
clipped_rivers = gpd_clip(river_gdf, clip_src).copy()
log.info(f"Clipped to {len(clipped_rivers)} reach(es)")


# ── 2. Write output ───────────────────────────────────────────────────────────

Path(snakemake.output.spec_river_network).parent.mkdir(parents=True, exist_ok=True)
clipped_rivers.to_file(snakemake.output.spec_river_network, driver="GPKG")
log.info(f"Written: {snakemake.output.spec_river_network} ({len(clipped_rivers)} reaches)")
profiler.stop()
log.info("Done")
