from pathlib import Path

import geopandas as gpd

from src.domain import load_domain
from src.log import setup_logging
from src.plots import (
    plot_bathymetry,
    plot_landuse,
    plot_river_network,
    plot_roughness,
    plot_topography,
)
from src.raster import (
    build_roughness_raster,
    clip_raster,
    find_diluviumdem_tiles,
    merge_tiled_raster,
)

log = setup_logging(snakemake.log[0])

# ── domain ────────────────────────────────────────────────────────────────────

wgs84_bounds, domain_crs, bbox_poly = load_domain(snakemake.input.spec_basins_meta)
lon_min, lat_min, lon_max, lat_max = wgs84_bounds
bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_poly], crs="EPSG:4326")
log.info(f"Domain WGS84 bounds: {wgs84_bounds}, CRS: {domain_crs}")

# ── topography (DiluviumDEM named tiles) ──────────────────────────────────────

tiles = find_diluviumdem_tiles(snakemake.input.global_topography_tiles, wgs84_bounds)
if not tiles:
    raise FileNotFoundError("No DiluviumDEM tiles found for domain")
log.info(f"Found {len(tiles)} topography tile(s)")

Path(snakemake.output.spec_topography).parent.mkdir(parents=True, exist_ok=True)
merge_tiled_raster(tiles, wgs84_bounds, snakemake.output.spec_topography)
log.info(f"Written: {snakemake.output.spec_topography}")

# ── bathymetry ────────────────────────────────────────────────────────────────

clip_raster(snakemake.input.global_bathymetry, wgs84_bounds, snakemake.output.spec_bathymetry)
log.info(f"Written: {snakemake.output.spec_bathymetry}")

# ── land use ──────────────────────────────────────────────────────────────────

clip_raster(snakemake.input.global_landuse, wgs84_bounds, snakemake.output.spec_landuse)
log.info(f"Written: {snakemake.output.spec_landuse}")

# ── roughness (reclassify land use via lookup table) ──────────────────────────

unmapped = build_roughness_raster(
    snakemake.output.spec_landuse,
    snakemake.input.matching_lu_roughness,
    snakemake.output.spec_roughness,
)
if unmapped:
    log.warning(f"Land-use codes without roughness mapping: {unmapped}")
log.info(f"Written: {snakemake.output.spec_roughness}")

# ── river network ─────────────────────────────────────────────────────────────

river_gdf = gpd.read_file(snakemake.input.global_river_network)
clip_bbox = bbox_gdf if river_gdf.crs == bbox_gdf.crs else bbox_gdf.to_crs(river_gdf.crs)
clipped_rivers = gpd.clip(river_gdf, clip_bbox)
Path(snakemake.output.spec_river_network).parent.mkdir(parents=True, exist_ok=True)
clipped_rivers.to_file(snakemake.output.spec_river_network, driver="GPKG")
log.info(f"Written: {snakemake.output.spec_river_network} ({len(clipped_rivers)} reaches)")

# ── summary plots ─────────────────────────────────────────────────────────────

log.info("--- Summary plots ---")

plot_topography(
    snakemake.output.spec_topography, bbox_poly,
    snakemake.input.land_polygons, snakemake.output.plot_topography,
)
plot_bathymetry(
    snakemake.output.spec_bathymetry, bbox_poly,
    snakemake.input.land_polygons, snakemake.output.plot_bathymetry,
)
plot_landuse(
    snakemake.output.spec_landuse, bbox_poly,
    snakemake.input.land_polygons, snakemake.output.plot_landuse,
)
plot_roughness(
    snakemake.output.spec_roughness, bbox_poly,
    snakemake.input.land_polygons, snakemake.output.plot_roughness,
)
plot_river_network(
    snakemake.output.spec_river_network, bbox_poly,
    snakemake.input.land_polygons, snakemake.output.plot_river_network,
)

log.info("Done")
