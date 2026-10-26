"""Write land use with the river channel set to water; run by rule channel_landuse."""

import geopandas as gpd
import rasterio
from src.landuse import write_channel_landuse
from src.log import setup_logging
from src.river_burn import channel_polygons

log = setup_logging(snakemake.log[0])

with rasterio.open(snakemake.input.landuse) as _src:
    crs = _src.crs
rivers = gpd.read_file(snakemake.input.river_network).to_crs(crs)
converted = write_channel_landuse(
    snakemake.input.landuse,
    channel_polygons(rivers),
    crs,
    int(snakemake.params.water_code),
    snakemake.output.landuse_channels,
)
log.info(
    f"Written: {snakemake.output.landuse_channels} ({sum(converted.values()):,} channel pixel(s) "
    f"set to {snakemake.params.water_code}; from class: {converted})"
)
