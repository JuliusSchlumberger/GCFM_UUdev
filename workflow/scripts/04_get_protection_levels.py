import json
from pathlib import Path

import geopandas as gpd

from src.log import setup_logging
from src.plots import plot_protection_levels
from src.profiling import ScriptProfiler
from src.protection_levels import (
    identify_dominant_protection,
    load_flopros_table,
    load_geogunit_iso_lookup,
)

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
identify_dominant_protection = profiler.wrap(identify_dominant_protection)

delta = gpd.read_file(snakemake.input.specific_delta)
delta_polygon = delta.to_crs("EPSG:4326").geometry.union_all()
log.info(f"Delta polygon bounds (WGS84): {delta_polygon.bounds}")

flopros_df = load_flopros_table(snakemake.input.flopros_table)
iso_lookup = load_geogunit_iso_lookup(snakemake.input.geogunit_list)
log.info(f"Loaded FLOPROS table: {len(flopros_df)} geounit(s)")

summary = identify_dominant_protection(
    delta_polygon,
    snakemake.input.geogunit_raster,
    flopros_df,
    iso_lookup,
    default_rp_yr=snakemake.params.default_rp_yr,
    max_rp_yr=snakemake.params.max_rp_yr,
)

Path(snakemake.output.protection_levels).parent.mkdir(parents=True, exist_ok=True)
with open(snakemake.output.protection_levels, "w") as f:
    json.dump(summary, f, indent=2)
log.info(f"Written: {snakemake.output.protection_levels}")

plot_protection_levels(
    delta_polygon,
    snakemake.input.geogunit_raster,
    flopros_df,
    summary,
    snakemake.input.osm_land,
    snakemake.output.plot_protection,
)
log.info(f"Plot written: {snakemake.output.plot_protection}")

profiler.stop()
log.info("Done")
