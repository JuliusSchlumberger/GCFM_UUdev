import json
import tempfile
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
from src.raster import vectorize_land_from_landuse

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

# plot_protection_levels' own background needs a vector land-polygon file,
# not the raw landuse raster -- vectorized here into a throwaway temp file
# (never a declared Snakemake output) rather than depending on rule
# get_land_polygons' (03) own per-basin output, which this rule deliberately
# does NOT depend on (see this script's own module docstring / the rule's
# docstring: protection-level lookup depends only on the delta polygon, not
# the model domain chain). margin_frac matches plot_protection_levels' own
# default so the vectorized window fully covers its expanded display bbox.
lon_min, lat_min, lon_max, lat_max = delta_polygon.bounds
_margin = max(lon_max - lon_min, lat_max - lat_min) * 0.5
land_bounds = (lon_min - _margin, lat_min - _margin, lon_max + _margin, lat_max + _margin)
land = vectorize_land_from_landuse(snakemake.input.global_landuse, land_bounds)
log.info(f"Protection-level plot background: {len(land)} land polygon(s) (landuse != 200)")

with tempfile.TemporaryDirectory() as _tmp_dir:
    _land_tmp_path = Path(_tmp_dir) / "land_from_landuse.gpkg"
    land.to_file(_land_tmp_path, driver="GPKG")
    plot_protection_levels(
        delta_polygon,
        snakemake.input.geogunit_raster,
        flopros_df,
        summary,
        str(_land_tmp_path),
        snakemake.output.plot_protection,
    )
log.info(f"Plot written: {snakemake.output.plot_protection}")

profiler.stop()
log.info("Done")
