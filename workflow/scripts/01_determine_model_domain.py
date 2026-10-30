"""Extract the basin's delta polygon and pick the UTM CRS and clipping bbox.

Run by rule determine_model_domain.
"""

import json

import geopandas as gpd

from src.geometry import buffered_bbox, pick_utm_crs
from src.io import catalogue_entry, load_catalogue, read_geometry
from src.log import setup_logging
from src.profiling import ScriptProfiler

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
read_geometry = profiler.wrap(read_geometry)
buffered_bbox = profiler.wrap(buffered_bbox)

# Delta polygon
CATALOGUE = load_catalogue(snakemake.config["data_catalogue"])
basin_id = int(snakemake.wildcards.basin_id)
log.info(f"Extracting basin {basin_id}")

deltas = read_geometry(snakemake.input.delta_polygons)
attribute_name = catalogue_entry(CATALOGUE, "delta_polygons")["attributes"][0]["name"]
delta = deltas[deltas[attribute_name] == basin_id]
if delta.empty:
    raise ValueError(f"BasinID2 {basin_id} not found in {snakemake.input.delta_polygons}")
delta.to_file(snakemake.output.specific_delta, driver="GPKG")
log.info(f"Wrote {snakemake.output.specific_delta} ({len(delta)} features)")

# Model domain
delta_buffer_m = snakemake.params.delta_buffer_m
delta = gpd.read_file(snakemake.output.specific_delta)
log.info(f"Delta polygon: {len(delta)} feature(s), CRS={delta.crs}")

target_crs = pick_utm_crs(delta)
log.info(f"Target CRS: {target_crs}")

domain_geom = delta.to_crs(target_crs)[["geometry"]]

clip_geom = delta[["geometry"]]
_, bbox_bounds = buffered_bbox(
    clip_geom,
    buffer_m=delta_buffer_m,
    target_crs=target_crs,
    source_crs=clip_geom.crs,
)
log.info(f"Clipping bbox buffer={delta_buffer_m} m")

domain_geom.to_file(snakemake.output.domain_gpkg, driver="GPKG")

with open(snakemake.output.spec_basins_meta, "w") as f:
    json.dump({
        "basin_id":            basin_id,
        "crs":                 str(target_crs),
        "buffer_m":            delta_buffer_m,
        "bounds": {
            "xmin": bbox_bounds[0],
            "ymin": bbox_bounds[1],
            "xmax": bbox_bounds[2],
            "ymax": bbox_bounds[3],
        },
    }, f, indent=2)

profiler.stop()
log.info("Done")
