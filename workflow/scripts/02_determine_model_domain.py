import json
import geopandas as gpd

from src.geometry import pick_utm_crs, buffered_bbox
from src.log import setup_logging
from src.profiling import ScriptProfiler

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
buffered_bbox = profiler.wrap(buffered_bbox)

delta_buffer_m = snakemake.params.delta_buffer_m

delta = gpd.read_file(snakemake.input.specific_delta)
log.info(f"Delta polygon: {len(delta)} feature(s), CRS={delta.crs}")

# ── resolve UTM CRS ───────────────────────────────────────────────────────────
target_crs = pick_utm_crs(delta)
log.info(f"Target CRS: {target_crs}")

# domain.gpkg = the delta polygon itself (reprojected to UTM)
domain_geom = delta.to_crs(target_crs)[["geometry"]]

# intersecting_basins.gpkg = delta polygon as a stand-in
intersecting = delta.to_crs(target_crs)[["geometry"]]

# Clipping bbox = small buffer around the delta polygon for input data clipping
_, bbox_bounds = buffered_bbox(
    delta[["geometry"]],
    buffer_m=delta_buffer_m,
    target_crs=target_crs,
    source_crs=delta.crs,
)
log.info(f"Clipping bbox buffer={delta_buffer_m} m")

# ── write outputs ─────────────────────────────────────────────────────────────
intersecting.to_file(snakemake.output.specific_basins, driver="GPKG")
domain_geom.to_file(snakemake.output.domain_gpkg, driver="GPKG")

with open(snakemake.output.spec_basins_meta, "w") as f:
    json.dump({
        "basin_id":            int(snakemake.wildcards.basin_id),
        "crs":                 str(target_crs),
        "buffer_m":            delta_buffer_m,
        "bounds": {
            "xmin": bbox_bounds[0],
            "ymin": bbox_bounds[1],
            "xmax": bbox_bounds[2],
            "ymax": bbox_bounds[3],
        },
        "n_intersecting_basins": 0,
    }, f, indent=2)

profiler.stop()
log.info("Done")
