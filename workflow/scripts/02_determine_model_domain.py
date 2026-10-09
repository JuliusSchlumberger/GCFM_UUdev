import json
import geopandas as gpd

from src.geometry import pick_utm_crs, buffered_bbox
from src.grid import fit_grid_frame, frame_footprint
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

# Rectangle the SFINCS grid lives in (src.grid): the minimum rotated
# rectangle around the polygon when sfincs.grid.rotated, else left to rule
# 08c (axis-aligned, snapped to the grid resolution).
grid_frame = fit_grid_frame(domain_geom, rotated=bool(snakemake.params.rotated))

# Clipping bbox = small buffer around the delta polygon for input data
# clipping. A rotated grid's corners stick out of the polygon's own bbox, so
# the bbox is then taken around the rotated rectangle instead -- every
# preprocessing raster covers the whole model grid.
if grid_frame["rotated"]:
    clip_geom = gpd.GeoDataFrame(geometry=[frame_footprint(grid_frame)], crs=target_crs)
    log.info(
        f"Rotated grid frame: origin ({grid_frame['x0']:.0f}, {grid_frame['y0']:.0f}), "
        f"{grid_frame['length_x_m']} x {grid_frame['length_y_m']} m, "
        f"rotation {grid_frame['rotation']:.3f} deg"
    )
else:
    clip_geom = delta[["geometry"]]
_, bbox_bounds = buffered_bbox(
    clip_geom,
    buffer_m=delta_buffer_m,
    target_crs=target_crs,
    source_crs=clip_geom.crs,
)
log.info(f"Clipping bbox buffer={delta_buffer_m} m")

# ── write outputs ─────────────────────────────────────────────────────────────
domain_geom.to_file(snakemake.output.domain_gpkg, driver="GPKG")

with open(snakemake.output.spec_basins_meta, "w") as f:
    json.dump({
        "basin_id":            int(snakemake.wildcards.basin_id),
        "crs":                 str(target_crs),
        "buffer_m":            delta_buffer_m,
        "grid_frame":          grid_frame,
        "bounds": {
            "xmin": bbox_bounds[0],
            "ymin": bbox_bounds[1],
            "xmax": bbox_bounds[2],
            "ymax": bbox_bounds[3],
        },
    }, f, indent=2)

profiler.stop()
log.info("Done")
