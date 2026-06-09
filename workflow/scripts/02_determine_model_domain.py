import json
import geopandas as gpd

from src.geometry import (select_intersecting, merge_geometries, pick_utm_crs, buffered_bbox)
from src.log import setup_logging
from src.profiling import ScriptProfiler

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
select_intersecting = profiler.wrap(select_intersecting)
merge_geometries    = profiler.wrap(merge_geometries)
buffered_bbox       = profiler.wrap(buffered_bbox)

mode           = snakemake.params.mode
buffer_m       = snakemake.params.buffer_m
delta_buffer_m = snakemake.params.delta_buffer_m
target_crs     = snakemake.params.target_crs

delta = gpd.read_file(snakemake.input.specific_delta)
log.info(f"Delta polygon: {len(delta)} feature(s), CRS={delta.crs}")

# ── resolve UTM CRS ───────────────────────────────────────────────────────────
if target_crs == "auto_utm":
    target_crs = pick_utm_crs(delta)
log.info(f"Target CRS: {target_crs}  |  mode: {mode}")

# ── mode: basins ──────────────────────────────────────────────────────────────
if mode == "basins":
    basins_all = gpd.read_file(snakemake.input.river_basins)
    log.info(f"Hydro basins: {len(basins_all)} feature(s), CRS={basins_all.crs}")

    intersecting = select_intersecting(basins_all, delta)
    log.info(f"Intersecting basins: {len(intersecting)}")
    if intersecting.empty:
        raise ValueError(
            f"No basins intersect delta polygon for basin_id={snakemake.wildcards.basin_id}"
        )

    # Domain polygon = buffered bbox of the merged intersecting basins
    merged = merge_geometries(intersecting)
    domain_geom, bbox_bounds = buffered_bbox(
        merged,
        buffer_m=buffer_m,
        target_crs=target_crs,
        source_crs=intersecting.crs,
    )
    n_basins = len(intersecting)

# ── mode: delta_polygon ───────────────────────────────────────────────────────
elif mode == "delta_polygon":
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
    n_basins = 0
    log.info(f"Using delta polygon directly; clipping bbox buffer={delta_buffer_m} m")

else:
    raise ValueError(f"Unknown domain mode '{mode}'. Expected 'basins' or 'delta_polygon'.")

# ── write outputs ─────────────────────────────────────────────────────────────
intersecting.to_file(snakemake.output.specific_basins, driver="GPKG")
domain_geom.to_file(snakemake.output.domain_gpkg, driver="GPKG")

with open(snakemake.output.spec_basins_meta, "w") as f:
    json.dump({
        "basin_id":            int(snakemake.wildcards.basin_id),
        "mode":                mode,
        "crs":                 str(target_crs),
        "buffer_m":            buffer_m if mode == "basins" else delta_buffer_m,
        "bounds": {
            "xmin": bbox_bounds[0],
            "ymin": bbox_bounds[1],
            "xmax": bbox_bounds[2],
            "ymax": bbox_bounds[3],
        },
        "n_intersecting_basins": n_basins,
    }, f, indent=2)

profiler.stop()
log.info("Done")
