import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS as RasterioCRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

from src.grid import load_grid_def
from src.log import setup_logging
from src.profiling import ScriptProfiler
from src.river_conditioning import enforce_river_monotonicity

log = setup_logging(snakemake.log[0])
profiler = ScriptProfiler(snakemake)
enforce_river_monotonicity = profiler.wrap(enforce_river_monotonicity)

# ── run conditioning ──────────────────────────────────────────────────────────
# river_network_clean.gpkg (rule 08) -- topology/width only (reach_id,
# rch_id_dn, is_seed, width); enforce_river_monotonicity never reads
# rivdph, so conditioning no longer waits on either depth-estimation branch.

rivers = gpd.read_file(snakemake.input.river_network)
log.info(f"Loaded {len(rivers)} reaches from river_network_clean.gpkg")

n_modified, n_checked, centerline_max_m = enforce_river_monotonicity(
    rivers=rivers,
    elevation_path=snakemake.input.elevation_merged,
    output_path=snakemake.output.elevation_conditioned,
)
n_total_pixels = sum(n_modified.values())
pct_modified = 100.0 * n_total_pixels / n_checked if n_checked else 0.0
log.info(
    f"Conditioning complete: {n_total_pixels}/{n_checked} pixel(s) lowered "
    f"({pct_modified:.1f}%) across {len(n_modified)} reach(es)"
)
log.info(f"Max conditioned centerline elevation: {centerline_max_m:.2f} m")

# Used by rule modelled_depth_estimation (10)/13 to set an elevation ceiling on the active-cell mask
# (sfincs.grid.active_mask.elevation_buffer_m added on top there) -- see
# enforce_river_monotonicity's own docstring for why the network's own
# seed reach(es) always attain this maximum.
with open(snakemake.output.river_elevation_max, "w") as f:
    json.dump({"river_elevation_max_m": centerline_max_m}, f, indent=2)
log.info(f"Written: {snakemake.output.river_elevation_max}")

# ── resample onto the shared SFINCS grid ─────────────────────────────────────
# Single shared coarse "background" elevation layer for rule
# modelled_depth_estimation (10, modelled depth calibration) and rule 13
# (production build)'s own
# sf.elevation.create() base layer -- resampling ONCE, here, guarantees they
# use the identical coarse raster, rather than each independently letting
# HydroMT resample from elevation_conditioned.tif itself.
grid_def = load_grid_def(snakemake.input.sfincs_grid)
grid_transform = grid_def["transform"]
dst_crs = RasterioCRS.from_string(grid_def["crs"])

with rasterio.open(snakemake.output.elevation_conditioned) as src:
    src_arr = src.read(1)
    src_transform = src.transform
    src_crs = src.crs
    src_nodata = src.nodata if src.nodata is not None else -9999.0
    src_bounds = rasterio.transform.array_bounds(src.height, src.width, src_transform)

# Extend the destination raster to cover the FULL native elevation extent
# (not just the exact SFINCS grid bounds), snapped to the grid's own
# resolution/phase -- HydroMT's elevation.create() reads its own RasterDataset
# source with a margin beyond the model grid for safe reprojection/
# interpolation at the edges; cropping tightly to the grid bounds would leave
# that margin as nodata, which HydroMT has no valid data to fall back on
# (triggers a "Dataset ... does not fully cover bbox ..." warning and
# spurious flooding from the resulting gap).
# elevation_merged.tif/elevation_conditioned.tif already carry a generous
# buffer around the domain (rule 05a), far more than HydroMT's own margin
# needs, so aligning to that full extent is a safe, simple fix.
inv_grid = ~grid_transform
corners = [
    (src_bounds[0], src_bounds[1]), (src_bounds[2], src_bounds[1]),
    (src_bounds[0], src_bounds[3]), (src_bounds[2], src_bounds[3]),
]
cols, rows = zip(*(inv_grid * c for c in corners))
col_off = int(np.floor(min(cols)))
row_off = int(np.floor(min(rows)))
col_stop = int(np.ceil(max(cols)))
row_stop = int(np.ceil(max(rows)))
dst_transform = grid_transform * Affine.translation(col_off, row_off)
dst_shape = (row_stop - row_off, col_stop - col_off)

dst_arr = np.full(dst_shape, src_nodata, dtype=src_arr.dtype)
reproject(
    source=src_arr,
    destination=dst_arr,
    src_transform=src_transform,
    src_crs=src_crs,
    src_nodata=src_nodata,
    dst_transform=dst_transform,
    dst_crs=dst_crs,
    dst_nodata=src_nodata,
    resampling=Resampling.average,
)

dst_meta = dict(
    driver="GTiff", dtype=src_arr.dtype, width=dst_shape[1], height=dst_shape[0],
    count=1, crs=dst_crs, transform=dst_transform, nodata=src_nodata,
    compress="deflate",
)
Path(snakemake.output.elevation_conditioned_sfincs_grid).parent.mkdir(parents=True, exist_ok=True)
with rasterio.open(snakemake.output.elevation_conditioned_sfincs_grid, "w", **dst_meta) as dst:
    dst.write(dst_arr, 1)
log.info(
    f"Written: {snakemake.output.elevation_conditioned_sfincs_grid} "
    f"({dst_shape[0]}x{dst_shape[1]} px @ {grid_def['resolution']} m)"
)

profiler.stop()
log.info("Done")
