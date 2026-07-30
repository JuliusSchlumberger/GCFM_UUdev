import json
from collections import deque
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS as RasterioCRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

from src.log import setup_logging
from src.profiling import ScriptProfiler
from src.river_conditioning import enforce_river_monotonicity
from src.river_network import (
    _as_linestring,
    _sample_line_cells,
    build_downstream_adjacency,
    normalize_reach_id,
)

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
with open(snakemake.input.sfincs_grid) as f:
    grid_def = json.load(f)
grid_transform = Affine(*grid_def["transform"])
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

# ── diagnostic plot ───────────────────────────────────────────────────────────
# Downstream elevation profile along river centerlines comparing the DEM
# before (elevation_merged) and after (elevation_conditioned) conditioning.
# Layout: n_seeds rows × 2 columns.  Each reach gets a distinct tab20 colour;
# dashed vertical lines mark reach start positions for localisation.

Path(snakemake.output.plot_conditioning).parent.mkdir(parents=True, exist_ok=True)
plt.ioff()

with rasterio.open(snakemake.input.elevation_merged) as _src:
    merged_arr  = _src.read(1)
    transform   = _src.transform
    merged_nd   = _src.nodata
    raster_crs  = _src.crs

with rasterio.open(snakemake.output.elevation_conditioned) as _src:
    cond_arr = _src.read(1)
    cond_nd  = _src.nodata

step_m = abs(transform.a)
rivers_proj = rivers.to_crs(raster_crs)

line_by_rid:   dict[str, object] = {}
length_by_rid: dict[str, float]  = {}
for _row in rivers_proj.itertuples(index=False):
    _rid = normalize_reach_id(_row.reach_id)
    if _rid is None:
        continue
    _g = _as_linestring(_row.geometry)
    if _g is not None and _g.length > 0:
        line_by_rid[_rid]   = _g
        length_by_rid[_rid] = _g.length

downstream_adj = build_downstream_adjacency(rivers)
seeds = [
    normalize_reach_id(r.reach_id)
    for r in rivers.itertuples(index=False)
    if not pd.isna(getattr(r, "is_seed", None)) and bool(getattr(r, "is_seed", False))
]

def _sample_elev(rid, dist_from_seed, elev_arr, elev_nd):
    line = line_by_rid.get(rid)
    if line is None:
        return [], []
    xs, ys = [], []
    for c in _sample_line_cells(line, transform, elev_arr.shape, step_m):
        v = float(elev_arr[c["row"], c["col"]])
        if (elev_nd is not None and v == elev_nd) or not np.isfinite(v):
            continue
        xs.append((dist_from_seed + c["along_m"]) / 1000.0)
        ys.append(v)
    return xs, ys

cmap = plt.cm.tab20

if not seeds:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(0.5, 0.5, "No seed reaches found — cannot plot downstream profile",
            ha="center", va="center", transform=ax.transAxes, color="grey")
    fig.tight_layout()
    fig.savefig(snakemake.output.plot_conditioning, dpi=150, bbox_inches="tight")
    plt.close(fig)
else:
    n_seeds = len(seeds)
    panel_specs = [
        ("elevation_merged (before)",      merged_arr, merged_nd),
        ("elevation_conditioned (after)",   cond_arr,   cond_nd),
    ]
    fig, axes = plt.subplots(n_seeds, 2, figsize=(14, 5 * n_seeds), squeeze=False)

    for ax_row, seed in enumerate(seeds):
        # BFS downstream from this seed
        dist_from_seed: dict[str, float] = {seed: 0.0}
        queue: deque[str] = deque([seed])
        visit_order: list[str] = [seed]
        while queue:
            rid = queue.popleft()
            for dn in downstream_adj.get(rid, []):
                if dn not in dist_from_seed:
                    dist_from_seed[dn] = dist_from_seed[rid] + length_by_rid.get(rid, 0.0)
                    visit_order.append(dn)
                    queue.append(dn)

        reach_colour = {rid: cmap(i % 20) for i, rid in enumerate(visit_order)}

        for ax_col, (label, elev_arr, elev_nd) in enumerate(panel_specs):
            ax = axes[ax_row][ax_col]

            for rid in visit_order:
                xs, ys = _sample_elev(rid, dist_from_seed[rid], elev_arr, elev_nd)
                if xs:
                    ax.scatter(xs, ys, s=3, color=reach_colour[rid],
                               alpha=0.65, zorder=2, rasterized=True)

            # Reach-boundary vertical lines + short ID labels
            ax_ymin, ax_ymax = ax.get_ylim()
            for i, rid in enumerate(visit_order):
                x_km = dist_from_seed[rid] / 1000.0
                ax.axvline(x_km, color=reach_colour[rid], linewidth=0.8,
                           linestyle="--", alpha=0.7, zorder=3)
                y_lbl = ax_ymax - 0.05 * (ax_ymax - ax_ymin) * (1 + i % 2)
                ax.text(x_km + 0.3, y_lbl, rid[-6:], fontsize=4,
                        color=reach_colour[rid], rotation=90, va="top",
                        clip_on=True, zorder=4)

            ax.legend(handles=[
                Line2D([0], [0], marker="o", linestyle="", color="grey",
                       markersize=5, alpha=0.7, label="colour = reach"),
                Line2D([0], [0], linestyle="--", color="grey", linewidth=1,
                       alpha=0.6, label="reach boundary"),
            ], fontsize=8, loc="upper right", framealpha=0.85)
            ax.set_xlabel("Distance from seed (km)")
            ax.set_ylabel("DEM elevation (m)")
            ax.set_title(
                f"Seed {seed} — {label} ({len(visit_order)} reaches)"
            )
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"06d river conditioning: {n_total_pixels}/{n_checked} centerline pixel(s) lowered "
        f"({pct_modified:.1f}%) across {len(n_modified)} reach(es)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(snakemake.output.plot_conditioning, dpi=150, bbox_inches="tight")
    plt.close(fig)

log.info(f"Plot written: {snakemake.output.plot_conditioning}")

profiler.stop()
log.info("Done")
