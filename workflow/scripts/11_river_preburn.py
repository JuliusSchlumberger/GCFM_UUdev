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

from src.log import setup_logging
from src.profiling import ScriptProfiler
from src.river_preburn import compute_river_bed_points
from src.river_network import (
    _as_linestring,
    _sample_line_cells,
    build_downstream_adjacency,
    normalize_reach_id,
)

log = setup_logging(snakemake.log[0])
profiler = ScriptProfiler(snakemake)
compute_river_bed_points = profiler.wrap(compute_river_bed_points)

# ── compute bed points ────────────────────────────────────────────────────────

rivers = gpd.read_file(snakemake.input.river_network)
log.info(f"Loaded {len(rivers)} reaches from {Path(snakemake.input.river_network).name}")

zbed_gdf = compute_river_bed_points(
    rivers=rivers,
    elevation_path=snakemake.input.elevation_conditioned,
    depth_column="rivdph",
)

Path(snakemake.output.zbed_anchors).parent.mkdir(parents=True, exist_ok=True)
zbed_gdf.to_file(snakemake.output.zbed_anchors, driver="GPKG")
log.info(f"Bed points written: {len(zbed_gdf)} points → {snakemake.output.zbed_anchors}")

# ── diagnostic plot ───────────────────────────────────────────────────────────
# For each seed: one subplot showing DEM_conditioned elevation (grey) and
# rivbed = DEM - rivdph (coloured per reach) along the downstream network.
# This lets us verify the bed profile is physically consistent.

Path(snakemake.output.plot_preburn).parent.mkdir(parents=True, exist_ok=True)
plt.ioff()

with rasterio.open(snakemake.input.elevation_conditioned) as _src:
    cond_arr   = _src.read(1)
    transform  = _src.transform
    cond_nd    = _src.nodata
    raster_crs = _src.crs

step_m = abs(transform.a)
rivers_proj = rivers.to_crs(raster_crs)

line_by_rid:   dict[str, object] = {}
length_by_rid: dict[str, float]  = {}
rivdph_by_rid: dict[str, float]  = {}
for _row in rivers_proj.itertuples(index=False):
    _rid = normalize_reach_id(_row.reach_id)
    if _rid is None:
        continue
    _g = _as_linestring(_row.geometry)
    if _g is not None and _g.length > 0:
        line_by_rid[_rid]   = _g
        length_by_rid[_rid] = _g.length
    _d = getattr(_row, "rivdph", None)
    rivdph_by_rid[_rid] = float(_d) if _d is not None and np.isfinite(float(_d)) else 0.0

downstream_adj = build_downstream_adjacency(rivers)
seeds = [
    normalize_reach_id(r.reach_id)
    for r in rivers.itertuples(index=False)
    if not pd.isna(getattr(r, "is_seed", None)) and bool(getattr(r, "is_seed", False))
]


def _sample_dem_and_bed(rid, dist_from_seed):
    """Return (x_km, z_dem, z_bed) lists for this reach."""
    line = line_by_rid.get(rid)
    if line is None:
        return [], [], []
    rivdph = rivdph_by_rid.get(rid, 0.0)
    xs, z_dem, z_bed = [], [], []
    for c in _sample_line_cells(line, transform, cond_arr.shape, step_m):
        v = float(cond_arr[c["row"], c["col"]])
        if (cond_nd is not None and v == cond_nd) or not np.isfinite(v):
            continue
        x_km = (dist_from_seed + c["along_m"]) / 1000.0
        xs.append(x_km)
        z_dem.append(v)
        z_bed.append(v - rivdph)
    return xs, z_dem, z_bed


cmap = plt.cm.tab20

if not seeds:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(0.5, 0.5, "No seed reaches found — cannot plot downstream profile",
            ha="center", va="center", transform=ax.transAxes, color="grey")
    fig.tight_layout()
    fig.savefig(snakemake.output.plot_preburn, dpi=150, bbox_inches="tight")
    plt.close(fig)
else:
    n_seeds = len(seeds)
    fig, axes = plt.subplots(n_seeds, 1, figsize=(13, 5 * n_seeds), squeeze=False)

    for ax_row, seed in enumerate(seeds):
        ax = axes[ax_row][0]

        # BFS downstream
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

        # DEM_conditioned: grey background scatter
        for rid in visit_order:
            xs, z_dem, _ = _sample_dem_and_bed(rid, dist_from_seed[rid])
            if xs:
                ax.scatter(xs, z_dem, s=2, color="lightgrey", alpha=0.5,
                           zorder=1, rasterized=True)

        # rivbed = DEM - rivdph: coloured per reach
        for rid in visit_order:
            xs, _, z_bed = _sample_dem_and_bed(rid, dist_from_seed[rid])
            if xs:
                ax.scatter(xs, z_bed, s=3, color=reach_colour[rid], alpha=0.75,
                           zorder=2, rasterized=True)

        # Reach-boundary vertical lines + short ID labels
        ax_ymin, ax_ymax = ax.get_ylim()
        for i, rid in enumerate(visit_order):
            x_km = dist_from_seed[rid] / 1000.0
            ax.axvline(x_km, color=reach_colour[rid], linewidth=0.7,
                       linestyle="--", alpha=0.6, zorder=3)
            y_lbl = ax_ymax - 0.05 * (ax_ymax - ax_ymin) * (1 + i % 2)
            ax.text(x_km + 0.3, y_lbl, rid[-6:], fontsize=4,
                    color=reach_colour[rid], rotation=90, va="top",
                    clip_on=True, zorder=4)

        ax.legend(handles=[
            Line2D([0], [0], marker="o", linestyle="", color="lightgrey",
                   markersize=5, alpha=0.6, label="DEM_conditioned"),
            Line2D([0], [0], marker="o", linestyle="", color="black",
                   markersize=5, alpha=0.75, label="rivbed = DEM − rivdph (colour = reach)"),
            Line2D([0], [0], linestyle="--", color="grey", linewidth=1,
                   alpha=0.6, label="reach boundary"),
        ], fontsize=8, loc="upper right", framealpha=0.85)
        ax.set_xlabel("Distance from seed (km)")
        ax.set_ylabel("Elevation (m)")
        ax.set_title(
            f"Seed {seed} — river bed profile: DEM_conditioned − rivdph "
            f"({len(visit_order)} reaches)"
        )
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"River bed points: {len(zbed_gdf)} points "
        f"(rivbed = DEM_conditioned − rivdph, passed as gdf_zb to SFINCS subgrid)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(snakemake.output.plot_preburn, dpi=150, bbox_inches="tight")
    plt.close(fig)

log.info(f"Plot written: {snakemake.output.plot_preburn}")

profiler.stop()
log.info("Done")
