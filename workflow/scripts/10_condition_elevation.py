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

rivers = gpd.read_file(snakemake.input.river_network)
log.info(f"Loaded {len(rivers)} reaches from river_network_processed.gpkg")

n_modified, n_checked = enforce_river_monotonicity(
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
