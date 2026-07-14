"""
11b_burn_river_dem.py — Burn the zbed_anchors.gpkg river-bed profile directly
into a channel-only DEM at native (fine) resolution.

See workflow/rules/11b_burn_river_dem.smk and workflow/src/river_burn.py for
the full rationale (bypasses hydromt_sfincs's own per-tile burn_river_rect,
which was confirmed to introduce spurious multi-metre artifacts).

Output covers only the buffered river-channel network (nodata elsewhere) —
fed to hydromt_sfincs as a higher-priority elevation_list entry ahead of
elevation_merged/elevation_conditioned in 13_build_sfincs.py.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import shapely

from src.log import setup_logging
from src.profiling import ScriptProfiler
from src.river_burn import burn_river_channel
from src.river_network import _as_linestring, normalize_reach_id

log = setup_logging(snakemake.log[0])
profiler = ScriptProfiler(snakemake)
burn_river_channel = profiler.wrap(burn_river_channel)

# ── inputs ────────────────────────────────────────────────────────────────────
zbed_anchors_path  = Path(snakemake.input.zbed_anchors)
river_network_path = Path(snakemake.input.river_network)
elevation_merged_path = Path(snakemake.input.elevation_merged)
topo_tiles_dir     = snakemake.input.global_topography_tiles
goco_path          = snakemake.input.goco06s_gfc
egm_path           = snakemake.input.egm2008_gfc

out_path      = Path(snakemake.output.river_burned_dem)
out_plot_path = Path(snakemake.output.plot_river_burn)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_plot_path.parent.mkdir(parents=True, exist_ok=True)

# Always burn at elevation_merged.tif's own native resolution (rule 05a's
# auto-derived working resolution) -- burning at a coarser resolution than
# the main merged DEM would be pointless (the channel-only raster is already
# cheap regardless of resolution, since it only covers the buffered river
# corridor, not the whole domain).
with rasterio.open(elevation_merged_path) as _src:
    utm_crs = _src.crs
    resolution_m = abs(_src.transform.a)

rivers = gpd.read_file(river_network_path)
zbed_anchors = gpd.read_file(zbed_anchors_path)
log.info(
    f"Loaded {len(rivers)} reach(es) from {river_network_path.name}, "
    f"{len(zbed_anchors)} zbed anchor point(s) from {zbed_anchors_path.name}"
)

# ── burn ──────────────────────────────────────────────────────────────────────
burned_arr, transform, _nodata, stats = burn_river_channel(
    rivers=rivers,
    zbed_anchors=zbed_anchors,
    topo_tiles_dir=topo_tiles_dir,
    goco_path=goco_path,
    egm_path=egm_path,
    utm_crs=utm_crs,
    resolution_m=resolution_m,
)

NODATA = np.float32(-9999.0)
out_meta = dict(
    driver="GTiff", dtype="float32",
    width=burned_arr.shape[1], height=burned_arr.shape[0],
    count=1, crs=utm_crs, transform=transform,
    nodata=float(NODATA), compress="deflate", tiled=True,
)
with rasterio.open(out_path, "w", **out_meta) as dst:
    dst.write(np.where(np.isnan(burned_arr), NODATA, burned_arr).astype(np.float32), 1)
log.info(
    f"Written: {out_path} ({stats['n_reaches_burned']} reach(es) burned, "
    f"{stats['n_reaches_skipped']} skipped, {stats['n_pixels_burned']:,} pixel(s))"
)

# ── diagnostic plot ───────────────────────────────────────────────────────────
# Left: burned raster extent + river centerlines. Right: for the reach with
# the most zbed anchor points, its own rivbed profile vs. the newly-burned
# raster sampled along the same centerline -- these should closely track
# each other (unlike hydromt_sfincs's own burn_river_rect, which did not).
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
masked = np.ma.masked_invalid(burned_arr)
extent = (
    transform.c, transform.c + burned_arr.shape[1] * transform.a,
    transform.f + burned_arr.shape[0] * transform.e, transform.f,
)
im = ax.imshow(masked, cmap="viridis", extent=extent, origin="upper", aspect="auto")
fig.colorbar(im, ax=ax, shrink=0.8, label="Burned bed elevation (m)")
rivers_utm = rivers.to_crs(utm_crs)
rivers_utm.plot(ax=ax, color="red", linewidth=0.5, alpha=0.6)
ax.set_title(
    f"Burned channel DEM ({resolution_m:.0f} m) — "
    f"{stats['n_reaches_burned']}/{stats['n_reaches_burned'] + stats['n_reaches_skipped']} reach(es)"
)
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")

ax = axes[1]
zbed_by_reach = zbed_anchors.assign(
    _reach_id=[normalize_reach_id(x) for x in zbed_anchors["reach_id"]]
)
counts = zbed_by_reach["_reach_id"].value_counts()
if len(counts) == 0:
    ax.text(0.5, 0.5, "No zbed anchor points found", ha="center", va="center",
            transform=ax.transAxes, color="grey")
else:
    sample_rid = counts.index[0]
    zbed_reach = zbed_by_reach[zbed_by_reach["_reach_id"] == sample_rid].to_crs(utm_crs)
    row = rivers_utm[
        rivers_utm["reach_id"].apply(lambda x: normalize_reach_id(x) == sample_rid)
    ]
    line = _as_linestring(row.geometry.iloc[0])

    zbed_pts = shapely.points(zbed_reach.geometry.x.to_numpy(), zbed_reach.geometry.y.to_numpy())
    zbed_along = shapely.line_locate_point(line, zbed_pts) / 1000.0
    order = np.argsort(zbed_along)

    step_m = resolution_m
    n = max(2, int(np.ceil(line.length / step_m)) + 1)
    sample_d = np.linspace(0.0, line.length, n)
    sample_pts = [line.interpolate(d) for d in sample_d]
    rows_i, cols_i = rasterio.transform.rowcol(
        transform, [p.x for p in sample_pts], [p.y for p in sample_pts]
    )
    rows_i, cols_i = np.asarray(rows_i), np.asarray(cols_i)
    in_bounds = (
        (rows_i >= 0) & (rows_i < burned_arr.shape[0])
        & (cols_i >= 0) & (cols_i < burned_arr.shape[1])
    )
    burned_vals = np.full(len(sample_d), np.nan)
    burned_vals[in_bounds] = burned_arr[rows_i[in_bounds], cols_i[in_bounds]]

    ax.scatter(
        zbed_along[order], zbed_reach["rivbed"].to_numpy()[order],
        s=10, color="black", label="zbed_anchors (source)", zorder=3,
    )
    ax.plot(sample_d / 1000.0, burned_vals, color="tab:blue", lw=1.2,
            label="burned raster (this rule's output)", zorder=2)
    ax.set_xlabel("Distance along reach (km)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(f"Sample reach {sample_rid} — burned vs. source")
    ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle(f"River burn diagnostics ({resolution_m:.0f} m native resolution)")
fig.tight_layout()
fig.savefig(out_plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
log.info(f"Plot written: {out_plot_path}")

profiler.stop()
log.info("Done")
