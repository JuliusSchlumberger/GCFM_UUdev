"""
plot_coastal_protection_weir.py — Read an already-produced
{basin_id}_coastal_protection_weir.gpkg (rule modelled_depth_estimation's
own output, river_processing.depth_method == "modelled" only) for one
basin and plot it: a full-domain overview plus one zoomed inset per seed
reach (is_seed=True, from river_network_clean.gpkg), so a seed's own weir
closure can actually be inspected up close -- e.g. to check whether
river_burn._flush_capped_buffer's flat cut produced a straight closure
across the channel instead of the old rounded-cap bulge.

Works for ANY basin that has already run rule modelled_depth_estimation --
does not rebuild anything, does not need rule build_sfincs to have run.

Usage:
    python tests/plot_coastal_protection_weir.py <basin_id>
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

if len(sys.argv) != 2:
    raise SystemExit("Usage: python tests/plot_coastal_protection_weir.py <basin_id>")
basin_id = sys.argv[1]

with open(REPO_ROOT / "config" / "config.yml") as f:
    config = yaml.safe_load(f)
results_dir = Path(config["results_dir"])
domain_dir = results_dir / str(basin_id) / "inputs" / "domain"

weir_path = domain_dir / f"{basin_id}_coastal_protection_weir.gpkg"
if not weir_path.exists():
    raise FileNotFoundError(
        f"{weir_path} not found -- only produced by rule modelled_depth_estimation "
        "(river_processing.depth_method == 'modelled'). Run that rule for this "
        "basin_id first."
    )

weir_gdf = gpd.read_file(weir_path)
if weir_gdf.empty:
    raise SystemExit(f"{weir_path} has no segments -- nothing to plot.")

utm_crs = weir_gdf.crs
land_path = domain_dir / f"{basin_id}_land_polygons.gpkg"
river_path = domain_dir / f"{basin_id}_river_network_clean.gpkg"

land_gdf = gpd.read_file(land_path).to_crs(utm_crs) if land_path.exists() else None
rivers_gdf = gpd.read_file(river_path).to_crs(utm_crs) if river_path.exists() else None

has_elevation = "elevation" in weir_gdf.columns
print(f"Weir segments: {len(weir_gdf)}")
if has_elevation:
    print(
        f"Crest elevation: min={weir_gdf['elevation'].min():.2f} m, "
        f"max={weir_gdf['elevation'].max():.2f} m, "
        f"median={weir_gdf['elevation'].median():.2f} m"
    )

seed_ids = []
if rivers_gdf is not None and "is_seed" in rivers_gdf.columns:
    seed_ids = rivers_gdf.loc[rivers_gdf["is_seed"].astype(bool), "reach_id"].tolist()
print(f"Seed reach(es) found for inset zoom: {len(seed_ids)}")


def _draw_base(ax):
    if land_gdf is not None:
        land_gdf.plot(
            ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=0
        )
    if rivers_gdf is not None:
        rivers_gdf.plot(ax=ax, color="#4a90d9", linewidth=0.6, zorder=1)
    if has_elevation:
        weir_gdf.plot(
            ax=ax, column="elevation", cmap="viridis", linewidth=1.2, zorder=2
        )
    else:
        weir_gdf.plot(ax=ax, color="black", linewidth=1.2, zorder=2)


n_panels = 1 + len(seed_ids)
fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7))
if n_panels == 1:
    axes = [axes]

_draw_base(axes[0])
axes[0].set_title(f"Basin {basin_id} — coastal protection weir (full domain)")
axes[0].set_aspect("equal")
axes[0].set_xlabel("x (m)")
axes[0].set_ylabel("y (m)")

for ax, seed_id in zip(axes[1:], seed_ids):
    seed_row = rivers_gdf.loc[rivers_gdf["reach_id"] == seed_id].iloc[0]
    seed_line = seed_row.geometry
    seed_start = seed_line.coords[0]
    width_m = float(getattr(seed_row, "width", 50.0) or 50.0)
    zoom_m = max(width_m * 6.0, 200.0)
    _draw_base(ax)
    ax.set_xlim(seed_start[0] - zoom_m, seed_start[0] + zoom_m)
    ax.set_ylim(seed_start[1] - zoom_m, seed_start[1] + zoom_m)
    ax.scatter(*seed_start, color="red", marker="x", s=80, zorder=3, label="seed start")
    ax.legend(loc="upper right")
    ax.set_title(f"Seed reach {seed_id} — closure check")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

fig.tight_layout()

out_dir = REPO_ROOT / "figs" / "coastal_protection_weir"
out_dir.mkdir(parents=True, exist_ok=True)
out_png = out_dir / f"{basin_id}_weir.png"
fig.savefig(out_png, dpi=150)
plt.close(fig)
print(f"Plot written: {out_png}")
