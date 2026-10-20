"""Plot flood depth maps per scenario and a depth | attribution comparison.

Scenarios: coast_500, river_500 and compound_500. Panels use the EPSG:4326 style
of rule 16's plot_inundation_check (land polygons + river network), so depth and
attribution columns share identical panel geometry.

Outputs (all in OUT_DIR = results/2433835/runs/):
  fig_flood_map_<scenario>.png          one per scenario, depth map alone
  fig_flood_attribution_comparison.png  3 rows (event) x 2 cols (depth | attribution)
"""

from __future__ import (
    annotations,
)  # must be the first statement; section 3 below needs it

import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from _common import REPO_ROOT, label, site_path  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(REPO_ROOT / "workflow"))
from src.plots import draw_depth_map, read_raster_reprojected_for_plot  # noqa: E402

BASIN_ID = "2433835"
# results_dir is a site setting; see config/site.example.yml.
BASIN_ROOT = site_path("results_dir") / BASIN_ID
RUNS_DIR = BASIN_ROOT / "runs"
OUT_DIR = RUNS_DIR
DOMAIN_DIR = BASIN_ROOT / "preprocessing_inputs" / "domain"
LAND_POLYGONS_PATH = DOMAIN_DIR / f"{BASIN_ID}_land_polygons.gpkg"
RIVER_NETWORK_PATH = DOMAIN_DIR / f"{BASIN_ID}_river_network_clean.gpkg"

SCENARIOS = ["coast_100", "river_500", "compound_100c_500r"]

ATTR_COLORS = ["#d1c740", "#3277d3", "#d883e9"]
ATTR_LABELS = ["River", "Coastal", "Compound"]
ATTR_CMAP = ListedColormap(ATTR_COLORS)
ATTR_NORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5], ATTR_CMAP.N)


def draw_depth_panel(ax, scen):
    """Draw a scenario's max depth like plot_inundation_check's left panel."""
    tif_path = RUNS_DIR / scen / "visuals" / "max_flood_depth.tif"
    wgs_arr, (left, right, bottom, top) = read_raster_reprojected_for_plot(
        str(tif_path)
    )
    margin = max(right - left, top - bottom) * 0.03

    land = gpd.read_file(
        LAND_POLYGONS_PATH,
        bbox=(left - margin, bottom - margin, right + margin, top + margin),
    )
    rivers = gpd.read_file(RIVER_NETWORK_PATH)
    if not rivers.empty and rivers.crs is not None and rivers.crs.to_epsg() != 4326:
        rivers = rivers.to_crs("EPSG:4326")

    valid = wgs_arr[~np.isnan(wgs_arr)]
    vmax = float(np.percentile(valid, 99)) if len(valid) > 0 else 1.0
    vmax = max(vmax, 0.01)
    n_flooded = int(np.sum(~np.isnan(wgs_arr)))

    im = draw_depth_map(ax, wgs_arr, (left, right, bottom, top), land, rivers, vmax)
    ax.set_xlim(left - margin, right + margin)
    ax.set_ylim(bottom - margin, top + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    return im, n_flooded, (left, right, bottom, top)


def draw_attribution_panel(ax, scen, dst_bounds):
    """Draw a scenario's attribution mask on the same frame as draw_depth_panel."""
    tif_path = RUNS_DIR / scen / "attribution_mask.tif"
    wgs_arr, (left, right, bottom, top) = read_raster_reprojected_for_plot(
        str(tif_path), dst_bounds=dst_bounds
    )
    margin = max(right - left, top - bottom) * 0.03

    land = gpd.read_file(
        LAND_POLYGONS_PATH,
        bbox=(left - margin, bottom - margin, right + margin, top + margin),
    )
    rivers = gpd.read_file(RIVER_NETWORK_PATH)
    if not rivers.empty and rivers.crs is not None and rivers.crs.to_epsg() != 4326:
        rivers = rivers.to_crs("EPSG:4326")

    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)
    # Class 4 (spin-up/permanent water, e.g. the river channel) has no colour
    # slot and would clip to "compound"; masked as in attribution_plot.py.
    attr_arr = np.nan_to_num(wgs_arr, nan=0)
    attr_plot = np.ma.masked_where((attr_arr == 0) | (attr_arr == 4), attr_arr)
    ax.imshow(
        attr_plot,
        cmap=ATTR_CMAP,
        norm=ATTR_NORM,
        extent=(left, right, bottom, top),
        origin="upper",
        aspect="auto",
        zorder=2,
    )
    if not rivers.empty:
        rivers.plot(ax=ax, color="steelblue", linewidth=0.6, zorder=3)
    ax.set_xlim(left - margin, right + margin)
    ax.set_ylim(bottom - margin, top + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linewidth=0.5)


# per-scenario figures
for scen in SCENARIOS:
    fig, ax = plt.subplots(figsize=(8, 7))
    im, n_flooded, _ = draw_depth_panel(ax, scen)
    cbar = fig.colorbar(im, ax=ax, extend="max", fraction=0.03, pad=0.04)
    cbar.set_label("Max flood depth (m)")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(f"Max flood depth -- {label(scen)}  ({n_flooded:,} flooded pixels)")
    ax.legend(
        handles=[
            Patch(color="#6baed6", label="Flooded"),
            Patch(color="#d9d9d9", edgecolor="#aaaaaa", label="Dry / land"),
            Line2D([0], [0], color="steelblue", linewidth=1.5, label="River network"),
        ],
        loc="lower right",
        framealpha=0.9,
    )
    fig.tight_layout()
    out_path = OUT_DIR / f"fig_flood_map_{scen}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

# depth | attribution comparison
fig, axes = plt.subplots(
    len(SCENARIOS),
    2,
    figsize=(16, 5.5 * len(SCENARIOS)),
    gridspec_kw={"width_ratios": [1, 1.3]},
)

for i, scen in enumerate(SCENARIOS):
    ax_depth, ax_attr = axes[i, 0], axes[i, 1]

    im, n_flooded, bounds = draw_depth_panel(ax_depth, scen)
    cbar = fig.colorbar(im, ax=ax_depth, extend="max", fraction=0.03, pad=0.04)
    cbar.set_label("Max flood depth (m)")
    if i == len(SCENARIOS) - 1:
        ax_depth.set_xlabel("Longitude (°)")
    ax_depth.set_ylabel("Latitude (°)")
    ax_depth.annotate(
        label(scen),
        xy=(-0.16, 0.5),
        xycoords="axes fraction",
        fontsize=15,
        rotation=90,
        ha="center",
        va="center",
    )
    if i == 0:
        ax_depth.set_title("Maximum flood depth (m)", size=14)
    ax_depth.legend(
        handles=[
            Patch(color="#6baed6", label="Flooded"),
            Patch(color="#d9d9d9", edgecolor="#aaaaaa", label="Dry / land"),
            Line2D([0], [0], color="steelblue", linewidth=1.5, label="River network"),
        ],
        loc="lower right",
        framealpha=0.9,
        fontsize=10,
    )

    left, right, bottom, top = bounds
    draw_attribution_panel(ax_attr, scen, dst_bounds=(left, bottom, right, top))
    ax_attr.set_xticks([])
    ax_attr.set_ylabel("")
    if i == 0:
        ax_attr.set_title("Flood source attribution", size=14)
    ax_attr.legend(
        handles=[Patch(color=c, label=lab) for c, lab in zip(ATTR_COLORS, ATTR_LABELS)],
        loc="lower right",
        framealpha=0.9,
        fontsize=10,
    )

# fig.suptitle(f"Flood depth and source attribution by event -- basin {BASIN_ID}", fontsize=14)
fig.tight_layout()
out_path = OUT_DIR / "fig_flood_attribution_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out_path}")
