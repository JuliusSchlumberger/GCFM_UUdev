"""
Standalone max flood depth map per flood event, for the no-adaptation baseline run
(<event>/visuals/max_flood_depth.tif) -- one publication figure per event, in the same
WGS84 style (land polygons, river network, domain outline) as study_area_figures.py.

Depth is shown in discrete classes on one shared scale, so the events compare directly;
open sea (the zsini sea mask) is painted over, since depth there is not flooding. The
flooded land / urban exposure annotation is read from the run's own
metrics/flood_metrics.csv, so it matches the numbers reported elsewhere.

Outputs: <basin>/runs/fig_flood_maps/fig_flood_map_<event>.png
"""

import os
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from src.plots import read_raster_reprojected_for_plot  # noqa: E402

BASIN_ID = "3279946"  # Chao Phraya delta
# Machine-specific path, read from the GCFM_RESULTS_DIR environment variable.
# Set it once in PowerShell, then restart your terminal (see
# CONTRIBUTING.md "Local machine paths"):
#   [Environment]::SetEnvironmentVariable("GCFM_RESULTS_DIR", "D:\your\results\path", "User")
BASIN_ROOT = Path(os.environ["GCFM_RESULTS_DIR"]) / BASIN_ID
RUNS = BASIN_ROOT / "runs"
DOMAIN = BASIN_ROOT / "preprocessing_inputs" / "domain"
OUT_DIR = RUNS / "fig_flood_maps"

EVENTS = {
    "coast_100_03SLR": "Coastal flood",
    "river_100_03SLR": "River flood",
    "compound_100_03SLR": "Compound flood",
}

# depth classes (m); the lowest edge is the pipeline's wet/dry cutoff (hmin)
DEPTH_BOUNDS = [0.05, 0.5, 1.0, 1.5, 2.0, 3.0]
DEPTH_COLORS = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b"]
LAND_COLOR = "#e0e0e0"
SEA_COLOR = "#f4f6f8"
MARGIN_FRAC = 0.015

_domain = gpd.read_file(DOMAIN / f"{BASIN_ID}_domain.gpkg").to_crs("EPSG:4326")
DOMAIN_POLY = _domain.geometry.union_all()


def event_subtitle(event):
    """'coast_100_03SLR' -> '100-year event, 0.3 m sea-level rise'."""
    _, rp, slr = event.split("_")
    return f"{rp}-year event, {int(slr[:-3]) / 10:g} m sea-level rise"


def plot_event(event, title, land, rivers):
    left, bottom, right, top = DOMAIN_POLY.bounds
    bounds = (left, bottom, right, top)
    depth, extent = read_raster_reprojected_for_plot(
        str(RUNS / event / "visuals" / "max_flood_depth.tif"), dst_bounds=bounds
    )
    sea, sea_extent = read_raster_reprojected_for_plot(
        str(DOMAIN / f"{BASIN_ID}_zsini_sea_cells_on_grid.tif"), dst_bounds=bounds
    )
    metrics = pd.read_csv(RUNS / event / "metrics" / "flood_metrics.csv").iloc[0]

    fig, ax = plt.subplots(figsize=(7.5, 7), facecolor="white", layout="constrained")
    ax.set_facecolor(SEA_COLOR)
    land.plot(ax=ax, color=LAND_COLOR, edgecolor="#b0b0b0", linewidth=0.3, zorder=1)
    # under the depth layer: the network only shows where the channel stays dry
    rivers.plot(ax=ax, color="#7f9db5", linewidth=0.5, zorder=2)

    cmap = ListedColormap(DEPTH_COLORS)
    norm = BoundaryNorm(DEPTH_BOUNDS, ncolors=cmap.N, extend="max")
    im = ax.imshow(
        depth,
        cmap=cmap,
        norm=norm,
        extent=extent,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        zorder=3,
    )
    is_sea = sea == 1.0
    ax.imshow(
        np.ma.masked_where(~is_sea, is_sea),
        cmap=ListedColormap([SEA_COLOR]),
        extent=sea_extent,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        zorder=4,
    )
    # outline the sea mask so its inland reaches (coastal ponds below sea level,
    # which the model starts as sea) read as water, not as a gap in the map
    s_left, s_right, s_bottom, s_top = sea_extent
    ax.contour(
        is_sea.astype(float),
        levels=[0.5],
        colors="#9aa5b1",
        linewidths=0.4,
        extent=(s_left, s_right, s_top, s_bottom),  # row 0 is the top row
        zorder=4,
    )
    gpd.GeoSeries([DOMAIN_POLY], crs="EPSG:4326").boundary.plot(
        ax=ax, color="#333333", linewidth=1.0, zorder=5
    )

    margin = max(right - left, top - bottom) * MARGIN_FRAC
    ax.set_xlim(left - margin, right + margin)
    ax.set_ylim(bottom - margin, top + margin)
    ax.set_aspect("equal")
    ax.locator_params(nbins=5)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Longitude (°)", fontsize=9)
    ax.set_ylabel("Latitude (°)", fontsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    cbar = fig.colorbar(
        im,
        ax=ax,
        extend="max",
        ticks=DEPTH_BOUNDS,
        format="%g",
        fraction=0.035,
        pad=0.02,
    )
    cbar.set_label("Max flood depth (m)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.6)

    ax.text(
        0.03,
        0.97,
        f"Flooded land  {metrics.flooded_area_km2:,.0f} km² "
        f"({metrics.flood_extent_pct:.0f}%)\n"
        f"Urban exposed  {metrics.urban_exposed_km2:,.0f} km²\n"
        f"Mean depth  {metrics.mean_depth_m:.1f} m",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#b0b0b0", lw=0.6),
        zorder=6,
    )
    ax.legend(
        handles=[
            Patch(facecolor=LAND_COLOR, edgecolor="#b0b0b0", label="Dry land"),
            Patch(facecolor=SEA_COLOR, edgecolor="#9aa5b1", label="Sea"),
        ],
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
        edgecolor="#b0b0b0",
    )
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=18)
    ax.text(
        0,
        1.012,
        event_subtitle(event),
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        va="bottom",
    )

    out = OUT_DIR / f"fig_flood_map_{event}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    left, bottom, right, top = DOMAIN_POLY.bounds
    margin = max(right - left, top - bottom) * MARGIN_FRAC
    land = gpd.read_file(
        DOMAIN / f"{BASIN_ID}_land_polygons.gpkg",
        bbox=(left - margin, bottom - margin, right + margin, top + margin),
    )
    rivers = gpd.read_file(DOMAIN / f"{BASIN_ID}_river_network_clean.gpkg").to_crs(
        "EPSG:4326"
    )
    for event, title in EVENTS.items():
        plot_event(event, title, land, rivers)
