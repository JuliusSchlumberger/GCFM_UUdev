"""
KL_domain_figures.py -- Elevation and land-use context maps for the study
basin, publication-ready and in the same WGS84 (land polygons + river-network
overlay) style as KL_flood_maps.py's depth maps, so all figures read as one
consistent set in the paper.

Uses the final SFINCS-grid-aligned rasters -- the ones actually fed to the
model (same data behind preprocessing_inputs/visuals/sfincs_build/
02_elevation.png and the roughness raster derived from landuse_on_grid.tif)
-- rather than preprocessing_inputs/visuals/05a_elevation.png /
05b_landuse.png, whose own map_background() framing/extent differs from the
flood maps' domain crop.

Both rasters are clipped to the actual model domain polygon (the rotated
footprint 09b_grid_align_landuse's create_grid_from_region built both grids
from) rather than their own raw bounding boxes: landuse_on_grid.tif has no
nodata of its own (unlike the elevation raster, it fills its whole rectangle
with a valid class, including area outside the rotated domain), so without
this clip the two panels would show different extents. The domain outline is
then drawn as a border so the clip edge reads as a real boundary rather than
an unexplained jagged edge, and panels are cropped tight to that polygon so
it fills the plot.

Outputs (all in OUT_DIR = results/2433835/runs/):
  fig_elevation_map.png      standalone elevation panel
  fig_landuse_map.png        standalone land-use panel
  fig_domain_input_data.png  combined elevation | land-use, 1x2
  fig_elevation_classes.png  elevation only, in 1 m classes (<0 .. >5 m)
  fig_return_period_curves.png  river discharge | coastal water level RP curves, 1x2
  fig_forced_hydrograph.png  built water-level / discharge forcing per scenario, 2x3
"""

import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import shapely
import xarray as xr
from hydromt_sfincs import SfincsModel
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm, to_hex
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from src.plots import (  # noqa: E402
    _BATHY_CMAP,
    _LC_COLORS,
    _LC_NAMES,
    read_raster_reprojected_for_plot,
)

BASIN_ID = "3279946"
# Machine-specific path, read from the GCFM_RESULTS_DIR environment variable.
# Set it once in PowerShell, then restart your terminal (see
# CONTRIBUTING.md "Local machine paths"):
#   [Environment]::SetEnvironmentVariable("GCFM_RESULTS_DIR", "D:\your\results\path", "User")
BASIN_ROOT = Path(os.environ["GCFM_RESULTS_DIR"]) / BASIN_ID
RUNS_DIR = BASIN_ROOT / "runs"
OUT_DIR = RUNS_DIR
DOMAIN_DIR = BASIN_ROOT / "preprocessing_inputs" / "domain"
FORCING_DIR = BASIN_ROOT / "preprocessing_inputs" / "forcing"
LAND_POLYGONS_PATH = DOMAIN_DIR / f"{BASIN_ID}_land_polygons.gpkg"
RIVER_NETWORK_PATH = DOMAIN_DIR / f"{BASIN_ID}_river_network_clean.gpkg"
DOMAIN_POLYGON_PATH = DOMAIN_DIR / f"{BASIN_ID}_domain.gpkg"
ELEVATION_PATH = DOMAIN_DIR / f"{BASIN_ID}_elevation_conditioned_sfincs_grid.tif"
LANDUSE_PATH = DOMAIN_DIR / f"{BASIN_ID}_landuse_on_grid.tif"
PROTECTION_PATH = DOMAIN_DIR / "protection_levels.json"

# The SFINCS grid is rotated inside its bounding box, so ~45% of the raw
# elevation raster is nodata outside the active grid -- capped, not the full
# -7..745 m range, so the flood-relevant lowlands (75% of cells are < 10 m)
# keep visible contrast; values above VMAX are shown as the colormap's max
# colour (colorbar extend="max" marks this explicitly).
ELEV_VMIN, ELEV_VMAX = -10.0, 50.0

# Discrete classes for the domain-input figure: the flood-relevant lowlands
# sit within a few metres of sea level, which the continuous -10..50 m ramp
# washes out -- 1 m bins up to 5 m make those differences readable.
ELEV_CLASS_BOUNDS = [0, 1, 2, 3, 4, 5]
ELEV_CLASS_LABELS = ["< 0 m", "0–1 m", "1–2 m", "2–3 m", "3–4 m", "4–5 m", "> 5 m"]
ELEV_CLASS_COLORS = ["#2c7fb8"] + [
    to_hex(c) for c in plt.get_cmap("YlOrBr")(np.linspace(0.1, 0.95, 6))
]

MARGIN_FRAC = 0.015  # thin enough that the domain border fills the panel

_domain_gdf = gpd.read_file(DOMAIN_POLYGON_PATH)
if _domain_gdf.crs is not None and _domain_gdf.crs.to_epsg() != 4326:
    _domain_gdf = _domain_gdf.to_crs("EPSG:4326")
DOMAIN_POLY = _domain_gdf.geometry.union_all()
DOMAIN_LEFT, DOMAIN_BOTTOM, DOMAIN_RIGHT, DOMAIN_TOP = DOMAIN_POLY.bounds
DOMAIN_BOUNDS = (DOMAIN_LEFT, DOMAIN_BOTTOM, DOMAIN_RIGHT, DOMAIN_TOP)


def _mask_to_domain(data, extent):
    """Set pixels to NaN whose centre falls outside DOMAIN_POLY."""
    left, right, bottom, top = extent
    h, w = data.shape
    dx, dy = (right - left) / w, (top - bottom) / h
    lon_centers = left + dx * (np.arange(w) + 0.5)
    lat_centers = top - dy * (np.arange(h) + 0.5)
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    inside = shapely.contains_xy(DOMAIN_POLY, lon_grid, lat_grid)
    out = data.copy()
    out[~inside] = np.nan
    return out


def _panel_context(margin_frac=MARGIN_FRAC):
    margin = max(DOMAIN_RIGHT - DOMAIN_LEFT, DOMAIN_TOP - DOMAIN_BOTTOM) * margin_frac
    land = gpd.read_file(
        LAND_POLYGONS_PATH,
        bbox=(
            DOMAIN_LEFT - margin,
            DOMAIN_BOTTOM - margin,
            DOMAIN_RIGHT + margin,
            DOMAIN_TOP + margin,
        ),
    )
    rivers = gpd.read_file(RIVER_NETWORK_PATH)
    if not rivers.empty and rivers.crs is not None and rivers.crs.to_epsg() != 4326:
        rivers = rivers.to_crs("EPSG:4326")
    return land, rivers, margin


def _finish_panel(ax, margin):
    gpd.GeoSeries([DOMAIN_POLY], crs="EPSG:4326").boundary.plot(
        ax=ax, color="black", linewidth=1.3, zorder=4
    )
    ax.set_xlim(DOMAIN_LEFT - margin, DOMAIN_RIGHT + margin)
    ax.set_ylim(DOMAIN_BOTTOM - margin, DOMAIN_TOP + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linewidth=0.5)


def draw_elevation_panel(ax, classified=False):
    data, extent = read_raster_reprojected_for_plot(
        str(ELEVATION_PATH), dst_bounds=DOMAIN_BOUNDS
    )
    data = _mask_to_domain(data, extent)
    land, rivers, margin = _panel_context()

    if classified:
        cmap = ListedColormap(ELEV_CLASS_COLORS)
        norm = BoundaryNorm(ELEV_CLASS_BOUNDS, ncolors=cmap.N, extend="both")
    else:
        cmap = _BATHY_CMAP
        norm = TwoSlopeNorm(vmin=ELEV_VMIN, vcenter=0.0, vmax=ELEV_VMAX)

    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)
    im = ax.imshow(
        data,
        cmap=cmap,
        norm=norm,
        extent=extent,
        origin="upper",
        aspect="auto",
        zorder=2,
    )
    if not rivers.empty:
        rivers.plot(ax=ax, color="black", linewidth=0.5, alpha=0.6, zorder=3)
    _finish_panel(ax, margin)
    return im


# Collapse the closed/open forest subtypes (evergreen/deciduous/mixed/
# unspecified needleleaf/broadleaf, LC100 codes 111-116 / 121-126) down to
# just "Closed forest" / "Open forest" for this paper figure's legend -- the
# full per-type breakdown belongs to src.plots's pipeline diagnostics, not a
# domain-overview figure.
_CLOSED_FOREST_CODES = list(range(111, 117))
_OPEN_FOREST_CODES = list(range(121, 127))
_GROUP_NAMES = {110: "Closed forest", 120: "Open forest"}
_GROUP_COLORS = {110: "#1a7a1a", 120: "#5cb85c"}
_LC_NAMES_GROUPED = {**_LC_NAMES, **_GROUP_NAMES}
_LC_COLORS_GROUPED = {**_LC_COLORS, **_GROUP_COLORS}


def draw_landuse_panel(ax):
    data, extent = read_raster_reprojected_for_plot(
        str(LANDUSE_PATH), dst_bounds=DOMAIN_BOUNDS
    )
    data = _mask_to_domain(data, extent)
    data[np.isin(data, _CLOSED_FOREST_CODES)] = 110
    data[np.isin(data, _OPEN_FOREST_CODES)] = 120
    land, rivers, margin = _panel_context()

    present = sorted({int(v) for v in np.unique(data[~np.isnan(data)])})
    code_to_idx = {c: i for i, c in enumerate(present)}
    data_idx = np.full(data.shape, np.nan)
    for c, i in code_to_idx.items():
        data_idx[data == c] = i
    cmap = ListedColormap([_LC_COLORS_GROUPED.get(c, "#808080") for c in present])
    norm = BoundaryNorm(boundaries=range(len(present) + 1), ncolors=len(present))

    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)
    ax.imshow(
        data_idx,
        cmap=cmap,
        norm=norm,
        extent=extent,
        origin="upper",
        aspect="auto",
        zorder=2,
    )
    if not rivers.empty:
        rivers.plot(ax=ax, color="steelblue", linewidth=0.6, zorder=3)
    _finish_panel(ax, margin)
    return present


def _lc_legend_handles(present):
    return [
        Patch(
            color=_LC_COLORS_GROUPED.get(c, "#808080"),
            label=_LC_NAMES_GROUPED.get(c, str(c)),
        )
        for c in present
    ]


# --------------------------------------------------------- 1. standalone elevation
fig, ax = plt.subplots(figsize=(8, 7))
im = draw_elevation_panel(ax)
cbar = fig.colorbar(im, ax=ax, extend="max", fraction=0.03, pad=0.04)
cbar.set_label("Elevation (m)")
ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")
ax.set_title(f"Elevation -- basin {BASIN_ID}")
fig.tight_layout()
# out_path = OUT_DIR / "fig_elevation_map.png"
# fig.savefig(out_path, dpi=150, bbox_inches="tight")
# plt.close(fig)
# print(f"Wrote {out_path}")

# --------------------------------------------------------- 2. standalone land use
fig, ax = plt.subplots(figsize=(9, 7))
present = draw_landuse_panel(ax)
ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")
ax.set_title(f"Land use (Copernicus LC100, 2019) -- basin {BASIN_ID}")
ax.legend(
    handles=_lc_legend_handles(present), loc="lower right", fontsize=8, framealpha=0.9
)
fig.tight_layout()
# out_path = OUT_DIR / "fig_landuse_map.png"
# fig.savefig(out_path, dpi=150, bbox_inches="tight")
# plt.close(fig)
# print(f"Wrote {out_path}")

# --------------------------------------------------------- 3. combined elevation | land use
# The colorbar sits between the two maps in its own gridspec column (rather
# than fig.colorbar(ax=ax1), which would shrink ax1's box relative to ax2's)
# so both map panels still start from an identical box and render at the
# same size, while the colorbar stays visually next to the elevation panel.
fig = plt.figure(figsize=(15, 7), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.05, 1], wspace=0.08)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 2])

# cax is an inset of a hidden placeholder rather than a bare gridspec subplot
# -- constrained_layout stretches a plain subplot to fill its whole cell
# (full figure height), which is what made the colorbar look oversized; a
# fixed-fraction inset lets height and thickness be tuned independently.
_cbar_cell = fig.add_subplot(gs[0, 1])
_cbar_cell.axis("off")
cax = _cbar_cell.inset_axes([0.3, 0.15, 0.45, 0.7])

im = draw_elevation_panel(ax1)
cbar = fig.colorbar(im, cax=cax, extend="max")
cbar.set_label("Elevation (m)")
ax1.set_xlabel("Longitude (°)")
ax1.set_ylabel("Latitude (°)")
# ax1.set_title("Elevation", size=13)

present = draw_landuse_panel(ax2)
ax2.set_xlabel("Longitude (°)")
ax2.set_ylabel("")
ax2.set_yticks([])
# ax2.set_title("Land use (Copernicus LC100)", size=13)
ax2.legend(
    handles=_lc_legend_handles(present), loc="lower right", fontsize=9, framealpha=0.9
)

for ax, panel in ((ax1, "(a)"), (ax2, "(b)")):
    ax.text(
        0.02,
        0.98,
        panel,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        zorder=5,
    )

out_path = OUT_DIR / "fig_domain_input_data.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out_path}")

# --------------------------------------------------------- 3b. classified elevation
fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
draw_elevation_panel(ax, classified=True)
ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")
ax.legend(
    handles=[
        Patch(color=c, label=label)
        for c, label in zip(ELEV_CLASS_COLORS, ELEV_CLASS_LABELS)
    ],
    title="Elevation",
    loc="lower right",
    fontsize=9,
    framealpha=0.9,
)

out_path = OUT_DIR / "fig_elevation_classes.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out_path}")

# --------------------------------------------------------- 4. return-period curves
# decode_times=False: "time"'s units ("hours since simulation start") is a
# relative offset, not a real calendar date -- xarray's CF-time decoder
# chokes trying to parse "simulation start" as a reference date.
river_ds = xr.open_dataset(FORCING_DIR / "river_forcing.nc", decode_times=False)
surge_ds = xr.open_dataset(FORCING_DIR / "surge_forcing.nc", decode_times=False)

# the single most significant river crossing (largest high-RP discharge),
# same selection basis 07_get_boundary_forcings.py's own diagnostic plot uses
active = river_ds["has_glofas"].values.astype(bool)
rp_table = river_ds["discharge_rp_table"].values[active]
i_main = np.nanargmax(rp_table[:, -1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")

river_rp = river_ds["return_period"].values
in_range = river_rp <= 1000
ax1.plot(river_rp[in_range], rp_table[i_main][in_range], color="#1b4965")
ax1.set_xscale("log")
ax1.set_xlabel("Return period (years)")
ax1.set_ylabel("Discharge (m3/s)")
ax1.set_title("River discharge")

table_rp = surge_ds["table_rp"].values
storm_tide_table = surge_ds["storm_tide_rp_table"].values
for i in range(storm_tide_table.shape[0]):
    ax2.plot(table_rp, storm_tide_table[i], marker="o", markersize=3)
ax2.set_xscale("log")
ax2.set_xlabel("Return period (years)")
ax2.set_ylabel("Water level (m)")
ax2.set_title("Coastal water level")

# each panel's own protection standard (FLOPROS, 04_protection_levels), next to the
# 100-yr design event, so the gap between what's protected and what's forced shows
protection = json.loads(PROTECTION_PATH.read_text())
panel_protection = {ax1: protection["riverine_rp_yr"], ax2: protection["coastal_rp_yr"]}

for ax, panel in ((ax1, "(a)"), (ax2, "(b)")):
    ax.grid(alpha=0.3)
    ax.set_xlim(0.7, 1.5e3)
    ax.axvline(
        100, color="#b44225", linestyle="--", linewidth=1, label="100-year event"
    )
    rp_protected = panel_protection[ax]
    ax.axvline(
        rp_protected,
        color="grey",
        linestyle="--",
        linewidth=1.2,
        label=f"Protection standard ({rp_protected:.0f}-year)",
    )
    # top left, just below the (a)/(b) panel letter
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.9), fontsize=8, frameon=False)
    ax.text(
        0.02,
        0.97,
        panel,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

# fig.suptitle("Flood hazard return-period curves")
fig.tight_layout()
out_path = OUT_DIR / "fig_return_period_curves.png"
fig.savefig(out_path, dpi=150, facecolor="white")
plt.close(fig)
river_ds.close()
surge_ds.close()
print(f"Wrote {out_path}")

# --------------------------------------------------------- 5. forced hydrographs
# The REAL, actually-built forcing for each scenario's own SFINCS run, read
# straight off its sfincs.bzs/sfincs.dis via hydromt_sfincs (real calendar
# time, RP/protection-floor/SLR/multiplier all already resolved). NOT
# river_forcing.nc's/surge_forcing.nc's own stored 'discharge'/'water_level'
# -- those are a basin-level, scenario-independent preview at a fixed
# diagnostic RP, never what a specific scenario actually ran with (same fix
# 13_build_sfincs.py's own 05_forcing.png already applies -- see its comment
# above the forcing-plot block).
# 2x3 grid: one column per scenario, water level on top / discharge on
# bottom, y-axis shared within each row so magnitudes are directly
# comparable across scenarios.
SCENARIOS = ["coast_100_03SLR", "river_100_03SLR", "compound_100_03SLR"]
SCENARIO_LABELS = {
    "coast_100_03SLR": "Coastal flood",
    "river_100_03SLR": "River flood",
    "compound_100_03SLR": "Compound flood",
}
SCENARIO_COLOR = "#14b5dd"
# if these scenarios' own slr_m (config/scenarios.yml, per-scenario) is 0.0,
# the plotted water level IS the actual built forcing, with no SLR -- shown
# as a flat reference line AT y=SLR_M (an illustrative SLR magnitude, not a
# baseline+SLR shift) on the coastal scenarios only, so the flood peak's
# height can be read directly against it.
SLR_M = 0.5
SLR_SCENARIOS = {"coast_100"}  # , "compound_100c_500r"}

fig, axes = plt.subplots(
    2,
    len(SCENARIOS),
    figsize=(3.2 * len(SCENARIOS), 5.5),
    sharex=True,
    sharey="row",
    facecolor="white",
)

for j, scen in enumerate(SCENARIOS):
    mod = SfincsModel(root=str(RUNS_DIR / scen / "sfincs"), mode="r")

    wl_comp = mod.get_component("water_level")
    wl_comp.read()
    wl = wl_comp.data  # dims (time, index), var 'bzs'
    wl_now = wl["bzs"].mean(dim="index").values
    axes[0, j].plot(wl.time.values, wl_now, color=SCENARIO_COLOR, lw=1.8)

    if scen in SLR_SCENARIOS:
        axes[0, j].axhline(
            SLR_M,
            color=SCENARIO_COLOR,
            lw=1.5,
            linestyle=":",
            label=f"{SLR_M:g} m SLR",
        )
        axes[0, j].legend(fontsize=7, frameon=False, loc="upper left")

    dis_comp = mod.get_component("discharge_points")
    dis_comp.read()
    dis = dis_comp.data  # dims (time, index), var 'dis'
    axes[1, j].plot(
        dis.time.values, dis["dis"].mean(dim="index").values, color="#2411ce", lw=1.8
    )

    for i in (0, 1):
        axes[i, j].grid(alpha=0.3)
        axes[i, j].spines[["top", "right"]].set_visible(False)

    axes[0, j].set_title(SCENARIO_LABELS.get(scen, scen))

axes[0, 0].set_ylabel("Water level (m)")
axes[1, 0].set_ylabel("Discharge (m³/s)")

# sharex=True only links the axis range, not tick-label visibility, and
# AutoDateLocator's default tick density (fine in 13_build_sfincs.py's own
# single wide 10" figure) is too cluttered at this grid's ~3" column width --
# force 12-hourly ticks instead and rotate labels to keep them legible.
locator = mdates.HourLocator(byhour=[0, 12])
formatter = mdates.ConciseDateFormatter(locator)
for ax in axes.flat:
    ax.label_outer()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", labelrotation=45)
    # "2000" is just this pipeline's dummy simulation reference year, not a
    # meaningful date -- drop ConciseDateFormatter's repeated offset text
    ax.xaxis.get_offset_text().set_visible(False)

fig.suptitle("Forcing hydrographs")
fig.tight_layout()
out_path = OUT_DIR / "fig_forced_hydrograph.png"
fig.savefig(out_path, dpi=200, facecolor="white")
plt.close(fig)
print(f"Wrote {out_path}")
