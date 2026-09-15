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
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from src.plots import (  # noqa: E402
    _BATHY_CMAP,
    _LC_COLORS,
    _LC_NAMES,
    read_raster_reprojected_for_plot,
)

BASIN_ID = "2433835"
BASIN_ROOT = Path(r"D:\GCFM_UU\results") / BASIN_ID
RUNS_DIR = BASIN_ROOT / "runs"
OUT_DIR = RUNS_DIR
DOMAIN_DIR = BASIN_ROOT / "preprocessing_inputs" / "domain"
LAND_POLYGONS_PATH = DOMAIN_DIR / f"{BASIN_ID}_land_polygons.gpkg"
RIVER_NETWORK_PATH = DOMAIN_DIR / f"{BASIN_ID}_river_network_clean.gpkg"
DOMAIN_POLYGON_PATH = DOMAIN_DIR / f"{BASIN_ID}_domain.gpkg"
ELEVATION_PATH = DOMAIN_DIR / f"{BASIN_ID}_elevation_conditioned_sfincs_grid.tif"
LANDUSE_PATH = DOMAIN_DIR / f"{BASIN_ID}_landuse_on_grid.tif"

# The SFINCS grid is rotated inside its bounding box, so ~45% of the raw
# elevation raster is nodata outside the active grid -- capped, not the full
# -7..745 m range, so the flood-relevant lowlands (75% of cells are < 10 m)
# keep visible contrast; values above VMAX are shown as the colormap's max
# colour (colorbar extend="max" marks this explicitly).
ELEV_VMIN, ELEV_VMAX = -10.0, 50.0

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


def draw_elevation_panel(ax):
    data, extent = read_raster_reprojected_for_plot(
        str(ELEVATION_PATH), dst_bounds=DOMAIN_BOUNDS
    )
    data = _mask_to_domain(data, extent)
    land, rivers, margin = _panel_context()

    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)
    im = ax.imshow(
        data,
        cmap=_BATHY_CMAP,
        norm=TwoSlopeNorm(vmin=ELEV_VMIN, vcenter=0.0, vmax=ELEV_VMAX),
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

out_path = OUT_DIR / "fig_domain_input_data.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out_path}")
