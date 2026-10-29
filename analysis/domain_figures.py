"""Plot elevation and land-use maps of the study basin for the paper.

Same WGS84 style (land polygons + river network) as flood_maps.py. Uses the
SFINCS-grid-aligned rasters fed to the model. Both are clipped to the
model domain polygon: landuse_on_grid.tif has no nodata and fills its whole
rectangle with valid classes, so unclipped panels would differ in extent.

Output, in OUT_DIR = <results_dir>/<BASIN_ID>/runs/ (BASIN_ID = 3279946):
  fig_domain_input_data.png  combined elevation | land-use, 1x2
"""

import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely
from _common import REPO_ROOT, script_parser, site_path  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

sys.path.insert(0, str(REPO_ROOT / "workflow"))
from src.plots import (  # noqa: E402
    _BATHY_CMAP,
    _LC_COLORS,
    _LC_NAMES,
    read_raster_reprojected_for_plot,
)

BASIN_ID = "3279946"

# Capped well below the -7..745 m range so the lowlands (75% of cells < 10 m)
# keep contrast; values above VMAX take the max colour (colorbar extend="max").
ELEV_VMIN, ELEV_VMAX = -10.0, 50.0

MARGIN_FRAC = 0.015  # thin enough that the domain border fills the panel


def _read_domain_poly(domain_dir):
    domain_gdf = gpd.read_file(domain_dir / f"{BASIN_ID}_domain.gpkg")
    if domain_gdf.crs is not None and domain_gdf.crs.to_epsg() != 4326:
        domain_gdf = domain_gdf.to_crs("EPSG:4326")
    return domain_gdf.geometry.union_all()


def _mask_to_domain(data, extent, domain_poly):
    """Set pixels to NaN whose centre falls outside domain_poly."""
    left, right, bottom, top = extent
    h, w = data.shape
    dx, dy = (right - left) / w, (top - bottom) / h
    lon_centers = left + dx * (np.arange(w) + 0.5)
    lat_centers = top - dy * (np.arange(h) + 0.5)
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    inside = shapely.contains_xy(domain_poly, lon_grid, lat_grid)
    out = data.copy()
    out[~inside] = np.nan
    return out


def _panel_context(domain_dir, domain_poly, margin_frac=MARGIN_FRAC):
    left, bottom, right, top = domain_poly.bounds
    margin = max(right - left, top - bottom) * margin_frac
    land = gpd.read_file(
        domain_dir / f"{BASIN_ID}_land_polygons.gpkg",
        bbox=(
            left - margin,
            bottom - margin,
            right + margin,
            top + margin,
        ),
    )
    rivers = gpd.read_file(domain_dir / f"{BASIN_ID}_river_network_clean.gpkg")
    if not rivers.empty and rivers.crs is not None and rivers.crs.to_epsg() != 4326:
        rivers = rivers.to_crs("EPSG:4326")
    return land, rivers, margin


def _finish_panel(ax, margin, domain_poly):
    left, bottom, right, top = domain_poly.bounds
    gpd.GeoSeries([domain_poly], crs="EPSG:4326").boundary.plot(
        ax=ax, color="black", linewidth=1.3, zorder=4
    )
    ax.set_xlim(left - margin, right + margin)
    ax.set_ylim(bottom - margin, top + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linewidth=0.5)


def draw_elevation_panel(ax, domain_dir, domain_poly):
    """Draw the domain-clipped elevation map on ax and return the image."""
    data, extent = read_raster_reprojected_for_plot(
        str(domain_dir / f"{BASIN_ID}_elevation_conditioned_sfincs_grid.tif"),
        dst_bounds=domain_poly.bounds,
    )
    data = _mask_to_domain(data, extent, domain_poly)
    land, rivers, margin = _panel_context(domain_dir, domain_poly)

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
    _finish_panel(ax, margin, domain_poly)
    return im


# LC100 forest subtypes 111-116 / 121-126 collapse to "Closed forest" / "Open
# forest" in this figure's legend.
_CLOSED_FOREST_CODES = list(range(111, 117))
_OPEN_FOREST_CODES = list(range(121, 127))
_GROUP_NAMES = {110: "Closed forest", 120: "Open forest"}
_GROUP_COLORS = {110: "#1a7a1a", 120: "#5cb85c"}
_LC_NAMES_GROUPED = {**_LC_NAMES, **_GROUP_NAMES}
_LC_COLORS_GROUPED = {**_LC_COLORS, **_GROUP_COLORS}


def draw_landuse_panel(ax, domain_dir, domain_poly):
    """Draw the domain-clipped land-use map on ax; return the class codes shown."""
    data, extent = read_raster_reprojected_for_plot(
        str(domain_dir / f"{BASIN_ID}_landuse_on_grid.tif"),
        dst_bounds=domain_poly.bounds,
    )
    data = _mask_to_domain(data, extent, domain_poly)
    data[np.isin(data, _CLOSED_FOREST_CODES)] = 110
    data[np.isin(data, _OPEN_FOREST_CODES)] = 120
    land, rivers, margin = _panel_context(domain_dir, domain_poly)

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
    _finish_panel(ax, margin, domain_poly)
    return present


def _lc_legend_handles(present):
    return [
        Patch(
            color=_LC_COLORS_GROUPED.get(c, "#808080"),
            label=_LC_NAMES_GROUPED.get(c, str(c)),
        )
        for c in present
    ]


def main() -> None:
    """Run the domain figure rendering and write fig_domain_input_data.png."""
    script_parser(__doc__).parse_args()

    # results_dir is a site setting; see config/site.example.yml.
    basin_root = site_path("results_dir") / BASIN_ID
    out_dir = basin_root / "runs"
    domain_dir = basin_root / "preprocessing_inputs" / "domain"
    domain_poly = _read_domain_poly(domain_dir)

    # standalone elevation
    fig, ax = plt.subplots(figsize=(8, 7))
    im = draw_elevation_panel(ax, domain_dir, domain_poly)
    cbar = fig.colorbar(im, ax=ax, extend="max", fraction=0.03, pad=0.04)
    cbar.set_label("Elevation (m)")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(f"Elevation -- basin {BASIN_ID}")
    fig.tight_layout()
    # out_path = out_dir / "fig_elevation_map.png"
    # fig.savefig(out_path, dpi=150, bbox_inches="tight")
    # plt.close(fig)
    # print(f"Wrote {out_path}")

    # standalone land use
    fig, ax = plt.subplots(figsize=(9, 7))
    present = draw_landuse_panel(ax, domain_dir, domain_poly)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(f"Land use (Copernicus LC100, 2019) -- basin {BASIN_ID}")
    ax.legend(
        handles=_lc_legend_handles(present),
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
    )
    fig.tight_layout()
    # out_path = out_dir / "fig_landuse_map.png"
    # fig.savefig(out_path, dpi=150, bbox_inches="tight")
    # plt.close(fig)
    # print(f"Wrote {out_path}")

    # combined elevation | land use
    # Colorbar in its own gridspec column: fig.colorbar(ax=ax1) would shrink ax1
    # relative to ax2.
    fig = plt.figure(figsize=(15, 7), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.05, 1], wspace=0.08)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 2])

    # An inset, because constrained_layout stretches a plain subplot to full height.
    _cbar_cell = fig.add_subplot(gs[0, 1])
    _cbar_cell.axis("off")
    cax = _cbar_cell.inset_axes([0.3, 0.15, 0.45, 0.7])

    im = draw_elevation_panel(ax1, domain_dir, domain_poly)
    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label("Elevation (m)")
    ax1.set_xlabel("Longitude (°)")
    ax1.set_ylabel("Latitude (°)")
    # ax1.set_title("Elevation", size=13)

    present = draw_landuse_panel(ax2, domain_dir, domain_poly)
    ax2.set_xlabel("Longitude (°)")
    ax2.set_ylabel("")
    ax2.set_yticks([])
    # ax2.set_title("Land use (Copernicus LC100)", size=13)
    ax2.legend(
        handles=_lc_legend_handles(present),
        loc="lower right",
        fontsize=9,
        framealpha=0.9,
    )

    out_path = out_dir / "fig_domain_input_data.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
