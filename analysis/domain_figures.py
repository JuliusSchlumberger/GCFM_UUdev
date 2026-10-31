"""Plot the study basin's input maps, return-period curves and forced hydrographs.

Same WGS84 style (land polygons + river network) as flood_maps.py. Uses the
SFINCS-grid-aligned rasters fed to the model. Both are clipped to the
model domain polygon: landuse_on_grid.tif has no nodata and fills its whole
rectangle with valid classes, so unclipped panels would differ in extent.

Output, in <results_dir>/<BASIN_ID>/figures/analysis/ (BASIN_ID = 2433835):
  90_domain_figures_input_data.png            elevation | land use, 1x2
  90_domain_figures_elevation_classes.png     elevation only, in 1 m classes (<0 .. >5 m)
  90_domain_figures_return_period_curves.png  river discharge | coastal water level
  90_domain_figures_forced_hydrograph.png     built forcing per scenario, 2 x SCENARIOS
"""

import json
import sys

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import shapely
import xarray as xr
from _common import REPO_ROOT, analysis_path, script_parser, site_path  # noqa: E402
from hydromt_sfincs import SfincsModel
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm, to_hex
from matplotlib.patches import Patch

sys.path.insert(0, str(REPO_ROOT / "workflow"))
from src.plots import (  # noqa: E402
    _BATHY_CMAP,
    _LC_COLORS,
    _LC_NAMES,
    read_raster_reprojected_for_plot,
)

BASIN_ID = "2433835"

# Capped well below the basin's full elevation range so the lowlands keep
# contrast; values above VMAX take the max colour (colorbar extend="max").
ELEV_VMIN, ELEV_VMAX = -10.0, 50.0

# Discrete classes for the classified elevation map: the flood-relevant lowlands
# sit within a few metres of sea level, which the continuous -10..50 m ramp
# washes out -- 1 m bins up to 5 m make those differences readable.
ELEV_CLASS_BOUNDS = [0, 1, 2, 3, 4, 5]
ELEV_CLASS_LABELS = ["< 0 m", "0–1 m", "1–2 m", "2–3 m", "3–4 m", "4–5 m", "> 5 m"]
ELEV_CLASS_COLORS = ["#2c7fb8"] + [
    to_hex(c) for c in plt.get_cmap("YlOrBr")(np.linspace(0.1, 0.95, 6))
]

# Forced-hydrograph columns, left to right.
SCENARIOS = ["coast_100_03SLR", "river_100_03SLR", "compound_100_03SLR"]
SCENARIO_LABELS = {
    "coast_100_03SLR": "Coastal flood",
    "river_100_03SLR": "River flood",
    "compound_100_03SLR": "Compound flood",
}
DESIGN_RP_YR = 100  # marked on the return-period curves

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


def draw_elevation_panel(ax, domain_dir, domain_poly, classified=False):
    """Draw the domain-clipped elevation map on ax and return the image.

    classified: ELEV_CLASS_BOUNDS classes instead of the continuous ramp.
    """
    data, extent = read_raster_reprojected_for_plot(
        str(domain_dir / f"{BASIN_ID}_elevation_conditioned_sfincs_grid.tif"),
        dst_bounds=domain_poly.bounds,
    )
    data = _mask_to_domain(data, extent, domain_poly)
    land, rivers, margin = _panel_context(domain_dir, domain_poly)

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
    """Write the input-data map, return-period curves and forced hydrographs."""
    script_parser(__doc__).parse_args()

    # results_dir is a site setting; see config/site.example.yml.
    basin_root = site_path("results_dir") / BASIN_ID
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

    out_path = analysis_path(BASIN_ID, __file__, "input_data.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    # classified elevation
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    draw_elevation_panel(ax, domain_dir, domain_poly, classified=True)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.legend(
        handles=[
            Patch(color=c, label=lbl)
            for c, lbl in zip(ELEV_CLASS_COLORS, ELEV_CLASS_LABELS)
        ],
        title="Elevation",
        loc="lower right",
        fontsize=9,
        framealpha=0.9,
    )
    out_path = analysis_path(BASIN_ID, __file__, "elevation_classes.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    plot_return_period_curves(basin_root)
    plot_forced_hydrograph(basin_root)


def plot_return_period_curves(basin_root):
    """Plot river discharge and coastal water level against return period, 1x2.

    Each panel marks the DESIGN_RP_YR event and the basin's own protection
    standard (FLOPROS, protection_levels.json), so the gap between what is
    protected and what is forced shows.
    """
    forcing_dir = basin_root / "preprocessing_inputs" / "forcing"
    protection = json.loads(
        (
            basin_root / "preprocessing_inputs" / "domain" / "protection_levels.json"
        ).read_text()
    )
    # decode_times=False: units "hours since simulation start" have no reference
    # date for xarray's CF-time decoder.
    with (
        xr.open_dataset(forcing_dir / "river_forcing.nc", decode_times=False) as rds,
        xr.open_dataset(forcing_dir / "surge_forcing.nc", decode_times=False) as sds,
    ):
        # Dominant river crossing (largest highest-RP discharge), as in
        # 07d_get_boundary_forcings.py's diagnostic plot.
        active = rds["has_glofas"].values.astype(bool)
        rp_table = rds["discharge_rp_table"].values[active]
        i_main = np.nanargmax(rp_table[:, -1])
        river_rp_yr = rds["rp_yr"].values
        surge_rp_yr = sds["rp_yr"].values
        storm_tide_table = sds["storm_tide_rp_table"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")

    in_range = river_rp_yr <= 1000
    ax1.plot(river_rp_yr[in_range], rp_table[i_main][in_range], color="#1b4965")
    ax1.set_xscale("log")
    ax1.set_xlabel("Return period (years)")
    ax1.set_ylabel("Discharge (m3/s)")
    ax1.set_title("River discharge")

    for row in storm_tide_table:
        ax2.plot(surge_rp_yr, row, marker="o", markersize=3)
    ax2.set_xscale("log")
    ax2.set_xlabel("Return period (years)")
    ax2.set_ylabel("Water level (m)")
    ax2.set_title("Coastal water level")

    panel_protection = {
        ax1: protection["river_protection_rp_yr"],
        ax2: protection["surge_protection_rp_yr"],
    }
    for ax, panel in ((ax1, "(a)"), (ax2, "(b)")):
        ax.grid(alpha=0.3)
        ax.set_xlim(0.7, 1.5e3)
        ax.axvline(
            DESIGN_RP_YR,
            color="#b44225",
            linestyle="--",
            linewidth=1,
            label=f"{DESIGN_RP_YR}-year event",
        )
        rp_protected = panel_protection[ax]
        ax.axvline(
            rp_protected,
            color="grey",
            linestyle="--",
            linewidth=1.2,
            label=f"Protection standard ({rp_protected:.0f}-year)",
        )
        # below the panel letter
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

    fig.tight_layout()
    out_path = analysis_path(BASIN_ID, __file__, "return_period_curves.png")
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_forced_hydrograph(basin_root):
    """Plot each scenario's built water-level and discharge forcing, 2 x SCENARIOS.

    Read from the scenario's sfincs.bzs/sfincs.dis, so RP, protection floor and
    SLR are resolved; the 'discharge'/'water_level' in river_forcing.nc and
    surge_forcing.nc are a scenario-independent preview at a fixed diagnostic RP.
    Shown as the mean over boundary points; y shared per row.
    """
    fig, axes = plt.subplots(
        2,
        len(SCENARIOS),
        figsize=(3.2 * len(SCENARIOS), 5.5),
        sharex=True,
        sharey="row",
        facecolor="white",
        squeeze=False,
    )

    for j, scen in enumerate(SCENARIOS):
        mod = SfincsModel(root=str(basin_root / "runs" / scen / "sfincs"), mode="r")

        wl_comp = mod.get_component("water_level")
        wl_comp.read()
        wl = wl_comp.data  # dims (time, index), var 'bzs'
        axes[0, j].plot(
            wl.time.values, wl["bzs"].mean(dim="index").values, color="#14b5dd", lw=1.8
        )

        dis_comp = mod.get_component("discharge_points")
        dis_comp.read()
        dis = dis_comp.data  # dims (time, index), var 'dis'
        axes[1, j].plot(
            dis.time.values,
            dis["dis"].mean(dim="index").values,
            color="#2411ce",
            lw=1.8,
        )

        for i in (0, 1):
            axes[i, j].grid(alpha=0.3)
            axes[i, j].spines[["top", "right"]].set_visible(False)
        axes[0, j].set_title(SCENARIO_LABELS.get(scen, scen))

    axes[0, 0].set_ylabel("Water level (m)")
    axes[1, 0].set_ylabel("Discharge (m³/s)")

    # AutoDateLocator is too dense at ~3" column width: 12-hourly, rotated ticks.
    locator = mdates.HourLocator(byhour=[0, 12])
    formatter = mdates.ConciseDateFormatter(locator)
    for ax in axes.flat:
        ax.label_outer()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis="x", labelrotation=45)
        # "2000" is the pipeline's dummy reference year: hide the offset text.
        ax.xaxis.get_offset_text().set_visible(False)

    fig.suptitle("Forcing hydrographs")
    fig.tight_layout()
    out_path = analysis_path(BASIN_ID, __file__, "forced_hydrograph.png")
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
