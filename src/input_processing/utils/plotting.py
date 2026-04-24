from shapely.geometry import Polygon
import matplotlib.colors as mcolors
import geopandas as gpd
from shapely.geometry import MultiLineString, LineString
from geopandas import GeoDataFrame, GeoSeries
from src.input_processing.config.loader import config
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from src.input_processing.utils.util_unify_typing_and_schema import (
    CRS_STANDARD,
    BASIN_COL,
)
from pathlib import Path

from matplotlib.axes import Axes


def create_bounds_around_delta(ax, polygon: Polygon):
    # poly = polygon_gdf['geometry'].values[0]
    minx, miny, maxx, maxy = polygon.bounds
    padding = 0.1  # 5% of extent

    dx = (maxx - minx) * padding
    dy = (maxy - miny) * padding

    xmin = minx - dx
    xmax = maxx + dx
    ymin = miny - dy
    ymax = maxy + dy

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def annotate_below_plot(fig, ax, info: dict):
    """Add dictionary contents as annotation below the current plot."""
    text = "\n".join(f"{k}: {v}" for k, v in info.items())

    fig.text(
        x=0.98,
        y=0.98,  # below the axes (negative = below)
        s=text,
        ha="right",
        va="top",
        fontsize=9,
        transform=ax.transAxes,  # relative to axes, not figure
        bbox=dict(boxstyle="round", facecolor="lightgrey", alpha=0.5),
    )
    fig.tight_layout()


def get_esa_cmap():
    esa_colors = {
        10: (0, 100, 0),  # tree cover
        20: (255, 187, 34),  # shrubland       ← was (0, 160, 0)
        30: (255, 255, 76),  # grassland       ← was (170, 240, 0)
        40: (240, 150, 255),  # cropland        ← was (255, 255, 0)
        50: (250, 0, 0),  # built-up        ← was (255, 0, 0)
        60: (180, 180, 180),  # bare            ← was (190, 140, 90)
        70: (240, 240, 240),  # snow/ice        ← was (255, 255, 255)
        80: (0, 100, 200),  # water           ← was (0, 0, 255)
        90: (0, 150, 160),  # wetland         ← was (0, 200, 255)
        95: (0, 207, 117),  # mangroves       ← was (0, 150, 160)
        100: (250, 230, 160),  # moss/lichen     ← was (200, 200, 200)
        200: (0, 0, 130),  # open sea        (unchanged)
    }
    esa_colors = {k: tuple(v_i / 255 for v_i in v) for k, v in esa_colors.items()}
    bounds = list(esa_colors.keys())
    bounds_edges = bounds + [bounds[-1] + 1]  # need n+1 edges for n colors

    color_list = [esa_colors[b] for b in bounds]
    cmap_esa = mcolors.ListedColormap(color_list)
    norm = mcolors.BoundaryNorm(bounds_edges, cmap_esa.N)
    return cmap_esa, norm


def _make_patch(color: str, label: str, **kwargs) -> mpatches.Patch:
    return mpatches.Patch(color=color, label=label, **kwargs)


def _make_line(color: str, label: str, **kwargs) -> mlines.Line2D:
    return mlines.Line2D([], [], color=color, label=label, **kwargs)


def _make_marker(color: str, label: str, **kwargs) -> mlines.Line2D:
    return mlines.Line2D(
        [], [], color="none", marker="o", markerfacecolor=color, label=label, **kwargs
    )


def plot_model_domain(
    delta_polygon: Polygon,
    river_basins_gpd: GeoDataFrame,
    relevant_basins: GeoDataFrame,
    relevant_rivers_gpd: GeoDataFrame,
    rivers_gpd: GeoDataFrame,
    domain_boundary: LineString | MultiLineString,
    inland_boundary: GeoSeries,
    glofas_p: GeoSeries,
    gdf_all_sources: GeoDataFrame,
    gdf_unique_sources: GeoDataFrame,
    identifier: str = config["Testcase"]["id_delta1"],
) -> None:
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 8), sharex=True, sharey=True)
    ax0, ax1, ax2, ax3 = ax.flatten()

    # Subplot 0: basins overview
    river_basins_gpd.plot(ax=ax0, color="lightgrey")
    gpd.GeoSeries([relevant_basins.union_all()]).plot(ax=ax0, color="green")
    gpd.GeoSeries([delta_polygon]).plot(
        ax=ax0, facecolor="none", edgecolor="orange", linewidth=2
    )
    ax0.set_title("1 - Identifying relevant basins")
    handles = [
        _make_patch("lightgrey", "other basins"),
        _make_patch("green", "relevant basins"),
        _make_patch("orange", "delta domain"),
    ]
    create_bounds_around_delta(ax0, delta_polygon)

    # Subplot 1: SWORD river alignment
    river_basins_gpd.plot(ax=ax1, color="lightgrey")
    relevant_basins.plot(ax=ax1, color="green")
    gpd.GeoSeries([delta_polygon]).plot(
        ax=ax1, facecolor="none", edgecolor="orange", linewidth=2
    )
    rivers_gpd.plot(ax=ax1, color="darkgrey")
    relevant_rivers_gpd.plot(ax=ax1, color="blue", linewidth=2)
    ax1.set_title("2 - Relevant river stretches in domain and basins")
    handles += [
        _make_line("darkgrey", "all rivers"),
        _make_line("blue", "relevant rivers", linewidth=2),
    ]
    # Subplot 2: active boundaries
    river_basins_gpd.plot(ax=ax2, color="lightgrey")
    # gpd.GeoSeries([relevant_basins.union_all().boundary]).plot(ax=ax2, color='grey')
    gpd.GeoSeries([domain_boundary]).plot(ax=ax2, color="darkgrey")
    gpd.GeoSeries([inland_boundary]).plot(ax=ax2, color="red")
    glofas_p.plot(ax=ax2, color="grey")

    handles += [
        _make_line("darkgrey", "domain boundary"),
        _make_line("red", "inland domain boundary"),
        _make_marker("grey", "GloFAS cells (Q input)", markersize=10),
    ]
    ax2.set_title("3 - Relevant active boundary")

    # Subplot 3: discharge sources
    river_basins_gpd.plot(ax=ax3, color="lightgrey")
    gpd.GeoSeries([relevant_basins.union_all().boundary]).plot(ax=ax3, color="grey")
    gpd.GeoSeries([inland_boundary]).plot(ax=ax3, color="red")
    relevant_rivers_gpd.plot(ax=ax3, color="blue", linewidth=2)
    gdf_all_sources.plot(ax=ax3, color="black", markersize=20, zorder=6)
    gdf_unique_sources.plot(ax=ax3, color="red", markersize=20, zorder=6)

    ax3.set_title("4 - Identified river discharge sources")
    handles += [
        _make_marker(
            "black", f"all possible Q sources (n={len(gdf_all_sources)})", markersize=10
        ),
        _make_marker(
            "red",
            f"automatically selected sources (n={len(gdf_unique_sources)})",
            markersize=10,
        ),
    ]
    ax3.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5))

    for ax in ax.flatten():
        create_bounds_around_delta(ax, delta_polygon)
    plt.tight_layout()
    plt.savefig(f"{config['filepaths']['delta_domain_test']}_{identifier}.png", dpi=300)


def plot_river_locations(
    glofas_points: GeoDataFrame,
    domain_boundary: MultiLineString | LineString,
    rivers: GeoDataFrame,
    delta_edmonds: GeoDataFrame,
    basin_polygons: GeoDataFrame,
    basin_polygons_domain: GeoDataFrame,
    all_rivers: GeoDataFrame,
    debugging: bool,
) -> None:
    """
    Save a two-panel debug figure showing why no boundary cells were found.

    Called from ``extract_cells_within_delta`` when ``cells_boundary`` is empty.
    Reads the full delta and basin files from disk to provide spatial context.

    Left panel:  General spatial set-up — all basins, the relevant basin,
                 the Edmonds delta polygon, and the river network.
    Right panel: GloFAS cells, the derived domain boundary used for cell
                 detection, and the basin polygons for context.

    Args:
        glofas_points:    GeoDataFrame of all above-threshold GloFAS cells
                          (output of ``_filter_cells_by_threshold``).
        domain_boundary:  The inland boundary geometry used for cell detection.
        rivers:           River network GeoDataFrame clipped to the basin,
                          must contain column ``BASIN_COL``.

    Raises:
        ValueError: If *rivers* is empty (cannot determine delta ID).
    """
    if rivers.empty:
        raise ValueError(
            "[debugging_plot_river_locations] `rivers` is empty — cannot determine "
            "delta ID for debug plot."
        )

    delta_id: int | str = rivers[BASIN_COL].iloc[0]
    if "delta_name_x" in delta_edmonds.columns:
        figure_title = f"{delta_edmonds.delta_name_x.values[0]} delta"
    else:
        figure_title = f"Delta {str(delta_id)}"

    # --- Figure ---
    fig, axes_arr = plt.subplots(ncols=2, nrows=1, figsize=(12, 5), sharey=True)
    ax_left: Axes = axes_arr[0]
    ax_right: Axes = axes_arr[1]
    handles: list[mpatches.Patch | mlines.Line2D] = []

    # Left panel: overview
    basin_polygons.plot(ax=ax_left, color="lightgrey")
    basin_polygons_domain.plot(ax=ax_left, color="green")
    delta_edmonds.plot(ax=ax_left, facecolor="none", edgecolor="orange", linewidth=2)
    all_rivers.plot(ax=ax_left, color="darkgrey", linewidth=2)
    rivers.plot(ax=ax_left, color="blue", linewidth=2)
    ax_left.set_title("1 - The general inputs")
    handles += [
        _make_patch("lightgrey", "other basins"),
        _make_patch("green", "relevant basins"),
        _make_patch("orange", "delta domain"),
        _make_line("darkgrey", "all rivers", linewidth=2),
        _make_line("blue", "relevant rivers", linewidth=2),
    ]

    # Right panel: derived boundary + GloFAS cells
    basin_polygons.plot(ax=ax_right, color="lightgrey")
    glofas_points.plot(ax=ax_right, cmap="Purples", alpha=0.4, zorder=2, legend=False)
    GeoSeries([domain_boundary], crs=CRS_STANDARD).plot(ax=ax_right, color="red")
    handles += [
        _make_line("red", "domain boundary for source detection", linewidth=2),
        _make_patch("purple", "GloFAS cells"),
    ]
    ax_right.set_title("2 - Derived inputs for identifying sources")
    ax_right.legend(handles=handles, loc="lower left", bbox_to_anchor=(1.05, 0))

    plt.suptitle(f"{figure_title}")

    # Zoom both panels to the relevant domain — rename loop var to avoid
    # shadowing the `axes_arr` array defined above.
    for _ax in axes_arr.flatten():
        create_bounds_around_delta(_ax, basin_polygons_domain.geometry.union_all())

    plt.tight_layout()
    if debugging:
        out_path: Path = Path(
            f"{config['filepaths']['debug_river_sources']}_{delta_id}.png"
        )
        print(f"[DEBUG] Saved debug plot → {out_path}")
    else:
        out_path: Path = Path(
            f"{config['filepaths']['river_sources_plots']}_{delta_id}.png"
        )
        print(f"Saved complete plot → {out_path}")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)  # prevent figure accumulation in long batch runs
