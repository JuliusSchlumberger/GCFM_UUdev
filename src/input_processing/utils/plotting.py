"""Plotting utilities for delta domain visualisation and debugging.

Provides helper functions for creating legend handles, zooming axes to a
delta extent, and producing two types of multi-panel figures:

- :func:`plot_model_domain`: four-panel overview of the full domain-building
  pipeline (basins, rivers, boundaries, discharge sources).
- :func:`plot_river_locations`: two-panel debug or summary figure showing
  GloFAS cells against the derived inland boundary.

Example:
    >>> from src.input_processing.utils.plotting import plot_river_locations
    >>> plot_river_locations(
    ...     glofas_points, domain_boundary, rivers,
    ...     delta_edmonds, basin_polygons, basin_polygons_domain,
    ...     all_rivers, debugging=True
    ... )
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import geopandas as gpd
from typing import Any
from geopandas import GeoDataFrame, GeoSeries
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from shapely.geometry import MultiLineString, LineString, Polygon
from shapely.geometry.base import BaseGeometry

from src.input_processing.config.loader import config
from src.input_processing.utils.util_unify_typing_and_schema import (
    CRS_STANDARD,
    BASIN_COL,
)


# ---------------------------------------------------------------------------
# Axes helpers
# ---------------------------------------------------------------------------


def create_bounds_around_delta(ax: Axes, polygon: BaseGeometry) -> None:
    """Zoom *ax* to the bounding box of *polygon* with a small padding margin.

    Computes a 10 % padding on each side of the polygon extent and applies it
    to both axes limits so the polygon is not clipped at the panel edge.

    Args:
        ax: The matplotlib axes to zoom.
        polygon: Any shapely geometry whose ``.bounds`` will be used as the
            reference extent. Typically a delta polygon or basin union.

    Example:
        >>> fig, ax = plt.subplots()
        >>> create_bounds_around_delta(ax, delta_polygon)
    """
    minx, miny, maxx, maxy = polygon.bounds
    padding: float = 0.1

    dx: float = (maxx - minx) * padding
    dy: float = (maxy - miny) * padding

    ax.set_xlim(minx - dx, maxx + dx)
    ax.set_ylim(miny - dy, maxy + dy)


def annotate_below_plot(
    fig: Figure,
    ax: Axes,
    info: dict[str, int | float | str],
) -> None:
    """Add dictionary contents as a text annotation in the upper-right corner of the plot.

    Formats each key-value pair as ``"key: value"`` on a separate line and
    places the block inside a rounded light-grey box anchored to the top-right
    of the axes.

    Args:
        fig: The matplotlib figure that owns *ax*. Used to call
            ``fig.tight_layout()`` after placing the annotation.
        ax: The axes to annotate. The text is placed relative to this axes
            using ``ax.transAxes`` coordinates.
        info: Dictionary of statistic labels to values. Keys are displayed
            as-is; values are converted to strings via ``f"{k}: {v}"``.

    Example:
        >>> fig, ax = plt.subplots()
        >>> annotate_below_plot(fig, ax, {"area [km2]": 1250, "distance [km]": 42})
    """
    text: str = "\n".join(f"{k}: {v}" for k, v in info.items())
    fig.text(
        x=0.98,
        y=0.98,
        s=text,
        ha="right",
        va="top",
        fontsize=9,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="lightgrey", alpha=0.5),
    )
    fig.tight_layout()


# ---------------------------------------------------------------------------
# ESA colormap
# ---------------------------------------------------------------------------


def get_esa_cmap() -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
    """Build a colormap and normalisation for the ESA World Cover classification.

    Maps each ESA land-cover class code to its standard display colour. The
    returned colormap and norm are intended for use with ``imshow`` or
    ``DataArray.plot.imshow``.

    Returns:
        A tuple of ``(cmap_esa, norm)`` where:

        - *cmap_esa*: A :class:`~matplotlib.colors.ListedColormap` with one
          colour per ESA class.
        - *norm*: A :class:`~matplotlib.colors.BoundaryNorm` mapping class
          codes to colormap indices.

    Example:
        >>> cmap, norm = get_esa_cmap()
        >>> lu_raster.plot.imshow(cmap=cmap, norm=norm)
    """
    esa_colors: dict[int, tuple[int, int, int]] = {
        10: (0, 100, 0),  # tree cover
        20: (255, 187, 34),  # shrubland
        30: (255, 255, 76),  # grassland
        40: (240, 150, 255),  # cropland
        50: (250, 0, 0),  # built-up
        60: (180, 180, 180),  # bare
        70: (240, 240, 240),  # snow/ice
        80: (0, 100, 200),  # water
        90: (0, 150, 160),  # wetland
        95: (0, 207, 117),  # mangroves
        100: (250, 230, 160),  # moss/lichen
        200: (0, 0, 130),  # open sea
    }
    normalised: dict[int, tuple[float, ...]] = {
        k: tuple(v_i / 255 for v_i in v) for k, v in esa_colors.items()
    }
    bounds: list[int] = list(normalised.keys())
    bounds_edges: list[int] = bounds + [bounds[-1] + 1]

    cmap_esa: mcolors.ListedColormap = mcolors.ListedColormap(
        [normalised[b] for b in bounds]
    )
    norm: mcolors.BoundaryNorm = mcolors.BoundaryNorm(bounds_edges, cmap_esa.N)
    return cmap_esa, norm


# ---------------------------------------------------------------------------
# Legend handle factories
# ---------------------------------------------------------------------------


def _make_patch(color: str, label: str, **kwargs: Any) -> mpatches.Patch:
    """Return a filled rectangle legend handle.

    Args:
        color: Fill colour accepted by matplotlib (name, hex, or RGB tuple).
        label: Text label shown in the legend.
        **kwargs: Additional keyword arguments forwarded to
            :class:`~matplotlib.patches.Patch`.

    Returns:
        A :class:`~matplotlib.patches.Patch` configured with the given colour
        and label.

    Example:
        >>> handle = _make_patch("lightgrey", "other basins")
    """
    return mpatches.Patch(color=color, label=label, **kwargs)


def _make_line(color: str, label: str, **kwargs: Any) -> mlines.Line2D:
    """Return a solid line legend handle.

    Args:
        color: Line colour accepted by matplotlib.
        label: Text label shown in the legend.
        **kwargs: Additional keyword arguments forwarded to
            :class:`~matplotlib.lines.Line2D`, e.g. ``linewidth``.

    Returns:
        A :class:`~matplotlib.lines.Line2D` with no data points, suitable
        for use as a legend proxy.

    Example:
        >>> handle = _make_line("blue", "relevant rivers", linewidth=2)
    """
    return mlines.Line2D([], [], color=color, label=label, **kwargs)


def _make_marker(color: str, label: str, **kwargs: Any) -> mlines.Line2D:
    """Return a circular marker legend handle.

    Args:
        color: Marker face colour accepted by matplotlib.
        label: Text label shown in the legend.
        **kwargs: Additional keyword arguments forwarded to
            :class:`~matplotlib.lines.Line2D`, e.g. ``markersize``.

    Returns:
        A :class:`~matplotlib.lines.Line2D` styled as a filled circle with
        no connecting line, suitable for use as a legend proxy for point data.

    Example:
        >>> handle = _make_marker("red", "selected sources", markersize=10)
    """
    return mlines.Line2D(
        [],
        [],
        color="none",
        marker="o",
        markerfacecolor=color,
        label=label,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Multi-panel domain overview figure
# ---------------------------------------------------------------------------


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
    """Save a four-panel overview figure of the full domain-building pipeline.

    Each panel corresponds to one stage of the processing pipeline:

    1. **Identifying relevant basins** — all basin polygons with the relevant
       subset highlighted and the Edmonds delta outline overlaid.
    2. **River stretches** — SWORD river network with relevant reaches in blue.
    3. **Active boundaries** — domain boundary, inland boundary, and GloFAS
       discharge cells.
    4. **Discharge sources** — all candidate and automatically selected river
       source points.

    The figure is saved to ``config['filepaths']['delta_domain_test']``.

    Args:
        delta_polygon: The Edmonds et al. (2020) delta polygon used as the
            reference extent for all panels.
        river_basins_gpd: Full set of basin polygons drawn as background.
        relevant_basins: Basin polygons that intersect the delta domain.
        relevant_rivers_gpd: River segments clipped to the relevant basins.
        rivers_gpd: Full river network clipped to the local bounding box.
        domain_boundary: Outer domain boundary geometry.
        inland_boundary: Derived inland boundary GeoSeries used for source
            detection.
        glofas_p: GeoSeries of buffered GloFAS discharge cells.
        gdf_all_sources: GeoDataFrame of all candidate discharge source points.
        gdf_unique_sources: GeoDataFrame of the automatically selected
            most-downstream source points.
        identifier: Delta identifier string used in the output filename.
            Defaults to ``config['Testcase']['id_delta1']``.

    Example:
        >>> plot_model_domain(
        ...     delta_polygon, river_basins_gpd, relevant_basins,
        ...     relevant_rivers_gpd, rivers_gpd, domain_boundary,
        ...     inland_boundary, glofas_p, gdf_all_sources,
        ...     gdf_unique_sources, identifier="delta_42"
        ... )
    """
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 8), sharex=True, sharey=True)
    ax0, ax1, ax2, ax3 = ax.flatten()

    # --- Panel 1: basin overview ---
    river_basins_gpd.plot(ax=ax0, color="lightgrey")
    gpd.GeoSeries([relevant_basins.union_all()]).plot(ax=ax0, color="green")
    gpd.GeoSeries([delta_polygon]).plot(
        ax=ax0, facecolor="none", edgecolor="orange", linewidth=2
    )
    ax0.set_title("1 - Identifying relevant basins")
    create_bounds_around_delta(ax0, delta_polygon)

    handles: list[mpatches.Patch | mlines.Line2D] = [
        _make_patch("lightgrey", "other basins"),
        _make_patch("green", "relevant basins"),
        _make_patch("orange", "delta domain"),
    ]

    # --- Panel 2: river alignment ---
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

    # --- Panel 3: active boundaries ---
    river_basins_gpd.plot(ax=ax2, color="lightgrey")
    gpd.GeoSeries([domain_boundary]).plot(ax=ax2, color="darkgrey")
    gpd.GeoSeries([inland_boundary]).plot(ax=ax2, color="red")
    glofas_p.plot(ax=ax2, color="grey")
    ax2.set_title("3 - Relevant active boundary")
    handles += [
        _make_line("darkgrey", "domain boundary"),
        _make_line("red", "inland domain boundary"),
        _make_marker("grey", "GloFAS cells (Q input)", markersize=10),
    ]

    # --- Panel 4: discharge sources ---
    river_basins_gpd.plot(ax=ax3, color="lightgrey")
    gpd.GeoSeries([relevant_basins.union_all().boundary]).plot(ax=ax3, color="grey")
    gpd.GeoSeries([inland_boundary]).plot(ax=ax3, color="red")
    relevant_rivers_gpd.plot(ax=ax3, color="blue", linewidth=2)
    gdf_all_sources.plot(ax=ax3, color="black", markersize=20, zorder=6)
    gdf_unique_sources.plot(ax=ax3, color="red", markersize=20, zorder=6)
    ax3.set_title("4 - Identified river discharge sources")
    handles += [
        _make_marker(
            "black",
            f"all possible Q sources (n={len(gdf_all_sources)})",
            markersize=10,
        ),
        _make_marker(
            "red",
            f"automatically selected sources (n={len(gdf_unique_sources)})",
            markersize=10,
        ),
    ]
    ax3.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5))

    for _ax in ax.flatten():
        create_bounds_around_delta(_ax, delta_polygon)

    plt.tight_layout()
    plt.savefig(f"{config['filepaths']['delta_domain_test']}_{identifier}.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Two-panel river source debug / summary figure
# ---------------------------------------------------------------------------


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
    """Save a two-panel figure showing GloFAS cells against the inland boundary.

    Produces either a debug figure (when ``debugging=True``) saved to the
    debug output path, or a standard summary figure saved to the regular plots
    path. The left panel shows the general spatial context; the right panel
    shows the derived boundary and GloFAS cells.

    Left panel: all basin polygons, the relevant basin highlighted, the
        Edmonds delta outline, and the full and clipped river networks.
    Right panel: GloFAS discharge cells, the derived inland boundary, and
        basin polygons for context.

    Args:
        glofas_points: GeoDataFrame of all above-threshold GloFAS cells, i.e.
            the output of ``_filter_cells_by_threshold``. Must not be empty —
            guard the call site with an emptiness check before calling this
            function.
        domain_boundary: The inland boundary geometry used for GloFAS cell
            detection. Drawn in red on the right panel.
        rivers: River network GeoDataFrame clipped to the basin. Must contain
            the ``BASIN_COL`` column so the delta ID can be read from it.
        delta_edmonds: Single-row GeoDataFrame with the Edmonds delta polygon.
            Used for the orange outline on the left panel and, if
            ``delta_name_x`` is present, for the figure title.
        basin_polygons: Full set of basin polygons drawn as the grey background
            on both panels.
        basin_polygons_domain: Basin polygons for the current delta only. Used
            to zoom both panels and drawn in green on the left panel.
        all_rivers: Full river network for the region, drawn in dark grey on
            the left panel for spatial context.
        debugging: If True, the figure is saved to the debug output path and
            a ``[DEBUG]`` prefix is printed. If False, it is saved to the
            standard plots path.

    Raises:
        ValueError: If *rivers* is empty, since the delta ID cannot be
            determined without at least one river row.

    Example:
        >>> plot_river_locations(
        ...     glofas_points, domain_boundary, rivers,
        ...     delta_edmonds, basin_polygons, basin_polygons_domain,
        ...     all_rivers, debugging=False
        ... )
    """
    if rivers.empty:
        raise ValueError(
            "[plot_river_locations] `rivers` is empty — cannot determine "
            "delta ID for the figure."
        )

    delta_id: int | str = rivers[BASIN_COL].iloc[0]
    figure_title: str = (
        f"{delta_edmonds.delta_name_x.values[0]} delta"
        if "delta_name_x" in delta_edmonds.columns
        else f"Delta {delta_id}"
    )

    fig, axes_arr = plt.subplots(ncols=2, nrows=1, figsize=(12, 5), sharey=True)
    ax_left: Axes = axes_arr[0]
    ax_right: Axes = axes_arr[1]
    handles: list[mpatches.Patch | mlines.Line2D] = []

    # --- Left panel: spatial context ---
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

    # --- Right panel: GloFAS cells and boundary ---
    basin_polygons.plot(ax=ax_right, color="lightgrey")
    glofas_points.plot(ax=ax_right, color="purple", alpha=0.4, zorder=2)
    GeoSeries([domain_boundary], crs=CRS_STANDARD).plot(ax=ax_right, color="red")
    handles += [
        _make_line("red", "domain boundary for source detection", linewidth=2),
        _make_patch("purple", "GloFAS cells"),
    ]
    ax_right.set_title("2 - Derived inputs for identifying sources")
    ax_right.legend(handles=handles, loc="lower left", bbox_to_anchor=(1.05, 0))

    plt.suptitle(figure_title)

    for _ax in axes_arr.flatten():
        create_bounds_around_delta(_ax, basin_polygons_domain.geometry.union_all())

    plt.tight_layout()

    out_path: Path = Path(
        f"{config['filepaths']['debug_river_sources']}_{delta_id}.png"
        if debugging
        else f"{config['filepaths']['river_sources_plots']}_{delta_id}.png"
    )
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    prefix: str = "[DEBUG]" if debugging else ""
    print(f"{prefix} Saved plot → {out_path}".strip())
