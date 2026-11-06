"""Figure conventions of the analysis figures: panel letters, land underlay, caption files."""

from pathlib import Path

import numpy as np

LAND_COLOUR = "0.88"


def panel_letters(axes, start: int = 0, xy: tuple[float, float] = (0.01, 0.99)) -> None:
    """Label each axis (a), (b), ... in its top-left corner, in the order given."""
    for k, ax in enumerate(np.ravel(axes)):
        ax.text(
            xy[0],
            xy[1],
            f"({chr(ord('a') + start + k)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
            zorder=10,
        )


def add_land(ax, basin_id: str) -> None:
    """Draw the basin's land polygons in light grey under the map, in the model CRS."""
    import geopandas as gpd
    from _basin_inputs import basin_paths
    from src.grid import load_grid_def

    p = basin_paths(basin_id)
    land = gpd.read_file(
        p["sfincs_grid"].parent / f"{basin_id}_land_polygons.gpkg"
    ).to_crs(load_grid_def(p["sfincs_grid"])["crs"])
    land.plot(ax=ax, color=LAND_COLOUR, edgecolor="none", zorder=0)


def map_axes(ax) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def write_caption(figure_path, text: str) -> None:
    """Write the figure's description next to it, as <figure>.txt."""
    Path(figure_path).with_suffix(".txt").write_text(
        text.strip() + "\n", encoding="utf-8"
    )
