"""Plotting utilities for SFINCS model domain visualisation and figure export."""

from __future__ import annotations

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from geopandas import GeoDataFrame
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from shapely.geometry.base import BaseGeometry

_PROJ = ccrs.PlateCarree()


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


def save_fig(fig: Figure, figs_dir_name: str) -> None:
    """Save *fig* as a PNG at ``<figs_dir_name>.png``.

    Args:
        fig: The matplotlib figure to save.
        figs_dir_name: Output path stem (without extension). The file is
            written to ``<figs_dir_name>.png``.
    """
    path = figs_dir_name + ".png"
    fig.savefig(path, dpi=225, bbox_inches="tight")


def plot_model_domain(
    delta_domain: GeoDataFrame,
    catalog: DataCatalog,
    figs_dir: str,
) -> None:
    """Plot the model domain basins and OSM water features on a satellite basemap.

    Args:
        delta_domain: GeoDataFrame of basin polygons for the delta domain.
        catalog: HydroMT DataCatalog used to retrieve the OSM water layer.
        figs_dir: Directory path for the output figure. The file is saved as
            ``<figs_dir>/01_model_domain.png``.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = plt.subplot(projection=_PROJ)
    #
    # tiler = cimgt.QuadtreeTiles()
    # ax.add_image(tiler, 10)

    delta_domain.geometry.boundary.plot(
        ax=ax, label="Individual basins", lw=1, color="yellow", alpha=0.5
    )

    global_water = catalog.get_geodataframe("osm_water", geom=delta_domain)
    if global_water is None:
        raise ValueError("Dataset 'osm_water' not found in DataCatalog")

    global_water.plot(ax=ax, color="red", label="Coastline", alpha=0.5)

    ax.legend()
    save_fig(fig, figs_dir + "/01_model_domain")


def plot_grid(sf: SfincsModel, figs_dir: str) -> None:
    """Plot the SFINCS model grid overlaid on a satellite basemap.

    Args:
        sf: Initialised :class:`SfincsModel` instance with a grid component.
        figs_dir: Directory path for the output figure. The file is saved as
            ``<figs_dir>/02_grid.png``.
    """
    grid_comp = sf.components["grid"]
    to_gdf = getattr(grid_comp, "to_gdf", None)
    if to_gdf is None:
        raise AttributeError("Grid component does not support to_gdf()")

    grid_gdf = to_gdf()

    fig, ax = sf.plot_basemap(plot_region=True, bmap="sat")
    grid_gdf.plot(ax=ax)

    save_fig(fig, figs_dir + "/02_grid")
