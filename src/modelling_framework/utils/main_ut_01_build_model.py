"""Model configuration, initialisation, and grid creation utilities for SFINCS."""

from __future__ import annotations

from logging import Logger
from pathlib import Path

import hydromt
from geopandas import GeoDataFrame
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel

from src.modelling_framework.utils.plotting import plot_model_domain
from src.modelling_framework.utils.utils import initialize_logger


def configure_model(
    delta_basin_id: int,
    data_libs: list[str],
    root_path: str,
    debug_plotting: bool,
) -> tuple[DataCatalog, GeoDataFrame, Logger]:
    """Configure the HydroMT data catalog and define the delta model domain.

    Loads the data catalog, filters the basin delta polygons to the requested
    basin ID, optionally saves a diagnostic plot, and initialises the logger.

    Args:
        delta_basin_id: Numeric ID of the delta basin to configure.
        data_libs: List of HydroMT data library paths or catalogue keys.
        root_path: Root directory for model output. A subdirectory named
            after *delta_basin_id* is created inside this path.
        debug_plotting: If True, saves a domain overview figure to
            ``<root_path>/<delta_basin_id>/build/``.

    Returns:
        A tuple of ``(catalog, delta_domain, logger)`` where *catalog* is the
        configured :class:`~hydromt.DataCatalog`, *delta_domain* is the
        single-row :class:`~geopandas.GeoDataFrame` for the requested basin,
        and *logger* is the initialised HydroMT :class:`~logging.Logger`.
    """
    catalog = hydromt.DataCatalog(data_libs=data_libs)
    root_path = root_path + f"/{delta_basin_id}"

    deltas = catalog.get_geodataframe("basin_deltas")

    if deltas is None:
        raise ValueError("Dataset 'basin_deltas' not found in DataCatalog")

    delta_domain = deltas.loc[deltas["BasinID2"] == delta_basin_id]

    if not isinstance(delta_domain, GeoDataFrame):
        raise TypeError("Expected GeoDataFrame after filtering basin_deltas")

    if delta_domain.empty:
        raise ValueError(f"No delta found for BasinID2={delta_basin_id}")

    if debug_plotting:
        figs_dir = Path(root_path) / "build"
        figs_dir.mkdir(parents=True, exist_ok=True)
        plot_model_domain(delta_domain, catalog, str(figs_dir))

    logger = initialize_logger(delta_basin_id, root_path)

    return catalog, delta_domain, logger


def initialize_model(data_libs: list[str], root_path: str) -> SfincsModel:
    """Initialise a SFINCS model in write mode.

    Args:
        data_libs: List of HydroMT data library paths or catalogue keys.
        root_path: Root directory passed to :class:`~hydromt_sfincs.SfincsModel`.

    Returns:
        A :class:`~hydromt_sfincs.SfincsModel` instance opened in
        overwrite mode (``"w+"``).
    """
    return SfincsModel(
        data_libs=data_libs,
        root=root_path,
        mode="w+",
        write_gis=True,
    )


def create_grid(
    sf: SfincsModel,
    delta_domain: GeoDataFrame,
    grid_resolution: int,
    debug_plotting: bool,
    logger: Logger,
) -> None:
    """Create a regular SFINCS grid from the delta domain geometry.

    Builds a non-rotated grid in the nearest UTM zone at *grid_resolution*
    metres, then logs the resulting grid dimensions.

    Args:
        sf: Initialised :class:`~hydromt_sfincs.SfincsModel` instance.
        delta_domain: Single-row GeoDataFrame defining the model domain.
        grid_resolution: Grid cell size in metres.
        debug_plotting: Reserved for future use; currently unused.
        logger: Logger instance used to report the grid dimensions.
    """
    sf.grid.create_from_region(
        region={"geom": delta_domain},
        res=grid_resolution,
        rotated=False,
        crs="utm",
    )

    grid_comp = sf.components["grid"]
    ds = getattr(grid_comp, "data")
    x, y = ds.dims["x"], ds.dims["y"]

    logger.info("Grid of %d cells created (x=%d, y=%d)", x * y, x, y)
