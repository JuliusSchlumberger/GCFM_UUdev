"""Visual alignment check between GloFAS v4.0 reanalysis and river datasets.

Investigates the spatial alignment of the GloFAS v4.0 reanalysis discharge
dataset against other river data products (MERIT, SWORD, Lin et al., GRIT)
for a selected delta test case.

References:
    Grimaldi et al. (2022): River discharge and related historical data from
    the Global Flood Awareness System, v4.0. European Commission, Joint
    Research Centre (JRC). DOI: 10.24381/cds.a4fdd6b9

Example:
    >>> from src import plot_glofas
    >>> plot_glofas_lin("id_delta1")
"""

from __future__ import annotations

import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame

from src import config


def plot_glofas(testcase_id: str) -> None:
    """Plot mean GloFAS discharge alongside river network datasets for one delta.

    Loads the delta domain polygon for *testcase_id*, clips the GloFAS
    reanalysis and MERIT datasets to that extent, and produces a single-panel
    figure overlaying mean discharge with the Lin et al. and GRIT river
    networks and the delta boundary. The figure is saved to the configured
    validation plots directory.

    Args:
        testcase_id: Key into ``config['Testcase']`` that maps to the numeric
            delta identifier, e.g. ``"id_delta1"``.

    Returns:
        None. The figure is saved to
        ``config['filepaths']['glofas_lin_plot']`` and displayed interactively.

    Raises:
        KeyError: If *testcase_id* is not present in ``config['Testcase']``.
        FileNotFoundError: If any of the configured input data files do not
            exist at the expected paths.

    Example:
        >>> plot_glofas_lin("id_delta1")
        -3.5 4.2 2.1 8.7
    """
    # --- Load and clip delta domain ---
    delta_domain: GeoDataFrame = gpd.read_file(
        "../src/input_processing/data/4_delta_polygons.geojson"
    )
    delta_domain = delta_domain[
        delta_domain["BasinID2"] == config["Testcase"][testcase_id]
    ].to_crs(epsg=config["CRS"]["standard"])

    minx: float
    miny: float
    maxx: float
    maxy: float
    minx, miny, maxx, maxy = delta_domain.total_bounds
    print(minx, miny, maxx, maxy)

    # --- Load and clip GloFAS (daily max discharge, one month globally) ---
    glofas_nc: xr.Dataset = xr.open_dataset(
        "../src/input_processing/data/GloFAS_rivQ_2018_12_test.nc"
    )
    glofas_clip: xr.Dataset = glofas_nc.sel(
        longitude=slice(minx, maxx),
        latitude=slice(maxy, miny),  # latitude is stored north→south
    )
    glofas_mean: xr.DataArray = glofas_clip.dis24.mean(dim="valid_time")
    print(glofas_mean)

    # --- Load and clip MERIT-BASINS ---
    merit_nc: xr.Dataset = xr.open_dataset(
        "../src/input_processing/data/merit.nc"
    )
    merit_clip: xr.Dataset = merit_nc.sel(
        lon=slice(minx, maxx),
        lat=slice(miny, maxy),
    )

    # --- Load river datasets clipped to delta extent ---
    rivers_lin: GeoDataFrame = gpd.read_file(
        "../src/input_processing/data/rivers_lin.gpkg", mask=delta_domain
    ).to_crs(epsg=config["CRS"]["standard"])

    rivers_grit: GeoDataFrame = gpd.read_file(
        "../src/input_processing/data/GRIT/GRITv1.0_segments_NA_EPSG4326.gpkg",
        mask=delta_domain,
    ).to_crs(epsg=config["CRS"]["standard"])

    # --- Figure ---
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

    glofas_mean.plot(ax=ax, cmap="Blues", alpha=1)
    delta_domain.boundary.plot(ax=ax, color="red")
    rivers_lin.plot(ax=ax, color="green", label="Lin et al.")
    rivers_grit.plot(ax=ax, color="blue", label="GRIT v1.0")

    ax.legend()
    fig.suptitle(
        f"Visual comparison between GloFAS and different data products "
        f"for Delta {config['Testcase'][testcase_id]}"
    )

    plt.tight_layout()
    plt.savefig(
        f"../src/input_processing/validation/plots/rivers/"
        f"glofas_lin_{config['Testcase'][testcase_id]}.png",
        dpi=300,
    )
    plt.show()
    plt.close(fig)