"""
With this script, we investigate alignment of the GloFAS v4.0 reanalysis dataset and other data products like MERIT and SWORD.



Refernces:
Grimaldi et al. (2022): River discharge and related historical data from the Global Flood Awareness System, v4.0. European Commission, Joint Research Centre (JRC). DOI: 10.24381/cds.a4fdd6b9
"""

import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from src.input_processing.config.loader import config

# Specify mask for plots

def plot_glofas_lin(testcase_id):
    delta_domain = gpd.read_file('../src/input_processing/data/4_delta_polygons.geojson')
    delta_domain = delta_domain[delta_domain['BasinID2'] == config['Testcase'][testcase_id]].to_crs(epsg=config['CRS']['standard'])
    minx, miny, maxx, maxy = delta_domain.total_bounds
    print(minx, miny, maxx, maxy)

    # 2. Load and mask relevant data products
    glofas_nc = xr.open_dataset('../src/input_processing/data/GloFAS_rivQ_2018_12_test.nc') # selected daily max discharge for one month globally
    glofas_clip = glofas_nc.sel(
        longitude=slice(minx, maxx),
        latitude=slice(maxy, miny)  # latitude is descending!
    )
    glofas_mean = glofas_clip.dis24.mean(dim="valid_time")
    print(glofas_mean)

    merit_nc = xr.open_dataset('../src/input_processing/data/merit.nc')    # MERIT-BASINS (?)
    merit_clip = merit_nc.sel(
        lon=slice(minx, maxx),
        lat=slice(miny, maxy)
    )

    rivers_sword = gpd.read_file('../src/input_processing/data/SWORD_global_unpublished.gpkg', mask=delta_domain).to_crs(epsg=config['CRS']['standard'])
    rivers_lin = gpd.read_file('../src/input_processing/data/rivers_lin.gpkg', mask=delta_domain).to_crs(epsg=config['CRS']['standard'])
    rivers_grit = gpd.read_file('../src/input_processing/data/GRIT/GRITv1.0_segments_NA_EPSG4326.gpkg', mask=delta_domain).to_crs(
        epsg=config['CRS']['standard'])


    # Create figure for overlap
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))

    glofas_mean.plot(
        ax=ax,
        cmap="Blues",
        alpha=1
    )
    delta_domain.boundary.plot(ax=ax,color='red')
    rivers_lin.plot(ax=ax, color='green', legend='Lin et al.')
    rivers_grit.plot(ax=ax, color='blue', legend='Lin et al.')
    # ax.set_title(f"Lin et al. (2019)")
    fig.suptitle(f"Visual comparison between Glofas and different data products for Delta {config['Testcase'][testcase_id]}")

    # glofas_mean.plot(
    #     ax=ax,
    #     cmap="Blues",
    #     alpha=0.6
    # )
    plt.show()
    # delta_domain.boundary.plot(ax=ax, color='red')
    # rivers_sword.plot(ax=ax, legend='SWORD (unpublished)')
    # ax.set_title(f"SWORD (unpublished)")
    #
    # glofas_mean.plot(
    #     ax=axes[2],
    #     cmap="Blues",
    #     alpha=0.6
    # )
    # delta_domain.boundary.plot(ax=axes[2], color='red')
    # rivers_Kiara.plot(ax=axes[2], legend='SWORD (Kiara)')
    # axes[2].set_title(f"SWORD (Kiara)")

    # glofas_mean.plot(
    #     ax=axes[3],
    #     cmap="Blues",
    #     alpha=0.6
    # )
    # delta_domain.boundary.plot(ax=axes[3], color='red')
    # merit_clip.plot(ax=axes[3], cmap="Greens",
    #     alpha=0.6)
    # axes[3].set_title(f"Merit")



    plt.show()
    plt.savefig(f"..src/input_processing/validation/plots/rivers/glofas_lin_{config['Testcase'][testcase_id]}.png", dpi=300)
