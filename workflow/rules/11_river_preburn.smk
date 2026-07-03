# Rule: compute per-pixel river bed anchor points for SFINCS subgrid burning.
#
# Only scheduled when river_processing.conditioning.enabled is True. The
# merge point of the two independent branches: needs BOTH 09b's
# estuarine-adjusted network (for rivdph) AND 10's conditioned elevation.
#
# Samples a point at every DEM-pixel step along each centerline (via
# src.river_network._sample_line_cells), carrying the absolute river bed
# elevation (rivbed = DEM - rivdph [m+REF]) in a 'rivbed' column -- see
# src/river_preburn.py for the actual per-pixel algorithm.
#
# The output GeoPackage (zbed_anchors.gpkg) is passed directly to
# sf.subgrid.create() as gdf_zb in 13_build_sfincs.py.  HydroMT-SFINCS
# interpolates along merged river lines and lowers the subgrid DEM where
# rivbed < DEM — the DEM file itself (elevation_conditioned.tif) is
# never modified.

rule burn_river_bed:
    input:
        elevation_conditioned = results_path("{basin_id}/inputs/domain/{basin_id}_elevation_conditioned.tif"),
        # Estuarine network: carries the final rivdph (power-law + estuarine blend)
        river_network         = results_path("{basin_id}/inputs/domain/{basin_id}_river_network_estuarine.gpkg"),
    output:
        zbed_anchors = results_path("{basin_id}/inputs/domain/{basin_id}_zbed_anchors.gpkg"),
        plot_preburn = results_path("{basin_id}/visuals/input_data/11_river_preburn.png"),
    log:
        "logs/{basin_id}/11_river_preburn.log"
    script:
        "../scripts/11_river_preburn.py"
