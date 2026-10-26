rule enforce_river_monotonicity:
    """
    Smoothed river water surface from FathomDEM, and the DEM conditioned to it.

    On the GVF node network (src.water_surface): median of the water pixels
    along each centreline near each node (islands skipped), structures at level drops, made
    non-increasing downstream and smoothed. Every land pixel of each channel
    buffer is set to that surface (sea pixels keep their GEBCO bed). Writes
    the surface (rule 10's GVF target), the conditioned DEM and its copy on
    the shared SFINCS grid (base layer of rules 10 and 13).
    """
    input:
        elevation_merged       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_merged.tif"),
        river_network          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
        sfincs_grid            = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_sfincs_grid.json"),
        # On elevation_merged's grid (rule 05b); land use 200 = sea.
        landuse                = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse.tif"),
        # Mean sea level in the model datum: level of all-sea reaches.
        surge_forcing          = results_path("{basin_id}/preprocessing_inputs/forcing/surge_forcing.nc"),
    output:
        elevation_conditioned  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_conditioned.tif"),
        elevation_conditioned_sfincs_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_conditioned_sfincs_grid.tif"),
        # Max conditioned elevation along the centerline; rules 10/13 add
        # active_mask.elevation_buffer_m to set the active-cell elevation ceiling.
        river_elevation_max    = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_elevation_max.json"),
        water_surface          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_water_surface.gpkg"),
        plot_water_surface     = results_path("{basin_id}/preprocessing_inputs/visuals/09_water_surface.png"),
    params:
        water_surface = config["river_processing"]["water_surface"],
        water_code    = config["landuse"]["water_code"],
    threads: rule_threads("enforce_river_monotonicity")
    log:
        "logs/{basin_id}/09_condition_elevation.log"
    script:
        "../scripts/09_condition_elevation.py"
