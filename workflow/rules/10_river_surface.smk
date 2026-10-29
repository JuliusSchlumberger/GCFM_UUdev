rule river_surface:
    """
    River water surface from FathomDEM, its slope, and the DEM conditioned to it.

    On the node network (src.water_surface): median of the water pixels along
    each centreline near each node (islands skipped), sea mouths pinned at the
    calm sea level, made non-increasing downstream and smoothed; the slope
    along each node's largest-Q path, checked against SWOT. Every land pixel of
    each channel buffer is set to that surface (sea pixels keep their GEBCO
    bed). Writes the node table (surface, slope, node class; rule 11's river
    bed and the zsini river cells of rules 11 and 12a), the conditioned DEM and
    its copy on the shared SFINCS grid (base layer of rules 11 and 12a).
    """
    input:
        elevation_merged       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_merged.tif"),
        river_network          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
        sfincs_grid            = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_sfincs_grid.json"),
        # On elevation_merged's grid (rule 06); land use 200 = sea.
        landuse                = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse.tif"),
        # Sea share of each node's cross-section (node class).
        landuse_channels       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_channels.tif"),
        # Calm sea level in the model datum: the mouths are pinned at it.
        surge_forcing          = results_path("{basin_id}/preprocessing_inputs/forcing/surge_forcing.nc"),
    output:
        elevation_conditioned  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_conditioned.tif"),
        elevation_conditioned_sfincs_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_conditioned_sfincs_grid.tif"),
        # Max conditioned elevation along the centerline; rules 11/12a add
        # active_mask.elevation_buffer_m to set the active-cell elevation ceiling.
        river_elevation_max    = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_elevation_max.json"),
        water_surface          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_water_surface.gpkg"),
        **analysis_outputs(plot_water_surface=analysis_figure_path("10_river_surface.png")),
    params:
        water_surface = config["river_processing"]["water_surface"],
        water_code    = config["landuse"]["water_code"],
        gebco_max_depth_m = config["terrain"]["gebco_max_depth_m"],
    threads: rule_threads("river_surface")
    log:
        "logs/{basin_id}/10_river_surface.log"
    script:
        "../scripts/10_river_surface.py"
