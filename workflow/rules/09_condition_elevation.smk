rule enforce_river_monotonicity:
    """
    Modifies the DEM (not the river network) -- this rule conditions the
    ELEVATION RASTER, using the river network purely as a guide for where and
    in what order to walk. Enforces a monotonically non-increasing DEM
    elevation profile along every river centerline in the downstream
    direction: any centerline pixel whose DEM elevation exceeds the running
    minimum from upstream is lowered to that minimum, producing
    elevation_conditioned.tif at NATIVE (FathomDEM) resolution.

    Operates on the CLEANED river network (river_network_clean.gpkg, rule
    08's output) -- only topology/width (reach_id, rch_id_dn, is_seed,
    width) are needed, never rivdph, so this rule does not depend on either
    depth-estimation branch (rule empirical_depth_estimation or
    modelled_depth_estimation, both numbered 10) -- and on the raw merged
    DEM (elevation_merged.tif, rule 05a's output).

    Also produces a SECOND output: the same conditioned DEM resampled onto
    the shared SFINCS grid (rule 08c's build_sfincs_grid,
    {basin_id}_sfincs_grid.json) -- the single, shared coarse "background"
    elevation layer both rule modelled_depth_estimation (10, modelled depth
    calibration) and rule 13 (production build) use as their
    sf.elevation.create() base layer, instead of each independently letting
    HydroMT resample from the native file. river_burned_dem.tif (produced
    directly by whichever of rule empirical_depth_estimation or
    modelled_depth_estimation actually runs, both native- and
    SFINCS-grid-resolution) is unaffected -- it still layers on top of this
    coarse background as the higher-priority elevation_list entry in
    rule 13.

    Always scheduled -- every basin gets a conditioned DEM, unconditionally.
    Rule empirical_depth_estimation or modelled_depth_estimation (10) is the
    merge point needing both this rule's native-resolution conditioned
    elevation and the depth-estimated network.
    """
    input:
        elevation_merged       = results_path("{basin_id}/inputs/domain/{basin_id}_elevation_merged.tif"),
        river_network          = results_path("{basin_id}/inputs/domain/{basin_id}_river_network_clean.gpkg"),
        sfincs_grid            = results_path("{basin_id}/inputs/domain/{basin_id}_sfincs_grid.json"),
    output:
        elevation_conditioned  = results_path("{basin_id}/inputs/domain/{basin_id}_elevation_conditioned.tif"),
        elevation_conditioned_sfincs_grid = results_path("{basin_id}/inputs/domain/{basin_id}_elevation_conditioned_sfincs_grid.tif"),
        # Max conditioned (post-monotonicity) elevation along the river's own
        # centerline -- rule modelled_depth_estimation (10)/13 add
        # sfincs.grid.active_mask's own elevation_buffer_m on top to set the
        # active-cell mask's elevation ceiling (see enforce_river_monotonicity's
        # own docstring).
        river_elevation_max    = results_path("{basin_id}/inputs/domain/{basin_id}_river_elevation_max.json"),
        plot_conditioning      = results_path("{basin_id}/visuals/input_data/09_condition_elevation.png"),
    log:
        "logs/{basin_id}/09_condition_elevation.log"
    script:
        "../scripts/09_condition_elevation.py"
