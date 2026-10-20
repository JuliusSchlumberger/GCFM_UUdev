rule enforce_river_monotonicity:
    """
    Make the DEM non-increasing downstream along every river centerline.

    Modifies the DEM, not the network: each centerline pixel above the running
    minimum from upstream is lowered to it, at native (FathomDEM) resolution.
    Needs only the cleaned network's topology/width, not rivdph, so it does not
    depend on rule 10. Also writes the conditioned DEM on the shared SFINCS
    grid, the base elevation layer of rules 10 and 13 (the burned channel is
    layered on top).
    """
    input:
        elevation_merged       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_merged.tif"),
        river_network          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
        sfincs_grid            = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_sfincs_grid.json"),
    output:
        elevation_conditioned  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_conditioned.tif"),
        elevation_conditioned_sfincs_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_conditioned_sfincs_grid.tif"),
        # Max conditioned elevation along the centerline; rules 10/13 add
        # active_mask.elevation_buffer_m to set the active-cell elevation ceiling.
        river_elevation_max    = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_elevation_max.json"),
    log:
        "logs/{basin_id}/09_condition_elevation.log"
    script:
        "../scripts/09_condition_elevation.py"
