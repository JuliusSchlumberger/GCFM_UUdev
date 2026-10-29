# Rule: river bed from the water-surface slope (src/river_bed.py) and dike-crest
# calibration with SFINCS -- the only way river depth and weir crests are set.
#
# The conditioned elevation (rule 10, channels at the smoothed water surface)
# is the base the bed is burned into: the native version for the subgrid,
# the SFINCS-grid version as the sf.elevation.create() base layer, identical
# to rule 12a's.

rule modelled_depth_estimation:
    input:
        elevation_conditioned = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_conditioned.tif"),
        elevation_conditioned_sfincs_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_conditioned_sfincs_grid.tif"),
        # Sets the active-cell elevation ceiling, as in rule 12a.
        river_elevation_max = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_elevation_max.json"),
        clean_river_network      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
        river_forcing           = results_path("{basin_id}/preprocessing_inputs/forcing/river_forcing.nc"),
        protection_levels        = results_path("{basin_id}/preprocessing_inputs/domain/protection_levels.json"),
        # Native resolution, subgrid table only; the regular-grid manning field
        # and the zsini/weir/ocean_mask sections use the grid-aligned rasters.
        roughness                = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_roughness.tif"),
        # Plot background only; the water-level boundary uses landuse_on_grid.
        land_mask_on_grid        = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
        domain_gpkg              = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        sfincs_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_sfincs_grid.json"),
        spec_basins_meta         = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        # On this rule's grid; no reprojection.
        landuse_on_grid          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_on_grid.tif"),
        roughness_on_grid        = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_roughness_on_grid.tif"),
        # Real, steady coastal baseline (calm sea level, MDT-corrected; SLR is
        # applied at build time) for the water-level boundary.
        surge_forcing            = results_path("{basin_id}/preprocessing_inputs/forcing/surge_forcing.nc"),
        # Where a non-seed, non-mouth, non-bifurcation reach crosses the delta
        # outline (flow leaves without reaching a coastal mouth): free outflow
        # boundary, as in rule 12a. Always present, possibly empty.
        delta_outflow_points     = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_outflow_points.gpkg"),
        # Rule 10's node table: surface, slope and node class on the shared network.
        water_surface            = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_water_surface.gpkg"),
        # SWORD node widths for the depth.
        sword_regions            = raw_region_inputs("river_network"),
    output:
        depth_estimated_river_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_depth_estimated.gpkg"),
        # The burned channel from round 0's per-cell excavation (as simulated
        # in rounds 1 and 2); rule 12a imports it rather than re-deriving it.
        # - river_burned_subgrid: on the subgrid pixel grid (main grid / nr_subgrid_pixels).
        # - river_burned_dem_sfincs_grid: per-cell mean of those subgrid pixels.
        river_burned_subgrid          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_burned_subgrid.tif"),
        river_burned_dem_sfincs_grid  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_burned_dem_sfincs_grid.tif"),
        # Traced weir; each edge's crest from round 1's water level on its
        # water side (verified in round 2), not derivable from per-reach values.
        coastal_protection_weir       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_coastal_protection_weir.gpkg"),
        # The sea/land classification for the skeleton's zsini, written on
        # this rule's grid (pixel-identical to the skeleton's, both from
        # sfincs_grid.json), so it is used with zero resampling.
        zsini_sea_cells               = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
        # Round 0's node table (bed, width, level, design discharge along the
        # network); read by the analysis checks (analysis/_upstream_profile.py).
        # Round 1's subgrid table and mask; rule 12a copies the table when its
        # mask is identical instead of building the same table again.
        calibration_subgrid          = calibration_subgrid_files(),
        river_bed_nodes              = results_path("{basin_id}/preprocessing_inputs/visuals/11_calibration/river_bed/{basin_id}_river_bed_nodes.csv"),
        # DEM, bed and crest along each seed-to-mouth path.
        plot_crest_profile           = figure_path("30_calib_crest_profile.png"),
        # Boundary forcing and max flood depth of every SFINCS round (calib_r1, calib_r2).
        **calibration_round_figures(),
        **analysis_outputs(
            # River network on land: colour = water depth at the design discharge
            # (dashed where the slope floor sets it), line width = channel width.
            plot_river_network_depth         = analysis_figure_path("11_calib_river_depth_network.png"),
            # Same network: colour = steady GVF level on the bed at the design
            # discharge minus rule 10's surface (fixed -3..3 m).
            plot_river_network_forward_check = analysis_figure_path("11_calib_river_forward_check.png"),
            # Crest minus the last round's period-max water level.
            plot_crest_gap_map               = analysis_figure_path("11_calib_crest_gap.png"),
            # Dike position with unprotected_ocean_wetlands off vs on; config picks the simulated one.
            plot_dike_positions              = analysis_figure_path("11_calib_dike_position_wetlands.png"),
        ),
    params:
        calib_root  = lambda wildcards: results_path(f"{wildcards.basin_id}/preprocessing_inputs/depth_crest_calibration"),
        resolution_m = lambda wildcards: grid_resolution_m(wildcards.basin_id),
        sfincs_exe  = sfincs_exe_path,
        timeout_s   = config["sfincs"]["simulation"]["timeout_s"],
        active_mask_enabled = config["sfincs"]["grid"]["active_mask"]["enabled"],
        active_mask_elevation_buffer_m = config["sfincs"]["grid"]["active_mask"]["elevation_buffer_m"],
        outflow_buffer_m = config["sfincs"]["boundary_setup"]["outflow_buffer_m"],
        # Same station matching radius as rules 13a/14; rounds 1-2 force the
        # coastal protection storm tide there.
        water_level_buffer_m = config["sfincs"]["boundary_setup"]["water_level_buffer_m"],
        bed                          = config["river_processing"]["river_depth_modelling"]["bed"],
        gebco_min_depth_m            = config["terrain"]["gebco_min_depth_m"],
        water_surface                = config["river_processing"]["water_surface"],
        channel_manning_n            = config["river_processing"]["channel_manning_n"],
        calibration_margin_hr        = config["river_processing"]["river_depth_modelling"]["calibration_margin_hr"],
        wave_regime_rT               = config["river_processing"]["river_depth_modelling"]["wave_regime_rT"],
        calibration_max_days         = config["river_processing"]["river_depth_modelling"]["calibration_max_days"],
        steady_check                 = config["river_processing"]["river_depth_modelling"]["steady_check"],
        verify_crests                = config["river_processing"]["river_depth_modelling"]["verify_crests"],
        ramp_hr                      = config["river_processing"]["river_depth_modelling"]["ramp_hr"],
        weir_crest_m                  = config["river_processing"]["river_depth_modelling"]["weir_crest_m"],
        weir_par1                     = config["river_processing"]["river_depth_modelling"]["weir_par1"],
        weir_freeboard_m              = config["river_processing"]["river_depth_modelling"]["freeboard_m"],
        unprotected_ocean_wetlands    = config["river_processing"]["river_depth_modelling"]["unprotected_ocean_wetlands"],
        min_component_cells           = config["river_processing"]["river_depth_modelling"]["min_component_cells"],
        animation_fps                 = config["sfincs"]["sanity_checks"]["animation_fps"],
        # Same subgrid setup as rules 12a/13a, so calibration sees the production bathymetry.
        nr_subgrid_pixels  = config["sfincs"]["subgrid"]["nr_subgrid_pixels"],
        nr_levels          = config["sfincs"]["subgrid"]["nr_levels"],
        nrmax              = config["sfincs"]["subgrid"]["nrmax"],
    threads: rule_threads("modelled_depth_estimation")
    log:
        "logs/{basin_id}/11_depth_estimation_modelled.log"
    script:
        "../scripts/11_depth_estimation_modelled.py"
