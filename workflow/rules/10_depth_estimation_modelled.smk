# Rule: river depth from GVF and dike-crest calibration with SFINCS -- the only
# way river depth and weir crests are set.
#
# The conditioned elevation (rule 09, channels at the smoothed water surface)
# is the base the GVF bed is burned into: the native version for the subgrid,
# the SFINCS-grid version as the sf.elevation.create() base layer, identical
# to rule 13's.

rule modelled_depth_estimation:
    input:
        elevation_conditioned = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_conditioned.tif"),
        elevation_conditioned_sfincs_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_conditioned_sfincs_grid.tif"),
        # Sets the active-cell elevation ceiling, as in rule 13.
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
        # boundary, as in rule 13. Always present, possibly empty.
        delta_outflow_points     = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_outflow_points.gpkg"),
        # Rule 09's smoothed water surface (the GVF target)
        # and the land use for the sea share of each GVF cross-section.
        water_surface            = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_water_surface.gpkg"),
        landuse                  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_channels.tif"),
    output:
        depth_estimated_river_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_depth_estimated.gpkg"),
        # The burned channel from round 0's per-cell excavation (as simulated
        # in rounds 1 and 2); rule 13 imports it rather than re-deriving it.
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
        # Fixed-name copies of round 2's (verification) files.
        plot_calibration             = results_path("{basin_id}/preprocessing_inputs/visuals/10_calibration/river_depth.png"),
        # River network on land: colour = river depth (GVF: water depth at the
        # mean discharge; modelled: excavated depth), line width = channel width.
        plot_river_network_depth     = results_path("{basin_id}/preprocessing_inputs/visuals/10_calibration/river_network_depth.png"),
        plot_water_level_timeseries  = results_path("{basin_id}/preprocessing_inputs/visuals/10_calibration/water_level_timeseries.png"),
        plot_max_inundation          = results_path("{basin_id}/preprocessing_inputs/visuals/10_calibration/max_inundation.png"),
        animation_flood_progress     = results_path("{basin_id}/preprocessing_inputs/visuals/10_calibration/flood_animation.mp4"),
        plot_crest_gap_map           = results_path("{basin_id}/preprocessing_inputs/visuals/10_calibration/crest_gap_map.png"),
        # Dike position with unprotected_ocean_wetlands off vs on; config picks the simulated one.
        plot_dike_positions          = results_path("{basin_id}/preprocessing_inputs/visuals/10_calibration/dike_positions_wetland_switch.png"),
        # Per-round diagnostics (round 0 = confined/un-excavated, round 1 =
        # confined/excavated, round 2 = verification with the real crests) are
        # written undeclared to round_visuals_dir.
    params:
        calib_root  = lambda wildcards: results_path(f"{wildcards.basin_id}/preprocessing_inputs/depth_crest_calibration"),
        resolution_m = lambda wildcards: grid_resolution_m(wildcards.basin_id),
        sfincs_exe  = sfincs_exe_path,
        timeout_s   = config["sfincs"]["simulation"]["timeout_s"],
        active_mask_enabled = config["sfincs"]["grid"]["active_mask"]["enabled"],
        active_mask_elevation_buffer_m = config["sfincs"]["grid"]["active_mask"]["elevation_buffer_m"],
        outflow_buffer_m = config["sfincs"]["boundary_setup"]["outflow_buffer_m"],
        # Same station matching radius as rules 13/14; rounds 1-2 force the
        # coastal protection storm tide there.
        water_level_buffer_m = config["sfincs"]["boundary_setup"]["water_level_buffer_m"],
        gvf                          = config["river_processing"]["river_depth_modelling"]["gvf"],
        water_surface                = config["river_processing"]["water_surface"],
        channel_manning_n            = config["river_processing"]["channel_manning_n"],
        calibration_days             = config["river_processing"]["river_depth_modelling"]["calibration_days"],
        gebco_max_depth_m            = config["terrain"]["gebco_max_depth_m"],
        discharge_ramp_hr            = config["river_processing"]["river_depth_modelling"]["discharge_ramp_hr"],
        weir_crest_m                  = config["river_processing"]["river_depth_modelling"]["weir_crest_m"],
        weir_par1                     = config["river_processing"]["river_depth_modelling"]["weir_par1"],
        weir_freeboard_m              = config["river_processing"]["river_depth_modelling"]["freeboard_m"],
        unprotected_ocean_wetlands    = config["river_processing"]["river_depth_modelling"]["unprotected_ocean_wetlands"],
        min_component_cells           = config["river_processing"]["river_depth_modelling"]["min_component_cells"],
        animation_fps                 = config["sfincs"]["sanity_checks"]["animation_fps"],
        # Same subgrid setup as rules 13/14, so calibration sees the production bathymetry.
        nr_subgrid_pixels  = config["sfincs"]["subgrid"]["nr_subgrid_pixels"],
        nr_levels          = config["sfincs"]["subgrid"]["nr_levels"],
        nrmax              = config["sfincs"]["subgrid"]["nrmax"],
    threads: rule_threads("modelled_depth_estimation")
    log:
        "logs/{basin_id}/10_depth_estimation_modelled.log"
    script:
        "../scripts/10_depth_estimation_modelled.py"
