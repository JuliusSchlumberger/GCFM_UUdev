# Rule: build the scenario-independent SFINCS skeleton once per basin: grid,
# elevation, masks, coastal protection weir, roughness, subgrid, observation
# points and zsini. All inputs are basin-level and RP-independent. Rules
# build_sfincs and run_spinup (mean river discharge, calm sea) reference its geometry
# files by relative path.
#
# Every grid input is resampled onto the SFINCS grid exactly once, upstream:
# a native-resolution round trip leaves about half of the corrected fringe
# cells wet (two nearest-neighbour passes do not invert each other).

rule build_sfincs_skeleton:
    input:
        domain_gpkg       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        sfincs_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_sfincs_grid.json"),
        # Native-resolution conditioned (post-monotonicity) elevation, not the raw merged DEM.
        elevation_conditioned = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_conditioned.tif"),
        # Burned channel (rule 11) on the subgrid pixel grid (the script raises
        # otherwise), and its per-cell mean for the main-grid "dep".
        river_burned_subgrid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_burned_subgrid.tif"),
        river_burned_dem_sfincs_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_burned_dem_sfincs_grid.tif"),
        elevation_conditioned_sfincs_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_conditioned_sfincs_grid.tif"),
        river_elevation_max = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_elevation_max.json"),
        coastal_protection_weir = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_coastal_protection_weir.gpkg"),
        # Native resolution, subgrid table only.
        roughness         = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_roughness.tif"),
        # Plot background only; the water-level boundary uses landuse_on_grid.
        land_mask_on_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
        river_network     = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_depth_estimated.gpkg"),
        delta_outflow_points = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_outflow_points.gpkg"),
        # On this model's grid: main "manning" field and weir diagnostics.
        landuse_on_grid   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_on_grid.tif"),
        roughness_on_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_roughness_on_grid.tif"),
        # zsini: sea cells on this model's grid (no reprojection); rule 11's
        # steady mean-discharge level on the bed for the channel cells.
        zsini_sea_cells   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
        river_bed_nodes   = results_path("{basin_id}/preprocessing_inputs/visuals/11_calibration/river_bed/{basin_id}_river_bed_nodes.csv"),
        # Only calm_sea_level_m/coastal_protection_crest_m and bankfull_discharge_m3s/
        # has_glofas (observation points); no forcing timeseries.
        surge_forcing     = results_path("{basin_id}/preprocessing_inputs/forcing/surge_forcing.nc"),
        river_forcing     = results_path("{basin_id}/preprocessing_inputs/forcing/river_forcing.nc"),
    output:
        sfincs_inp     = results_path("{basin_id}/sfincs_skeleton/sfincs.inp"),
        sfincs_subgrid = results_path("{basin_id}/sfincs_skeleton/sfincs_subgrid.nc"),
        sfincs_weir    = results_path("{basin_id}/sfincs_skeleton/sfincs.weir"),
        weir_gpkg      = results_path("{basin_id}/sfincs_skeleton/{basin_id}_coastal_protection_weir.gpkg"),
        zsini          = results_path("{basin_id}/sfincs_skeleton/zsini.tif"),
        plot_grid      = results_path("{basin_id}/preprocessing_inputs/visuals/sfincs_build/01_grid.png"),
        plot_elevation = results_path("{basin_id}/preprocessing_inputs/visuals/sfincs_build/02_elevation.png"),
        plot_mask      = results_path("{basin_id}/preprocessing_inputs/visuals/sfincs_build/03_mask.png"),
        plot_roughness = results_path("{basin_id}/preprocessing_inputs/visuals/sfincs_build/04_roughness.png"),
        plot_coastal_protection_weir = results_path("{basin_id}/preprocessing_inputs/visuals/sfincs_build/07_coastal_protection_weir.png"),
    params:
        resolution_m       = lambda wildcards: grid_resolution_m(wildcards.basin_id),
        nr_subgrid_pixels  = config["sfincs"]["subgrid"]["nr_subgrid_pixels"],
        nr_levels          = config["sfincs"]["subgrid"]["nr_levels"],
        nrmax              = config["sfincs"]["subgrid"]["nrmax"],
        inputs_dir         = lambda wildcards: results_path(f"{wildcards.basin_id}/preprocessing_inputs"),
        skeleton_root      = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        active_mask_enabled        = config["sfincs"]["grid"]["active_mask"]["enabled"],
        active_mask_elevation_buffer_m = config["sfincs"]["grid"]["active_mask"]["elevation_buffer_m"],
        outflow_buffer_m  = config["sfincs"]["boundary_setup"]["outflow_buffer_m"],
        n_top_crossings    = config["sfincs"]["observation_points"]["n_top_crossings"],
        n_per_crossing     = config["sfincs"]["observation_points"]["n_per_crossing"],
        max_downstream_hops = config["sfincs"]["observation_points"]["max_downstream_hops"],
    threads: rule_threads("build_sfincs_skeleton")
    log:
        "logs/{basin_id}/12a_build_sfincs_skeleton.log"
    script:
        "../scripts/12a_build_sfincs_skeleton.py"
