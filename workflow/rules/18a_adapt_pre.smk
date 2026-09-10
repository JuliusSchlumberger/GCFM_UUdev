# 4 rules for adaptation with preprocessing method

rule adapt_apply_pre:
    # Applies this strategy's measures (via adaptation_method_pre.dispatch_rules,
    # unmodified) to a COPY of the basin's skeleton -- same read-skeleton/
    # redirect-root pattern as build_sfincs.py. Scoped basin x scenario x
    # strategy because `retreat` needs THIS scenario's own baseline flood map.
    # Never re-runs spin-up (see design summary).
    input:
        skeleton_inp       = results_path("{basin_id}/sfincs_skeleton/sfincs.inp"),
        baseline_flood_map = results_path("{basin_id}/runs/{scenario}/visuals/max_flood_depth.tif"),
        landuse            = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse.tif"),
        roughness_native   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_roughness.tif"),
        lu_roughness_lookup = catalogue_path("lu_to_roughness_lookup"),
        sea_mask           = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
        # water_retention's own fixed sizing reference -- computed once by rule
        # attribution_mask (18c), never recomputed or attribution_mask.tif
        # itself touched here. Strategy-conditional (see the input function's
        # own docstring): only strategies that actually use water_retention
        # pull in rule attribution_mask's own prerequisites.
        baseline_excess_volume = water_retention_excess_volume_input,
        measure_data       = lambda wildcards: strategy_measure_input_paths(wildcards.strategy),
    output:
        sfincs_inp = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/sfincs.inp"),
        retreat_landuse = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/retreat_landuse.tif"),
        sea_mask   = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/sea_mask.tif"),
    params:
        strategy_def       = lambda wildcards: STRATEGY_DEFS[wildcards.strategy],
        measures_def       = MEASURES_DEFS,
        adaptation_root    = ADAPT_CATALOGUE_ROOT,
        skeleton_root      = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        adapted_root       = lambda wildcards: results_path(
            f"{wildcards.basin_id}/runs/{wildcards.scenario}/adaptation/pre/{wildcards.strategy}/sfincs_skeleton"),
    log: "logs/{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/18a_adapt_pre.log"
    script: "../scripts/18a_adapt_pre.py"


rule adapt_build_forcing_pre:
    # Reuses scripts/13_build_sfincs.py UNMODIFIED -- it never reads
    # snakemake.wildcards, only snakemake.params paths.
    input:
        skeleton_inp    = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/sfincs.inp"),
        river_network   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_depth_estimated.gpkg"),
        surge_forcing   = results_path("{basin_id}/preprocessing_inputs/forcing/surge_forcing.nc"),
        river_forcing   = results_path("{basin_id}/preprocessing_inputs/forcing/river_forcing.nc"),
        grid_resolution = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_grid_resolution.json"),
        rstart          = results_path("{basin_id}/spin_up/" + RST_FNAME),   # baseline restart, reused
    output:
        sfincs_inp = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs/sfincs.inp"),
    params:
        # identical to rule build_sfincs's own params block --
        depth_method = config["river_processing"]["depth_method"],
        resolution   = lambda wildcards, input: json.load(open(input.grid_resolution))["resolution"],
        tref = config["sfincs"]["simulation"]["tref"],
        dtmapout = config["sfincs"]["simulation"]["dtmapout"],
        dtmaxout = config["sfincs"]["simulation"]["dtmaxout"],
        dthisout = config["sfincs"]["simulation"]["dthisout"],
        storevelmax = config["sfincs"]["simulation"]["storevelmax"],
        storetwet = config["sfincs"]["simulation"]["storetwet"],
        include_rstart = config["sfincs"]["spinup"]["enabled"],
        spinup_days = config["sfincs"]["spinup"]["spinup_days"],
        rst_fname = RST_FNAME,
        river_only_flat_level_m = -(config["terrain"]["gebco_max_depth_m"] + 0.5),
        forcing_mode       = lambda wildcards: scenario_params(wildcards.scenario)["mode"],
        design_rp_river_yr = lambda wildcards: scenario_params(wildcards.scenario)["river_rp"],
        design_rp_surge_yr = lambda wildcards: scenario_params(wildcards.scenario)["surge_rp"],
        compound_lag_hr = config["sfincs"]["boundary_setup"]["compound"]["lag_hr"],
        discharge_multiplier = config["boundary_forcings"]["river"]["discharge_multiplier"],
        slr_enabled = config["boundary_forcings"]["surge"]["slr"]["enabled"],
        slr_m = config["boundary_forcings"]["surge"]["slr"]["slr_m"],
        flat_boundary_point_spacing_m = config["sfincs"]["boundary_setup"]["flat_boundary_point_spacing_m"],
        waterlevel_buffer_m = config["sfincs"]["boundary_setup"]["waterlevel_buffer_m"],
        # only these two differ from build_sfincs's own params:
        skeleton_root = lambda wildcards: results_path(
            f"{wildcards.basin_id}/runs/{wildcards.scenario}/adaptation/pre/{wildcards.strategy}/sfincs_skeleton"),
        sfincs_root   = lambda wildcards: results_path(
            f"{wildcards.basin_id}/runs/{wildcards.scenario}/adaptation/pre/{wildcards.strategy}/sfincs"),
        spin_up_root  = lambda wildcards: results_path(f"{wildcards.basin_id}/spin_up"),
    log: "logs/{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/13_build_sfincs.log"
    script: "../scripts/13_build_sfincs.py"          # REUSED, UNMODIFIED


rule adapt_run_event_pre:
    # Reuses scripts/16_run_event.py UNMODIFIED.
    input:
        sfincs_inp          = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs/sfincs.inp"),
        rstart              = results_path("{basin_id}/spin_up/" + RST_FNAME),
        land_polygons       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
        landuse             = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse.tif"),
        sea_mask            = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/sea_mask.tif"),
        domain_gpkg         = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        clean_river_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
    output:
        sfincs_map_nc            = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs/sfincs_map.nc"),
        plot_inundation_ratio    = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/01_inundation_ratio.png"),
        animation_flood_progress = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/02_flood_animation.mp4"),
        flood_timeseries_csv     = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/flood_timeseries.csv"),
    params:
        sfincs_root   = lambda wildcards: results_path(
            f"{wildcards.basin_id}/runs/{wildcards.scenario}/adaptation/pre/{wildcards.strategy}/sfincs"),
        # the ADAPTED skeleton -- subgrid dep/roughness differs there for `retreat`
        skeleton_root = lambda wildcards: results_path(
            f"{wildcards.basin_id}/runs/{wildcards.scenario}/adaptation/pre/{wildcards.strategy}/sfincs_skeleton"),
        sfincs_exe = config["sfincs"]["simulation"]["sfincs_exe"],
        timeout_s  = config["sfincs"]["simulation"]["timeout_s"],
        min_inundation_depth_m = config["sfincs"]["sanity_checks"]["min_inundation_depth_m"],
        include_subgrid = config["sfincs"]["subgrid"]["enabled"],
        animation_fps = config["sfincs"]["sanity_checks"]["animation_fps"],
    threads: workflow.cores
    log: "logs/{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/16_run_event.log"
    script: "../scripts/16_run_event.py"             # REUSED, UNMODIFIED


rule adapt_flood_metrics_pre:
    # Reuses scripts/17_flood_metrics.py UNMODIFIED -- it uses
    # snakemake.wildcards.basin_id/scenario directly, both still correct here.
    input:
        sfincs_map_nc = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs/sfincs_map.nc"),
        landuse       = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/retreat_landuse.tif"),
        sea_mask      = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/sea_mask.tif"),
        delta_polygon = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_polygon.gpkg"),
    output:
        flood_map_tif = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/max_flood_depth.tif"),
        metrics_csv   = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/flood_metrics.csv"),
    params:
        sfincs_root   = lambda wildcards: results_path(
            f"{wildcards.basin_id}/runs/{wildcards.scenario}/adaptation/pre/{wildcards.strategy}/sfincs"),
        skeleton_root = lambda wildcards: results_path(
            f"{wildcards.basin_id}/runs/{wildcards.scenario}/adaptation/pre/{wildcards.strategy}/sfincs_skeleton"),
        hmin = config["metrics"]["hmin"], urban_code = config["metrics"]["urban_landuse_code"],
        include_subgrid = config["sfincs"]["subgrid"]["enabled"],
    log: "logs/{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/17_flood_metrics.log"
    script: "../scripts/17_flood_metrics.py"         # REUSED, UNMODIFIED
