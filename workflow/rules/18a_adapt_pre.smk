# Rules for the preprocessing adaptation method (adapted SFINCS rerun).
# Not yet tested on a rotated grid (sfincs.grid.rotated is true in config.yml); only
# rules 02-17 were validated, on basin 620947. On a rotated grid dep_subgrid.tif,
# max_flood_depth.tif and attribution_mask.tif are rotated GeoTIFFs: check the
# first run, or set sfincs.grid.rotated_overrides: {<basin_id>: false}.

rule adapt_apply_pre:
    # Applies the strategy's measures to a copy of the basin skeleton. Scoped
    # basin x scenario x strategy because `retreat` needs the scenario's
    # baseline flood map. Never reruns spin-up.
    input:
        skeleton_inp       = results_path("{basin_id}/sfincs_skeleton/sfincs.inp"),
        baseline_flood_map = results_path("{basin_id}/runs/{scenario}/visuals/max_flood_depth.tif"),
        landuse            = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_channels.tif"),
        roughness_native   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_roughness.tif"),
        lu_roughness_lookup = raw_input("lu_to_roughness_lookup"),
        sea_mask           = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
        # water_retention sizing reference from rule attribution_mask (18c);
        # only required by strategies that use water_retention.
        baseline_excess_volume = water_retention_excess_volume_input,
        measure_data       = lambda wildcards: strategy_measure_input_paths(wildcards.strategy, wildcards.basin_id),
        # Basin spin-up restart, copied into adapted_root (patched for
        # water_retention, whose zs was computed on the original terrain).
        rstart = results_path("{basin_id}/spin_up/" + RST_FNAME),
        # For run-time-derived strategy params (src.adaptation_scaling): baseline
        # forcing (sfincs.bzs/.dis sit next to the declared sfincs.inp), baseline
        # zsmax, and the current-protection weir.
        baseline_sfincs_inp    = results_path("{basin_id}/runs/{scenario}/sfincs/sfincs.inp"),
        baseline_sfincs_map_nc = results_path("{basin_id}/runs/{scenario}/sfincs/sfincs_map.nc"),
        skeleton_weir          = results_path("{basin_id}/sfincs_skeleton/sfincs.weir"),
    output:
        sfincs_inp = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/sfincs.inp"),
        retreat_landuse = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/retreat_landuse.tif"),
        sea_mask   = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/sea_mask.tif"),
        # As rule land_bed_level, for the adapted skeleton and sea mask.
        land_bed_level = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/land_bed_level.nc"),
        rstart     = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/" + RST_FNAME),
    params:
        strategy_def       = basin_strategy_def,
        measures_def       = MEASURES_DEFS,
        baseline_sfincs_root = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{wildcards.scenario}/sfincs"),
        adaptation_root    = lambda wildcards: adaptation_data_root(),
        skeleton_root      = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        adapted_root       = lambda wildcards: results_path(
            f"{wildcards.basin_id}/runs/{wildcards.scenario}/adaptation/pre/{wildcards.strategy}/sfincs_skeleton"),
    threads: rule_threads("adapt_apply_pre")
    log: "logs/{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/18a_adapt_pre.log"
    script: "../scripts/18a_adapt_pre.py"


rule adapt_build_forcing_pre:
    # Reuses 13_build_sfincs.py, which reads only params, never wildcards.
    input:
        skeleton_inp    = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/sfincs.inp"),
        river_network   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_depth_estimated.gpkg"),
        surge_forcing   = results_path("{basin_id}/preprocessing_inputs/forcing/surge_forcing.nc"),
        river_forcing   = results_path("{basin_id}/preprocessing_inputs/forcing/river_forcing.nc"),
        # Strategy-local restart, so water_retention's adjusted water level is used.
        rstart          = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/" + RST_FNAME),
    output:
        sfincs_inp = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs/sfincs.inp"),
    params:
        # Same as rule build_sfincs's params except the roots below.
        **build_sfincs_params(),
        skeleton_root = lambda wildcards: results_path(
            f"{wildcards.basin_id}/runs/{wildcards.scenario}/adaptation/pre/{wildcards.strategy}/sfincs_skeleton"),
        sfincs_root   = lambda wildcards: results_path(
            f"{wildcards.basin_id}/runs/{wildcards.scenario}/adaptation/pre/{wildcards.strategy}/sfincs"),
        # Holds the strategy-local restart written by adapt_apply_pre.
        spin_up_root  = lambda wildcards: results_path(
            f"{wildcards.basin_id}/runs/{wildcards.scenario}/adaptation/pre/{wildcards.strategy}/sfincs_skeleton"),
    threads: rule_threads("adapt_build_forcing_pre")
    log: "logs/{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/13_build_sfincs.log"
    script: "../scripts/13_build_sfincs.py"


rule adapt_run_event_pre:
    input:
        sfincs_inp          = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs/sfincs.inp"),
        # Dependency tracking only; rstfile is set in sfincs.inp.
        rstart              = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/" + RST_FNAME),
        # Plot background.
        land_mask_on_grid   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
        sea_mask            = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/sea_mask.tif"),
        land_bed_level      = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/land_bed_level.nc"),
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
        **run_event_params(),
    threads: rule_threads("adapt_run_event_pre")
    log: "logs/{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/16_run_event.log"
    script: "../scripts/16_run_event.py"


rule adapt_flood_metrics_pre:
    input:
        sfincs_map_nc = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs/sfincs_map.nc"),
        landuse       = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/retreat_landuse.tif"),
        land_bed_level = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/sfincs_skeleton/land_bed_level.nc"),
        delta_polygon = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_polygon.gpkg"),
    output:
        flood_map_tif = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/max_flood_depth.tif"),
        metrics_csv   = results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/flood_metrics.csv"),
    params:
        sfincs_root   = lambda wildcards: results_path(
            f"{wildcards.basin_id}/runs/{wildcards.scenario}/adaptation/pre/{wildcards.strategy}/sfincs"),
        hmin = config["metrics"]["hmin"], urban_code = config["metrics"]["urban_landuse_code"],
        # Lets 17_flood_metrics.py exclude a water_retention zone from the metrics.
        strategy_def    = basin_strategy_def,
        adaptation_root = lambda wildcards: adaptation_data_root(),
    threads: rule_threads("adapt_flood_metrics_pre")
    log: "logs/{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/17_flood_metrics.log"
    script: "../scripts/17_flood_metrics.py"
