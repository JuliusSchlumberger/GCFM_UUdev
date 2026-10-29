# Rule for the postprocessing adaptation method (no SFINCS rerun).

rule adapt_metrics_post:
    input:
        baseline_sfincs_map_nc = results_path("{basin_id}/runs/{scenario}/sfincs/sfincs_map.nc"),
        baseline_flood_map_tif = results_path("{basin_id}/runs/{scenario}/visuals/max_flood_depth.tif"),
        attribution_mask_tif = results_path("{basin_id}/runs/{scenario}/attribution_mask.tif"),
        # water_retention sizing reference, from rule attribution_mask like attribution_mask_tif.
        baseline_excess_volume = results_path("{basin_id}/runs/{scenario}/baseline_excess_volume.json"),
        landuse        = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_channels.tif"),
        land_bed_level = results_path("{basin_id}/sfincs_skeleton/land_bed_level.nc"),
        delta_polygon  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_polygon.gpkg"),
        **map_figure_inputs(),
        measure_data   = lambda wildcards: strategy_measure_input_paths(wildcards.strategy, wildcards.basin_id),
        # For run-time-derived strategy params, as in rule adapt_apply_pre.
        baseline_sfincs_inp = results_path("{basin_id}/runs/{scenario}/sfincs/sfincs.inp"),
        skeleton_weir       = results_path("{basin_id}/sfincs_skeleton/sfincs.weir"),
    output:
        flood_map_tif = results_path("{basin_id}/runs/{scenario}/adaptation/post/{strategy}/max_flood_depth.tif"),
        metrics_csv   = results_path("{basin_id}/runs/{scenario}/adaptation/post/{strategy}/flood_metrics.csv"),
        plot_inundation_ratio    = figure_path("62_adapt_inundation_ratio__{scenario}__post_{strategy}.png"),
        plot_max_depth           = figure_path("40_run_max_flood_depth__{scenario}__post_{strategy}.png"),
    params:
        strategy_def         = basin_strategy_def,
        measures_def         = MEASURES_DEFS,
        adaptation_root      = lambda wildcards: adaptation_data_root(),
        baseline_sfincs_root = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{wildcards.scenario}/sfincs"),
        hmin                 = config["metrics"]["hmin"],
        urban_code           = config["metrics"]["urban_landuse_code"],
        run_label            = lambda wildcards: run_label(wildcards.scenario, "post", wildcards.strategy),
    threads: rule_threads("adapt_metrics_post")
    log: "logs/{basin_id}/runs/{scenario}/adaptation/post/{strategy}/17c_adapt_post.log"
    script: "../scripts/17c_adapt_post.py"
