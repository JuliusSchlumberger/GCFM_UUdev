# Rule for the postprocessing adaptation method (no SFINCS rerun).
# Not yet tested on a rotated grid (sfincs.grid.rotated defaults to true); only
# rules 02-17 were validated, on basin 620947. On a rotated grid dep_subgrid.tif,
# max_flood_depth.tif and attribution_mask.tif are rotated GeoTIFFs: check the
# first run, or set sfincs.grid.rotated_overrides: {<basin_id>: false}.

rule adapt_metrics_post:
    input:
        baseline_sfincs_map_nc = results_path("{basin_id}/runs/{scenario}/sfincs/sfincs_map.nc"),
        baseline_flood_map_tif = results_path("{basin_id}/runs/{scenario}/visuals/max_flood_depth.tif"),
        attribution_mask_tif = results_path("{basin_id}/runs/{scenario}/attribution_mask.tif"),
        # water_retention sizing reference, from rule attribution_mask like attribution_mask_tif.
        baseline_excess_volume = results_path("{basin_id}/runs/{scenario}/baseline_excess_volume.json"),
        landuse        = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse.tif"),
        sea_mask       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
        delta_polygon  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_polygon.gpkg"),
        land_mask_on_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),  # plot background
        river_network  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
        measure_data   = lambda wildcards: strategy_measure_input_paths(wildcards.strategy),
        # For run-time-derived strategy params, as in rule adapt_apply_pre.
        baseline_sfincs_inp = results_path("{basin_id}/runs/{scenario}/sfincs/sfincs.inp"),
        skeleton_weir       = results_path("{basin_id}/sfincs_skeleton/sfincs.weir"),
    output:
        flood_map_tif = results_path("{basin_id}/runs/{scenario}/adaptation/post/{strategy}/max_flood_depth.tif"),
        metrics_csv   = results_path("{basin_id}/runs/{scenario}/adaptation/post/{strategy}/flood_metrics.csv"),
        plot_inundation_ratio    = results_path("{basin_id}/runs/{scenario}/adaptation/post/{strategy}/01_inundation_ratio.png")
    params:
        strategy_def         = lambda wildcards: strategy_def_for(wildcards.strategy, wildcards.scenario),
        measures_def         = MEASURES_DEFS,
        adaptation_root      = ADAPT_CATALOGUE_ROOT,
        baseline_sfincs_root = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{wildcards.scenario}/sfincs"),
        skeleton_root        = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        hmin                 = config["metrics"]["hmin"],
        urban_code           = config["metrics"]["urban_landuse_code"],
    log: "logs/{basin_id}/runs/{scenario}/adaptation/post/{strategy}/18b_adapt_post.log"
    script: "../scripts/18b_adapt_post.py"
