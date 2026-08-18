# post processing rule for adaptation

rule adapt_metrics_post:
    input:
        baseline_sfincs_map_nc = results_path("{basin_id}/runs/{scenario}/sfincs/sfincs_map.nc"),
        baseline_flood_map_tif = results_path("{basin_id}/runs/{scenario}/visuals/max_flood_depth.tif"),
        attribution_mask_tif = results_path("{basin_id}/runs/{scenario}/attribution_mask.tif"),
        landuse        = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse.tif"),
        sea_mask       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
        delta_polygon  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_polygon.gpkg"),
        land_polygons  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
        river_network  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
        measure_data   = lambda wildcards: strategy_measure_input_paths(wildcards.strategy),
    output:
        flood_map_tif = results_path("{basin_id}/runs/{scenario}/adaptation/post/{strategy}/max_flood_depth.tif"),
        metrics_csv   = results_path("{basin_id}/runs/{scenario}/adaptation/post/{strategy}/flood_metrics.csv"),
        plot_inundation_ratio    = results_path("{basin_id}/runs/{scenario}/adaptation/post/{strategy}/01_inundation_ratio.png")
    params:
        strategy_def         = lambda wildcards: STRATEGY_DEFS[wildcards.strategy],
        measures_def         = MEASURES_DEFS,
        adaptation_root      = ADAPT_CATALOGUE_ROOT,
        baseline_sfincs_root = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{wildcards.scenario}/sfincs"),
        skeleton_root        = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        hmin                 = config["metrics"]["hmin"],
        urban_code           = config["metrics"]["urban_landuse_code"],
        include_subgrid      = config["sfincs"]["subgrid"]["enabled"],
    log: "logs/{basin_id}/runs/{scenario}/adaptation/post/{strategy}/18b_adapt_post.log"
    script: "../scripts/18b_adapt_post.py"
