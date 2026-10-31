# Flood attribution: classify flooding by source (river/coast). The matching
# river-only and coastal-only scenarios must be defined in scenarios.yml.

rule attribution_mask:
    input:
        river_tif    = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{attribution_counterparts(wildcards.scenario)[0]}/visuals/max_flood_depth.tif"),
        coastal_tif  = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{attribution_counterparts(wildcards.scenario)[1]}/visuals/max_flood_depth.tif"),
        # Class 4 reference: the last crest calibration round (rule 11; design
        # discharge and coastal storm tide, held by the dikes). It has no
        # max_flood_depth.tif; the script derives one from sfincs_map.nc as rule 16 does.
        reference_map_nc = results_path(
            f"{{basin_id}}/preprocessing_inputs/depth_crest_calibration/round{CALIB_ROUNDS[-1]}/sfincs_map.nc"
        ),
        land_bed_level = results_path("{basin_id}/sfincs_skeleton/land_bed_level.nc"),
        skeleton_inp  = results_path("{basin_id}/sfincs_skeleton/sfincs.inp"),  # mod_ref, for the diagnostic plot's own basemap
        # Plot background.
        land_mask_on_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
        # Unadapted flood map, source of baseline_excess_volume.
        baseline_flood_map = results_path("{basin_id}/runs/{scenario}/visuals/max_flood_depth.tif"),
    output:
        attribution_mask_tif = results_path("{basin_id}/runs/{scenario}/attribution_mask.tif"),
        attribution_mask_png = figure_path("61_adapt_attribution_mask__{scenario}.png"),
        # classes=(1,3), i.e. river and compound (coastal water and class 4
        # excluded); read by rules 17b and 17c for water_retention sizing.
        baseline_excess_volume = results_path("{basin_id}/runs/{scenario}/baseline_excess_volume.json"),
    params:
        skeleton_root   = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        reference_root  = lambda wildcards: results_path(
            f"{wildcards.basin_id}/preprocessing_inputs/depth_crest_calibration/round{CALIB_ROUNDS[-1]}"
        ),
        hmin = config["metrics"]["hmin"],
    threads: rule_threads("attribution_mask")
    log: "logs/{basin_id}/runs/{scenario}/17a_attribution_mask.log"
    script: "../scripts/17a_attribution_mask.py"
