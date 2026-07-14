# Rule: sanity checks for the baseline (spinup) condition.
#
# Reads the spinup sfincs_map.nc directly (produced by SFINCS in rule 14) and
# uses hydromt_sfincs.utils.downscale_floodmap to compute the proper flood map
# at the highest available resolution (subgrid dep if present, SFINCS-grid zb
# otherwise).  The land-use raster's "Sea" class (200) excludes open sea from
# both the flooded count and the land-domain denominator, so the flooded
# fraction is reported as a share of the land domain only.  Land polygons and
# the clean river network are overlaid on the diagnostic plot for context.

rule sanity_checks:
    input:
        sfincs_map_nc       = results_path("{basin_id}/sfincs/spinup/sfincs_map.nc"),
        landuse             = results_path("{basin_id}/inputs/domain/{basin_id}_landuse.tif"),
        land_polygons       = results_path("{basin_id}/inputs/domain/{basin_id}_land_polygons.gpkg"),
        clean_river_network = results_path("{basin_id}/inputs/domain/{basin_id}_river_network_clean.gpkg"),
        domain_gpkg         = results_path("{basin_id}/inputs/domain/{basin_id}_domain.gpkg"),
    output:
        plot_inundation_ratio      = results_path("{basin_id}/visuals/model_runs/spinup/01_inundation_ratio.png"),
        animation_flood_progress   = results_path("{basin_id}/visuals/model_runs/spinup/02_flood_animation.mp4"),
    params:
        sfincs_root                = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs"),
        min_inundation_depth_m     = config["sfincs"]["sanity_checks"]["min_inundation_depth_m"],
        include_subgrid            = config["sfincs"]["subgrid"]["enabled"],
        animation_fps              = config["sfincs"]["sanity_checks"]["animation_fps"],
    log:
        "logs/{basin_id}/15_sanity_checks.log"
    script:
        "../scripts/15_sanity_checks.py"
