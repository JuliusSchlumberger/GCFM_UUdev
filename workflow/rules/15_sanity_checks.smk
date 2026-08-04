# Rule: sanity checks for the baseline (spinup) condition. Basin-level (no
# {scenario} wildcard) since spin-up itself is now basin-level -- see
# 14_run_spinup.smk's own module comment.
#
# Reads the spinup sfincs_map.nc directly (produced by rule run_spinup) and
# uses hydromt_sfincs.utils.downscale_floodmap to compute the proper flood
# map at the highest available resolution (subgrid dep if present, SFINCS-
# grid zb otherwise) -- the subgrid reference raster lives in the skeleton
# (rule build_sfincs_skeleton), not spin_up_root itself, since spin_up only
# references it via a relative path in its own sfincs.inp.

rule sanity_checks:
    input:
        sfincs_map_nc       = results_path("{basin_id}/spin_up/sfincs_map.nc"),
        landuse             = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse.tif"),
        land_polygons       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
        clean_river_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
        domain_gpkg         = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
    output:
        plot_inundation_ratio      = results_path("{basin_id}/spin_up/01_inundation_ratio.png"),
        animation_flood_progress   = results_path("{basin_id}/spin_up/02_flood_animation.mp4"),
    params:
        spin_up_root               = lambda wildcards: results_path(f"{wildcards.basin_id}/spin_up"),
        skeleton_root              = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        min_inundation_depth_m     = config["sfincs"]["sanity_checks"]["min_inundation_depth_m"],
        include_subgrid            = config["sfincs"]["subgrid"]["enabled"],
        animation_fps              = config["sfincs"]["sanity_checks"]["animation_fps"],
    log:
        "logs/{basin_id}/15_sanity_checks.log"
    script:
        "../scripts/15_sanity_checks.py"
