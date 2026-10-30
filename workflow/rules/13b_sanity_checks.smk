# Rule: basin-level sanity checks of the spin-up (config analysis: true), drawn
# from its outputs. The flood map is downscaled (hydromt_sfincs.utils.downscale_floodmap)
# onto the skeleton's land bed level.

rule sanity_checks:
    input:
        sfincs_map_nc       = results_path("{basin_id}/spin_up/sfincs_map.nc"),
        sfincs_his          = results_path("{basin_id}/spin_up/sfincs_his.nc"),
        # Corrected sea/land classification; weir-protected cells are land.
        sea_mask            = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
        land_bed_level      = results_path("{basin_id}/sfincs_skeleton/land_bed_level.nc"),
        # Plot background.
        land_mask_on_grid   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
        clean_river_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
        domain_gpkg         = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
    output:
        plot_inundation_ratio      = analysis_figure_path("13_spinup_inundation_ratio.png"),
        animation_flood_progress   = analysis_figure_path("13_spinup_flood_progression.mp4"),
        plot_water_level           = analysis_figure_path("13_spinup_water_level.png"),
    params:
        run_root                   = lambda wildcards: results_path(f"{wildcards.basin_id}/spin_up"),
        run_label                  = "spinup",
        min_inundation_depth_m     = config["sfincs"]["sanity_checks"]["min_inundation_depth_m"],
        animation_fps              = config["sfincs"]["sanity_checks"]["animation_fps"],
        spinup_days                = lambda wildcards: spinup_days(wildcards.basin_id),
    threads: rule_threads("sanity_checks")
    log:
        "logs/{basin_id}/13b_sanity_checks.log"
    script:
        "../scripts/13b_sanity_checks.py"
