# Rule: sanity checks of each spin-up (config analysis: true), drawn from its
# outputs. The flood map is downscaled (hydromt_sfincs.utils.downscale_floodmap)
# onto the skeleton's land bed level.

rule sanity_checks:
    input:
        sfincs_map_nc       = results_path("{basin_id}/spin_up/slr_{slr}/sfincs_map.nc"),
        sfincs_his          = results_path("{basin_id}/spin_up/slr_{slr}/sfincs_his.nc"),
        # spinup_hr: where the restart is written.
        stability           = results_path("{basin_id}/spin_up/slr_{slr}/stability.csv"),
        # sfincs.obs order: kind and distance from the mouth of every series.
        obs_points          = results_path("{basin_id}/sfincs_skeleton/{basin_id}_obs_points.gpkg"),
        # Corrected sea/land classification; weir-protected cells are land.
        sea_mask            = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
        land_bed_level      = results_path("{basin_id}/sfincs_skeleton/land_bed_level.nc"),
        # Plot background.
        land_mask_on_grid   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
        clean_river_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
        domain_gpkg         = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
    output:
        plot_inundation_ratio      = analysis_figure_path("13_spinup_slr{slr}_inundation_ratio.png"),
        animation_flood_progress   = analysis_figure_path("13_spinup_slr{slr}_flood_progression.mp4"),
        plot_water_level           = analysis_figure_path("13_spinup_slr{slr}_water_level.png"),
        caption_water_level        = analysis_figure_path("13_spinup_slr{slr}_water_level.txt"),
    params:
        run_root                   = lambda wildcards: results_path(f"{wildcards.basin_id}/spin_up/slr_{wildcards.slr}"),
        run_label                  = lambda wildcards: f"spinup_slr{wildcards.slr}",
        min_inundation_depth_m     = config["sfincs"]["sanity_checks"]["min_inundation_depth_m"],
        animation_fps              = config["sfincs"]["sanity_checks"]["animation_fps"],
    wildcard_constraints:
        slr = r"-?[0-9]+(\.[0-9]+)?",
    threads: rule_threads("sanity_checks")
    log:
        "logs/{basin_id}/13b_sanity_checks_slr{slr}.log"
    script:
        "../scripts/13b_sanity_checks.py"
