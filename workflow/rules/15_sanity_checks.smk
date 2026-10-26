# Rule: basin-level sanity checks of the spin-up. The flood map is downscaled
# (hydromt_sfincs.utils.downscale_floodmap) onto the skeleton's land bed level.

rule sanity_checks:
    input:
        sfincs_map_nc       = results_path("{basin_id}/spin_up/sfincs_map.nc"),
        # Corrected sea/land classification; weir-protected cells are land.
        sea_mask            = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
        land_bed_level      = results_path("{basin_id}/sfincs_skeleton/land_bed_level.nc"),
        # Plot background.
        land_mask_on_grid   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
        clean_river_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
        domain_gpkg         = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
    output:
        plot_inundation_ratio      = results_path("{basin_id}/spin_up/01_inundation_ratio.png"),
        animation_flood_progress   = results_path("{basin_id}/spin_up/02_flood_animation.mp4"),
    params:
        spin_up_root               = lambda wildcards: results_path(f"{wildcards.basin_id}/spin_up"),
        min_inundation_depth_m     = config["sfincs"]["sanity_checks"]["min_inundation_depth_m"],
        animation_fps              = config["sfincs"]["sanity_checks"]["animation_fps"],
    threads: rule_threads("sanity_checks")
    log:
        "logs/{basin_id}/15_sanity_checks.log"
    script:
        "../scripts/15_sanity_checks.py"
