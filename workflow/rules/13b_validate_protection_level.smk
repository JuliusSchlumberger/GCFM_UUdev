# Rule: validate that the BUILT production model (real burned channel + real
# calibrated/coastal weir + real floodplain -- not a re-confined corridor)
# actually holds at its own protection standard, and shows only modest,
# controlled overtopping just beyond it.
#
# Runs two short, steady discharge scenarios against the SAME production
# model rule 13 already built (grid, elevation, roughness, subgrid, weir all
# REFERENCED via relative ../ paths, not rebuilt -- same technique
# 14_run_spinup.py already uses to reuse the main model's own static files):
#   1. protection-level design discharge (river_processing.river_depth_modelling's
#      own calibration target -- protection_levels.json's riverine_rp_yr,
#      bankfull as the fallback)
#   2. that same discharge x sfincs.protection_validation.higher_rp_factor
#
# Independent of rule 14 (spin-up)/16 (event) -- does not gate either, and
# neither gates this rule; it exists purely as an ongoing, repeatable check.

rule validate_protection_level:
    input:
        sfincs_inp    = results_path("{basin_id}/scenarios/default/sfincs/sfincs.inp"),
        river_forcing = results_path("{basin_id}/inputs/forcing/river_forcing.nc"),
        protection_levels = results_path("{basin_id}/inputs/domain/protection_levels.json"),
        land_polygons = results_path("{basin_id}/inputs/domain/{basin_id}_land_polygons.gpkg"),
        landuse       = results_path("{basin_id}/inputs/domain/{basin_id}_landuse.tif"),
        domain_gpkg   = results_path("{basin_id}/inputs/domain/{basin_id}_domain.gpkg"),
        clean_river_network = results_path("{basin_id}/inputs/domain/{basin_id}_river_network_clean.gpkg"),
    output:
        plot_max_inundation_protection = results_path("{basin_id}/visuals/model_runs/protection_validation/protection_level_max_inundation.png"),
        plot_water_level_protection    = results_path("{basin_id}/visuals/model_runs/protection_validation/protection_level_water_level.png"),
        plot_max_inundation_higher     = results_path("{basin_id}/visuals/model_runs/protection_validation/higher_rp_max_inundation.png"),
        plot_water_level_higher        = results_path("{basin_id}/visuals/model_runs/protection_validation/higher_rp_water_level.png"),
    params:
        sfincs_root       = lambda wildcards: results_path(f"{wildcards.basin_id}/scenarios/default/sfincs"),
        sfincs_exe        = config["sfincs"]["simulation"]["sfincs_exe"],
        timeout_s         = config["sfincs"]["simulation"]["timeout_s"],
        include_subgrid   = config["sfincs"]["subgrid"]["enabled"],
        higher_rp_factor  = config["sfincs"]["protection_validation"]["higher_rp_factor"],
        simulation_days   = config["sfincs"]["protection_validation"]["simulation_days"],
        convergence_window_hours    = config["sfincs"]["protection_validation"]["convergence_window_hours"],
        convergence_abs_tolerance_m = config["sfincs"]["protection_validation"]["convergence_abs_tolerance_m"],
        convergence_rel_tolerance   = config["sfincs"]["protection_validation"]["convergence_rel_tolerance"],
        river_only_flat_level_m = -(config["terrain"]["gebco_max_depth_m"] + 0.5),
    threads: workflow.cores
    log:
        "logs/{basin_id}/13b_validate_protection_level.log"
    script:
        "../scripts/13b_validate_protection_level.py"
