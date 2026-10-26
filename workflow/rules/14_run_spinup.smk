# Rule: basin-level SFINCS spin-up producing the restart file every scenario's
# event run uses, with the river at its mean discharge and a calm sea (the level every
# event's boundary lead-in starts at); independent of any design RP.

rule run_spinup:
    input:
        skeleton_inp         = results_path("{basin_id}/sfincs_skeleton/sfincs.inp"),
        land_mask_on_grid    = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
        land_bed_level       = results_path("{basin_id}/sfincs_skeleton/land_bed_level.nc"),
        domain_gpkg          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        clean_river_network  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
        surge_forcing        = results_path("{basin_id}/preprocessing_inputs/forcing/surge_forcing.nc"),
        river_forcing        = results_path("{basin_id}/preprocessing_inputs/forcing/river_forcing.nc"),
    output:
        rstart              = results_path("{basin_id}/spin_up/" + RST_FNAME),
        sfincs_map_nc       = results_path("{basin_id}/spin_up/sfincs_map.nc"),
        plot_spinup         = results_path("{basin_id}/spin_up/validation_spinup.png"),
        plot_max_inundation = results_path("{basin_id}/spin_up/validation_max_inundation.png"),
    params:
        skeleton_root             = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        spin_up_root              = lambda wildcards: results_path(f"{wildcards.basin_id}/spin_up"),
        resolution_m              = lambda wildcards: grid_resolution_m(wildcards.basin_id),
        tref                      = config["sfincs"]["simulation"]["tref"],
        spinup_days               = config["sfincs"]["spinup"]["spinup_days"],
        sfincs_exe                = sfincs_exe_path,
        rst_fname                 = RST_FNAME,
        dtmapout_s                = config["sfincs"]["spinup"]["dtmapout_s"],
        dthisout_s                = config["sfincs"]["spinup"]["dthisout_s"],
        timeout_s                 = config["sfincs"]["simulation"]["timeout_s"],
        water_level_buffer_m      = config["sfincs"]["boundary_setup"]["water_level_buffer_m"],
        boundary_ramp_hr          = config["sfincs"]["spinup"]["boundary_ramp_hr"],
    threads: rule_threads("run_spinup")
    log:
        "logs/{basin_id}/14_run_spinup.log"
    script:
        "../scripts/14_run_spinup.py"
