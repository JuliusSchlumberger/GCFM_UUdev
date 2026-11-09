# Rule: SFINCS spin-up at mean conditions (river at its mean discharge, calm sea +
# slr_m x fingerprint), one per slr_m the requested scenarios use: every event
# restarts from the one with its own slr_m. Runs in chunks of chunk_hr until the
# river network has settled (max_days at most); independent of any design RP.

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
        rstart              = results_path("{basin_id}/spin_up/slr_{slr}/" + RST_FNAME),
        # The last chunk's map; the observation series of all chunks.
        sfincs_map_nc       = results_path("{basin_id}/spin_up/slr_{slr}/sfincs_map.nc"),
        sfincs_his          = results_path("{basin_id}/spin_up/slr_{slr}/sfincs_his.nc"),
        plot_forcing        = figure_path("20_forcing_timeseries__spinup_slr{slr}.png"),
        plot_max_depth      = figure_path("40_run_max_flood_depth__spinup_slr{slr}.png"),
        # Spin-up length (spinup_hr, the events' restart time), the steady test
        # and the river-network swing check (src.sfincs_run).
        stability           = results_path("{basin_id}/spin_up/slr_{slr}/stability.csv"),
    params:
        skeleton_root             = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        spin_up_root              = lambda wildcards: results_path(f"{wildcards.basin_id}/spin_up/slr_{wildcards.slr}"),
        resolution_m              = lambda wildcards: grid_resolution_m(wildcards.basin_id),
        tref                      = config["sfincs"]["simulation"]["tref"],
        slr_m                     = lambda wildcards: float(wildcards.slr),
        chunk_hr                  = config["sfincs"]["spinup"]["chunk_hr"],
        max_days                  = config["sfincs"]["spinup"]["max_days"],
        steady_check              = config["river_processing"]["river_depth_modelling"]["steady_check"],
        rst_fname                 = RST_FNAME,
        dtmapout_s                = config["sfincs"]["spinup"]["dtmapout_s"],
        dthisout_s                = config["sfincs"]["spinup"]["dthisout_s"],
        water_level_buffer_m      = config["sfincs"]["boundary_setup"]["water_level_buffer_m"],
        boundary_ramp_hr          = config["sfincs"]["spinup"]["boundary_ramp_hr"],
        alpha_reversal_max_m      = config["sfincs"]["alpha_reversal_max_m"],
    wildcard_constraints:
        slr = r"-?[0-9]+(\.[0-9]+)?",
    threads: rule_threads("run_spinup")
    log:
        "logs/{basin_id}/13a_run_spinup_slr{slr}.log"
    script:
        "../scripts/13a_run_spinup.py"
