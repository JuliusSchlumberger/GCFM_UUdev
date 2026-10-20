# Rule: write a scenario's forcing (sfincs.bzs/.bnd/.dis/.src) and sfincs.inp on
# top of the basin skeleton. Depends on the skeleton's sfincs.inp so geometry
# changes rerun it; changing a scenario's RP reruns neither the skeleton nor
# run_spinup.

rule build_sfincs:
    input:
        skeleton_inp      = results_path("{basin_id}/sfincs_skeleton/sfincs.inp"),
        river_network     = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_depth_estimated.gpkg"),
        surge_forcing     = results_path("{basin_id}/preprocessing_inputs/forcing/surge_forcing.nc"),
        river_forcing     = results_path("{basin_id}/preprocessing_inputs/forcing/river_forcing.nc"),
        # Only when spin-up is enabled.
        rstart = lambda wildcards: (
            results_path(f"{wildcards.basin_id}/spin_up/" + RST_FNAME)
            if config["sfincs"]["spinup"]["enabled"] else []
        ),
    output:
        sfincs_inp = results_path("{basin_id}/runs/{scenario}/sfincs/sfincs.inp"),
    params:
        resolution         = lambda wildcards: grid_resolution_m(wildcards.basin_id),
        tref               = config["sfincs"]["simulation"]["tref"],
        dtmapout           = config["sfincs"]["simulation"]["dtmapout"],
        dtmaxout           = config["sfincs"]["simulation"]["dtmaxout"],
        dthisout           = config["sfincs"]["simulation"]["dthisout"],
        storevelmax        = config["sfincs"]["simulation"]["storevelmax"],
        storetwet          = config["sfincs"]["simulation"]["storetwet"],
        include_rstart     = config["sfincs"]["spinup"]["enabled"],
        spinup_days        = config["sfincs"]["spinup"]["spinup_days"],
        rst_fname          = RST_FNAME,
        # river_only boundary 0.5 m below terrain.gebco_max_depth_m (the deepest
        # clamped bed), so the coastal boundary is always dry.
        river_only_flat_level_m = -(config["terrain"]["gebco_max_depth_m"] + 0.5),
        # Derived from which of the scenario's surge_rp/river_rp are set.
        forcing_mode       = lambda wildcards: scenario_params(wildcards.scenario)["mode"],
        design_rp_river_yr = lambda wildcards: scenario_params(wildcards.scenario)["river_rp"],
        design_rp_surge_yr = lambda wildcards: scenario_params(wildcards.scenario)["surge_rp"],
        compound_lag_hr    = config["sfincs"]["boundary_setup"]["compound"]["lag_hr"],
        # Per-scenario discharge scaling (default 1.0), applied here only, so
        # changing it reruns just this scenario's build and event run.
        discharge_multiplier = lambda wildcards: scenario_params(wildcards.scenario)["discharge_multiplier"],
        # Real GloFAS flood event or synthetic wave; changing it only rebuilds scenarios.
        river_event        = config["boundary_forcings"]["river"]["event_hydrograph"],
        # Target global-mean SLR (m) × surge_forcing.nc's dimensionless
        # slr_fingerprint; per-scenario and applied here only, like discharge_multiplier.
        slr_enabled        = config["boundary_forcings"]["surge"]["slr"]["enabled"],
        slr_m              = lambda wildcards: scenario_params(wildcards.scenario)["slr_m"],
        # SLR ramps in over the event's first hours (the shared spin-up has none).
        slr_ramp_hours     = config["sfincs"]["boundary_setup"]["slr_ramp_hours"],
        flat_boundary_point_spacing_m = config["sfincs"]["boundary_setup"]["flat_boundary_point_spacing_m"],
        waterlevel_buffer_m = config["sfincs"]["boundary_setup"]["waterlevel_buffer_m"],
        skeleton_root      = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        sfincs_root        = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{wildcards.scenario}/sfincs"),
        spin_up_root       = lambda wildcards: results_path(f"{wildcards.basin_id}/spin_up"),
    log:
        "logs/{basin_id}/runs/{scenario}/13_build_sfincs.log"
    script:
        "../scripts/13_build_sfincs.py"
