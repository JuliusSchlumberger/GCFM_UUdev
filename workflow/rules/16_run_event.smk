# Rule: run the main flood-event SFINCS simulation and its sanity checks.
#
# Unlike rule 14 (run_spinup), which writes its OWN shortened sfincs.inp into
# a spinup/ subdirectory, this rule runs the MAIN sfincs.inp that rule 13
# (build_sfincs) already wrote directly into sfincs_root -- it is already
# configured with tstart = spin-up end, tstop = end of the full forcing
# timeseries, and rstfile pointing at rule 14's restart file (see rule 13's
# "Phase 2 config" section). This rule only executes it, then produces the
# same kind of sanity-check diagnostics as rule 15 (sanity_checks) -- but for
# this event run's own sfincs_map.nc rather than the spin-up's -- plus a
# per-timestep flooded-area/flood-volume CSV.

rule run_event:
    input:
        sfincs_inp           = results_path("{basin_id}/scenarios/{scenario}/sfincs/sfincs.inp"),
        rstart               = results_path("{basin_id}/scenarios/{scenario}/sfincs/spinup/" + RST_FNAME),
        land_polygons        = results_path("{basin_id}/inputs/domain/{basin_id}_land_polygons.gpkg"),
        landuse              = results_path("{basin_id}/inputs/domain/{basin_id}_landuse.tif"),
        domain_gpkg          = results_path("{basin_id}/inputs/domain/{basin_id}_domain.gpkg"),
        clean_river_network  = results_path("{basin_id}/inputs/domain/{basin_id}_river_network_clean.gpkg"),
    output:
        sfincs_map_nc              = results_path("{basin_id}/scenarios/{scenario}/sfincs/sfincs_map.nc"),
        plot_inundation_ratio      = results_path("{basin_id}/scenarios/{scenario}/visuals/01_inundation_ratio.png"),
        animation_flood_progress   = results_path("{basin_id}/scenarios/{scenario}/visuals/02_flood_animation.mp4"),
        flood_timeseries_csv       = results_path("{basin_id}/scenarios/{scenario}/visuals/flood_timeseries.csv"),
    params:
        sfincs_root                = lambda wildcards: results_path(f"{wildcards.basin_id}/scenarios/{wildcards.scenario}/sfincs"),
        sfincs_exe                 = config["sfincs"]["simulation"]["sfincs_exe"],
        timeout_s                  = config["sfincs"]["simulation"]["timeout_s"],
        min_inundation_depth_m     = config["sfincs"]["sanity_checks"]["min_inundation_depth_m"],
        include_subgrid            = config["sfincs"]["subgrid"]["enabled"],
        animation_fps              = config["sfincs"]["sanity_checks"]["animation_fps"],
    log:
        "logs/{basin_id}/scenarios/{scenario}/16_run_event.log"
    script:
        "../scripts/16_run_event.py"
