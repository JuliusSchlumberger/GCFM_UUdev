# Rule: run a short SFINCS spin-up simulation to produce a restart file.
#
# This is where the SFINCS solver is actually EXECUTED for the first time in
# the pipeline (rule 13, build_sfincs, only assembles config/input files --
# it never runs the executable). The main model's forcing files start with a
# constant lead period at bankfull discharge and 0 m water level, before
# ramping up to the event peak.  This rule runs SFINCS for the first
# `spinup_days` of that lead period and saves a restart file in a `spinup/`
# subdirectory.  When the user runs the main event simulation, SFINCS finds
# the restart file and starts from the already-filled river network instead
# of a dry initial state.
#
# SFINCS v2.3 names restart files as sfincs.YYYYMMDD.HHMMSS.rst where the
# timestamp equals trstout (= tref + spinup_days).  The filename is computed
# here from config so Snakemake knows exactly which file to expect.

from datetime import datetime, timedelta as _td

_tref       = datetime.strptime(config["sfincs"]["simulation"]["tref"], "%Y-%m-%d %H:%M:%S")
_spinup_end = _tref + _td(days=config["sfincs"]["spinup"]["spinup_days"])
RST_FNAME   = f"sfincs.{_spinup_end.strftime('%Y%m%d.%H%M%S')}.rst"


rule run_spinup:
    input:
        sfincs_inp           = results_path("{basin_id}/sfincs/sfincs.inp"),
        land_polygons        = results_path("{basin_id}/inputs/domain/{basin_id}_land_polygons.gpkg"),
        landuse              = results_path("{basin_id}/inputs/domain/{basin_id}_landuse.tif"),
        domain_gpkg          = results_path("{basin_id}/inputs/domain/{basin_id}_domain.gpkg"),
        clean_river_network  = results_path("{basin_id}/inputs/domain/{basin_id}_river_network_clean.gpkg"),
    output:
        rstart              = results_path("{basin_id}/sfincs/spinup/" + RST_FNAME),
        sfincs_map_nc       = results_path("{basin_id}/sfincs/spinup/sfincs_map.nc"),
        plot_spinup         = results_path("{basin_id}/visuals/model_runs/spinup/validation_spinup.png"),
        plot_max_inundation = results_path("{basin_id}/visuals/model_runs/spinup/validation_max_inundation.png"),
    params:
        sfincs_root               = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs"),
        spinup_days               = config["sfincs"]["spinup"]["spinup_days"],
        sfincs_exe                = config["sfincs"]["simulation"]["sfincs_exe"],
        rst_fname                 = RST_FNAME,
        dtmapout_s                = config["sfincs"]["spinup"]["dtmapout_s"],
        dthisout_s                 = config["sfincs"]["spinup"]["dthisout_s"],
        velocity_animation_enabled = config["sfincs"]["sanity_checks"]["velocity_animation"]["enabled"],
        include_subgrid            = config["sfincs"]["subgrid"]["enabled"],
    log:
        "logs/{basin_id}/14_run_spinup.log"
    script:
        "../scripts/14_run_spinup.py"
