# Rule: create a SFINCS restart (spin-up) file.
#
# The main model's forcing files start with a constant lead period at bankfull
# discharge and 0 m water level, before ramping up to the event peak.  This
# spin-up rule runs SFINCS for the first `spinup_days` of that lead period and
# saves a restart file in a `spinup/` subdirectory.  When the user runs the
# main event simulation, SFINCS finds the restart file and starts from the
# already-filled river network instead of a dry initial state.
#
# SFINCS v2.3 names restart files as sfincs.YYYYMMDD.HHMMSS.rst where the
# timestamp equals trstout (= tref + spinup_days).  The filename is computed
# here from config so Snakemake knows exactly which file to expect.

from datetime import datetime, timedelta as _td

_tref       = datetime.strptime(config["sfincs"]["simulation"]["tref"], "%Y-%m-%d %H:%M:%S")
_spinup_end = _tref + _td(days=config["sfincs"]["rstart"]["spinup_days"])
RST_FNAME   = f"sfincs.{_spinup_end.strftime('%Y%m%d.%H%M%S')}.rst"


rule create_rstart:
    input:
        sfincs_inp    = results_path("{basin_id}/sfincs/sfincs.inp"),
        land_polygons = results_path("{basin_id}/inputs/domain/land_polygons.gpkg"),
    output:
        rstart              = results_path("{basin_id}/sfincs/spinup/" + RST_FNAME),
        plot_spinup         = results_path("{basin_id}/sfincs/spinup/validation_spinup.png"),
        sfincs_map_nc       = results_path("{basin_id}/sfincs/spinup/sfincs_map.nc"),
        plot_max_inundation = results_path("{basin_id}/sfincs/spinup/validation_max_inundation.png"),
    params:
        sfincs_root = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs"),
        spinup_days = config["sfincs"]["rstart"]["spinup_days"],
        sfincs_exe  = config["sfincs"]["simulation"]["sfincs_exe"],
        rst_fname   = RST_FNAME,
        dtmapout_s  = config["sfincs"]["rstart"]["dtmapout"],
    threads:
        # Tells Snakemake how many cores this job consumes so it doesn't over-schedule.
        # The same value is passed as OMP_NUM_THREADS to the SFINCS executable.
        config["sfincs"]["simulation"]["threads"]
    log:
        "logs/{basin_id}/09_create_rstart.log"
    script:
        "../scripts/09_create_rstart.py"
