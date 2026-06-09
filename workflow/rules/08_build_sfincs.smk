# Rule: build a complete SFINCS model for a single basin.
#
# The model is built in one continuous in-memory session (mode="w+") so that
# no intermediate binary files (sfincs.ind, sfincs.dep, …) need to be read
# back from disk between steps.  The script is organised in clearly labelled
# sections (grid → elevation → mask → roughness → …) that can be extended one
# at a time as the pipeline develops.
#
# HydroMT data catalog format (v1.3.1):
#   uri (not path), driver: rasterio / pyogrio, no filesystem/crs top-level fields.

rule build_sfincs:
    input:
        domain_gpkg       = results_path("{basin_id}/inputs/domain/domain.gpkg"),
        elevation_merged  = results_path("{basin_id}/inputs/domain/elevation_merged.tif"),
        roughness         = results_path("{basin_id}/inputs/domain/roughness.tif"),
        land_polygons     = results_path("{basin_id}/inputs/domain/land_polygons.gpkg"),
        river_network     = results_path("{basin_id}/inputs/river_network_processed.gpkg"),
        zsini             = results_path("{basin_id}/inputs/domain/zsini.tif"),
        surge_forcing     = results_path("{basin_id}/inputs/forcing/surge_forcing.nc"),
        river_forcing     = results_path("{basin_id}/inputs/forcing/river_forcing.nc"),
    output:
        sfincs_inp     = results_path("{basin_id}/sfincs/sfincs.inp"),
        sfincs_sbg     = results_path("{basin_id}/sfincs/sfincs.sbg"),
        plot_grid      = results_path("{basin_id}/sfincs/figs/01_grid.png"),
        plot_elevation = results_path("{basin_id}/sfincs/figs/02_elevation.png"),
        plot_mask      = results_path("{basin_id}/sfincs/figs/03_mask.png"),
        plot_roughness = results_path("{basin_id}/sfincs/figs/04_roughness.png"),
    params:
        resolution         = config["sfincs"]["grid"]["resolution"],
        include_subgrid    = config["sfincs"]["subgrid"]["enabled"],
        clip_elevation_m   = config["topography"]["clip_elevation_m"][config["topography"]["source"]],
        include_rstart     = config["sfincs"]["rstart"]["enabled"],
        spinup_days        = config["sfincs"]["rstart"]["spinup_days"],
        nr_subgrid_pixels  = config["sfincs"]["subgrid"]["nr_subgrid_pixels"],
        nr_levels          = config["sfincs"]["subgrid"]["nr_levels"],
        nrmax              = config["sfincs"]["subgrid"]["nrmax"],
        tref               = config["sfincs"]["simulation"]["tref"],
        dtmapout           = config["sfincs"]["simulation"]["dtmapout"],
        dtmaxout           = config["sfincs"]["simulation"]["dtmaxout"],
        dthisout           = config["sfincs"]["simulation"]["dthisout"],
        storevelmax        = config["sfincs"]["simulation"]["storevelmax"],
        storetwet          = config["sfincs"]["simulation"]["storetwet"],
        inputs_dir         = lambda wildcards: results_path(f"{wildcards.basin_id}/inputs"),
        sfincs_root        = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs"),
    log:
        "logs/{basin_id}/08_build_sfincs.log"
    script:
        "../scripts/08_build_sfincs.py"
