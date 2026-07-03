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
        domain_gpkg       = results_path("{basin_id}/inputs/domain/{basin_id}_domain.gpkg"),
        # Priority: conditioned > merged.
        # When conditioning is enabled the DEM is NOT modified by rule 11; instead
        # rule 11 produces zbed_anchors.gpkg which is fed to SFINCS as gdf_zb.
        elevation_merged  = lambda wildcards: results_path(
            f"{wildcards.basin_id}/inputs/domain/{wildcards.basin_id}_elevation_conditioned.tif"
            if config["river_processing"]["conditioning"]["enabled"]
            else f"{wildcards.basin_id}/inputs/domain/{wildcards.basin_id}_elevation_merged.tif"
        ),
        # Optional: zbed anchor points — only present when conditioning enabled.
        # Using an empty list [] as the disabled fallback (Snakemake convention
        # for optional inputs that feed a lambda in the script).
        zbed_anchors      = lambda wildcards: (
            results_path(f"{wildcards.basin_id}/inputs/domain/{wildcards.basin_id}_zbed_anchors.gpkg")
            if config["river_processing"]["conditioning"]["enabled"]
            else []
        ),
        roughness         = results_path("{basin_id}/inputs/domain/{basin_id}_roughness.tif"),
        land_polygons     = results_path("{basin_id}/inputs/domain/{basin_id}_land_polygons.gpkg"),
        # Always use the estuarine-depth network (see add_estuarine_depth rule).
        river_network     = results_path("{basin_id}/inputs/domain/{basin_id}_river_network_estuarine.gpkg"),
        zsini             = results_path("{basin_id}/inputs/domain/{basin_id}_zsini.tif"),
        surge_forcing     = results_path("{basin_id}/inputs/forcing/surge_forcing.nc"),
        river_forcing     = results_path("{basin_id}/inputs/forcing/river_forcing.nc"),
    output:
        sfincs_inp     = results_path("{basin_id}/sfincs/sfincs.inp"),
        sfincs_subgrid = results_path("{basin_id}/sfincs/sfincs_subgrid.nc"),
        plot_grid      = results_path("{basin_id}/visuals/sfincs_build/01_grid.png"),
        plot_elevation = results_path("{basin_id}/visuals/sfincs_build/02_elevation.png"),
        plot_mask      = results_path("{basin_id}/visuals/sfincs_build/03_mask.png"),
        plot_roughness = results_path("{basin_id}/visuals/sfincs_build/04_roughness.png"),
        **({
            "refinement_polygons": results_path("{basin_id}/sfincs/{basin_id}_refinement_polygons.gpkg"),
            "plot_refinement": results_path("{basin_id}/visuals/sfincs_build/01b_refinement_zones.png"),
        } if config["sfincs"]["grid"]["quadtree"]["enabled"] else {}),
    params:
        preburn_enabled    = config["river_processing"]["conditioning"]["enabled"],
        resolution         = config["sfincs"]["grid"]["resolution"],
        include_subgrid    = config["sfincs"]["subgrid"]["enabled"],
        include_rstart     = config["sfincs"]["spinup"]["enabled"],
        spinup_days        = config["sfincs"]["spinup"]["spinup_days"],
        nr_subgrid_pixels  = config["sfincs"]["subgrid"]["nr_subgrid_pixels"],
        nr_levels          = config["sfincs"]["subgrid"]["nr_levels"],
        nrmax              = config["sfincs"]["subgrid"]["nrmax"],
        tref               = config["sfincs"]["simulation"]["tref"],
        dtmapout           = config["sfincs"]["simulation"]["dtmapout"],
        dtmaxout           = config["sfincs"]["simulation"]["dtmaxout"],
        dthisout           = config["sfincs"]["simulation"]["dthisout"],
        storevelmax        = config["sfincs"]["simulation"]["storevelmax"],
        storetwet          = config["sfincs"]["simulation"]["storetwet"],
        velocity_animation_enabled = config["sfincs"]["sanity_checks"]["velocity_animation"]["enabled"],
        forcing_mode       = config["boundary_setup"]["mode"],
        compound_lag_hr    = config["boundary_setup"]["compound"]["lag_hr"],
        flat_boundary_point_spacing_m = config["boundary_setup"]["flat_boundary_point_spacing_m"],
        waterlevel_buffer_m = config["boundary_setup"]["waterlevel_buffer_m"],
        n_top_crossings    = config["sfincs"]["observation_points"]["n_top_crossings"],
        n_per_crossing     = config["sfincs"]["observation_points"]["n_per_crossing"],
        max_downstream_hops = config["sfincs"]["observation_points"]["max_downstream_hops"],
        inputs_dir         = lambda wildcards: results_path(f"{wildcards.basin_id}/inputs"),
        sfincs_root        = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs"),
        quadtree_enabled   = config["sfincs"]["grid"]["quadtree"]["enabled"],
        river_refinement_level     = config["sfincs"]["grid"]["quadtree"]["river_refinement_level"],
        river_buffer_factor        = config["sfincs"]["grid"]["quadtree"]["river_buffer_factor"],
        coastal_refinement_enabled = config["sfincs"]["grid"]["quadtree"]["coastal_refinement_enabled"],
        coastal_refinement_level   = config["sfincs"]["grid"]["quadtree"]["coastal_refinement_level"],
        coastal_buffer_m           = config["sfincs"]["grid"]["quadtree"]["coastal_buffer_m"],
    log:
        "logs/{basin_id}/13_build_sfincs.log"
    script:
        "../scripts/13_build_sfincs.py"
