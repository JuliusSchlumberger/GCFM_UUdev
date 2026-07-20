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
        # Unused (not passed to hydromt_sfincs's burn_river_rect) when
        # burn_rivers is enabled instead -- see river_burned_dem below.
        zbed_anchors      = lambda wildcards: (
            results_path(f"{wildcards.basin_id}/inputs/domain/{wildcards.basin_id}_zbed_anchors.gpkg")
            if config["river_processing"]["conditioning"]["enabled"]
            else []
        ),
        # Optional: channel-only, native-resolution burned DEM (rule
        # burn_river_dem) — only present when burn_rivers is enabled.  Fed to
        # hydromt_sfincs as a higher-priority elevation_list entry ahead of
        # elevation_merged, bypassing burn_river_rect entirely (see
        # 11b_burn_river_dem.smk for why).
        river_burned_dem  = lambda wildcards: (
            results_path(f"{wildcards.basin_id}/inputs/domain/{wildcards.basin_id}_river_burned_dem.tif")
            if config["river_processing"]["burn_rivers"]["enabled"]
            else []
        ),
        roughness         = results_path("{basin_id}/inputs/domain/{basin_id}_roughness.tif"),
        land_polygons     = results_path("{basin_id}/inputs/domain/{basin_id}_land_polygons.gpkg"),
        # Always use the estuarine-depth network (see add_estuarine_depth rule).
        river_network     = results_path("{basin_id}/inputs/domain/{basin_id}_river_network_estuarine.gpkg"),
        # Points where a non-seed, non-mouth reach crosses the delta polygon
        # outline (rule clean_river_network) -- registered as an SFINCS
        # outflow boundary (mask=3) below. Always produced (possibly empty).
        delta_outflow_points = results_path("{basin_id}/inputs/domain/{basin_id}_delta_outflow_points.gpkg"),
        zsini             = results_path("{basin_id}/inputs/domain/{basin_id}_zsini.tif"),
        surge_forcing     = results_path("{basin_id}/inputs/forcing/surge_forcing.nc"),
        river_forcing     = results_path("{basin_id}/inputs/forcing/river_forcing.nc"),
    output:
        sfincs_inp     = results_path("{basin_id}/scenarios/{scenario}/sfincs/sfincs.inp"),
        sfincs_subgrid = results_path("{basin_id}/scenarios/{scenario}/sfincs/sfincs_subgrid.nc"),
        plot_grid      = results_path("{basin_id}/scenarios/{scenario}/visuals/sfincs_build/01_grid.png"),
        plot_elevation = results_path("{basin_id}/scenarios/{scenario}/visuals/sfincs_build/02_elevation.png"),
        plot_mask      = results_path("{basin_id}/scenarios/{scenario}/visuals/sfincs_build/03_mask.png"),
        plot_roughness = results_path("{basin_id}/scenarios/{scenario}/visuals/sfincs_build/04_roughness.png"),
        **({
            "refinement_polygons": results_path("{basin_id}/scenarios/{scenario}/sfincs/{basin_id}_refinement_polygons.gpkg"),
            "plot_refinement": results_path("{basin_id}/scenarios/{scenario}/visuals/sfincs_build/01b_refinement_zones.png"),
        } if config["sfincs"]["grid"]["quadtree"]["enabled"] else {}),
    params:
        preburn_enabled    = config["river_processing"]["conditioning"]["enabled"],
        burn_rivers_enabled = config["river_processing"]["burn_rivers"]["enabled"],
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
        # forcing_mode       = config["boundary_setup"]["mode"],
        # design_rp_river_yr = config["boundary_setup"]["design_rp_river_yr"],
        forcing_mode       = lambda wildcards: scenario_params(wildcards.scenario)["mode"],
        design_rp_river_yr = lambda wildcards: scenario_params(wildcards.scenario)["river_rp"],
        design_rp_surge_yr = lambda wildcards: scenario_params(wildcards.scenario)["surge_rp"],
        compound_lag_hr    = config["boundary_setup"]["compound"]["lag_hr"],
        flat_boundary_point_spacing_m = config["boundary_setup"]["flat_boundary_point_spacing_m"],
        waterlevel_buffer_m = config["boundary_setup"]["waterlevel_buffer_m"],
        outflow_buffer_m = config["boundary_setup"]["outflow_buffer_m"],
        n_top_crossings    = config["sfincs"]["observation_points"]["n_top_crossings"],
        n_per_crossing     = config["sfincs"]["observation_points"]["n_per_crossing"],
        max_downstream_hops = config["sfincs"]["observation_points"]["max_downstream_hops"],
        inputs_dir         = lambda wildcards: results_path(f"{wildcards.basin_id}/inputs"),
        sfincs_root        = lambda wildcards: results_path(f"{wildcards.basin_id}/scenarios/{wildcards.scenario}/sfincs"),
        quadtree_enabled   = config["sfincs"]["grid"]["quadtree"]["enabled"],
        river_refinement_level     = config["sfincs"]["grid"]["quadtree"]["river_refinement_level"],
        river_buffer_factor        = config["sfincs"]["grid"]["quadtree"]["river_buffer_factor"],
        coastal_refinement_enabled = config["sfincs"]["grid"]["quadtree"]["coastal_refinement_enabled"],
        coastal_refinement_level   = config["sfincs"]["grid"]["quadtree"]["coastal_refinement_level"],
        coastal_buffer_m           = config["sfincs"]["grid"]["quadtree"]["coastal_buffer_m"],
    log:
        "logs/{basin_id}/{scenario}/13_build_sfincs.log"
    script:
        "../scripts/13_build_sfincs.py"
