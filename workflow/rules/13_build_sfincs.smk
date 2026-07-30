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
        # Every basin always gets a conditioned DEM (rule
        # enforce_river_monotonicity, 09).
        elevation_merged  = results_path("{basin_id}/inputs/domain/{basin_id}_elevation_conditioned.tif"),
        # Channel-only burns (native + SFINCS-grid resolution), from
        # whichever of rule empirical_depth_estimation or
        # modelled_depth_estimation (both numbered 10) actually ran
        # (sibling alternatives, same filenames -- see
        # 10_depth_estimation_empirical.py's own module docstring). This
        # directly-burned channel is the only burning mechanism -- hydromt's
        # own gdf_zb/burn_river_rect path is not used.
        river_burned_dem  = results_path("{basin_id}/inputs/domain/{basin_id}_river_burned_dem.tif"),
        river_burned_dem_sfincs_grid = results_path("{basin_id}/inputs/domain/{basin_id}_river_burned_dem_sfincs_grid.tif"),
        # SFINCS-grid-resolution conditioned DEM (rule 10's second output)
        # -- the SAME pre-computed, pixel-aligned background raster
        # calibration (rule 9b) uses directly as its own sf.elevation.create()
        # source, used here as the main grid's fallback (everywhere outside
        # the burned channel) instead of letting HydroMT reproject the
        # native-resolution elevation_conditioned.tif, so the floodplain/
        # ocean background is identical between calibration and production.
        elevation_conditioned_sfincs_grid = results_path("{basin_id}/inputs/domain/{basin_id}_elevation_conditioned_sfincs_grid.tif"),
        # Max conditioned river centerline elevation (rule 10) -- used to set
        # the active-cell mask's own elevation ceiling, see active_mask param
        # below.
        river_elevation_max = results_path("{basin_id}/inputs/domain/{basin_id}_river_elevation_max.json"),
        # Optional: modelled_depth_estimation's own final, actual traced weir (seed-head/domain-
        # edge closure, coastal-probe correction, per-cell smoothing all
        # baked in) -- only exists in modelled mode, on the regular grid
        # (see 13_build_sfincs.py's own weir section for the quadtree
        # fallback, which still re-derives from weir_crest_calibrated).
        coastal_protection_weir = lambda wildcards: (
            results_path(f"{wildcards.basin_id}/inputs/domain/{wildcards.basin_id}_coastal_protection_weir.gpkg")
            if config["river_processing"]["depth_method"] == "modelled"
            and not config["sfincs"]["grid"]["quadtree"]["enabled"]
            else []
        ),
        roughness         = results_path("{basin_id}/inputs/domain/{basin_id}_roughness.tif"),
        landuse           = results_path("{basin_id}/inputs/domain/{basin_id}_landuse.tif"),
        land_polygons     = results_path("{basin_id}/inputs/domain/{basin_id}_land_polygons.gpkg"),
        # Empirical or SFINCS-modelled (rule 10, whichever alternative
        # ran), per river_processing.depth_method -- both write the same
        # unified filename, so this input needs no mode conditional.
        river_network = results_path("{basin_id}/inputs/domain/{basin_id}_river_network_depth_estimated.gpkg"),
        # Points where a non-seed, non-mouth reach crosses the delta polygon
        # outline (rule clean_river_network) -- registered as an SFINCS
        # outflow boundary (mask=3) below. Always produced (possibly empty).
        delta_outflow_points = results_path("{basin_id}/inputs/domain/{basin_id}_delta_outflow_points.gpkg"),
        sea_mask          = results_path("{basin_id}/inputs/domain/{basin_id}_sea_mask.tif"),
        surge_forcing     = results_path("{basin_id}/inputs/forcing/surge_forcing.nc"),
        river_forcing     = results_path("{basin_id}/inputs/forcing/river_forcing.nc"),
        # Per-basin computed resolution (rule 08b) -- see resolution param below.
        grid_resolution   = results_path("{basin_id}/inputs/domain/{basin_id}_grid_resolution.json"),
    output:
        sfincs_inp     = results_path("{basin_id}/scenarios/{scenario}/sfincs/sfincs.inp"),
        sfincs_subgrid = results_path("{basin_id}/scenarios/{scenario}/sfincs/sfincs_subgrid.nc"),
        # Placeholder-touched (empty) when the weir is disabled or not
        # applicable (e.g. no ocean cells) for this basin -- same pattern
        # as sfincs_subgrid above when subgrid is disabled.
        sfincs_weir    = results_path("{basin_id}/scenarios/{scenario}/sfincs/sfincs.weir"),
        weir_gpkg      = results_path("{basin_id}/scenarios/{scenario}/sfincs/{basin_id}_coastal_protection_weir.gpkg"),
        plot_grid      = results_path("{basin_id}/scenarios/{scenario}/visuals/sfincs_build/01_grid.png"),
        plot_elevation = results_path("{basin_id}/scenarios/{scenario}/visuals/sfincs_build/02_elevation.png"),
        plot_mask      = results_path("{basin_id}/scenarios/{scenario}/visuals/sfincs_build/03_mask.png"),
        plot_roughness = results_path("{basin_id}/scenarios/{scenario}/visuals/sfincs_build/04_roughness.png"),
        plot_coastal_protection_weir = results_path("{basin_id}/scenarios/{scenario}/visuals/sfincs_build/07_coastal_protection_weir.png"),
        **({
            "refinement_polygons": results_path("{basin_id}/scenarios/{scenario}/sfincs/{basin_id}_refinement_polygons.gpkg"),
            "plot_refinement": results_path("{basin_id}/scenarios/{scenario}/visuals/sfincs_build/01b_refinement_zones.png"),
        } if config["sfincs"]["grid"]["quadtree"]["enabled"] else {}),
    params:
        depth_method       = config["river_processing"]["depth_method"],
        resolution         = lambda wildcards, input: json.load(open(input.grid_resolution))["resolution"],
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
        # "river_only" flat boundary level: set BELOW terrain.gebco_max_depth_m
        # (the deepest any clamped ocean bed cell can be), so the boundary is
        # guaranteed dry everywhere -- the model's behavior is then driven
        # entirely by river discharge, not by any water entering from the
        # coastal boundary. The extra 0.5 m is a safety margin below the clamp.
        river_only_flat_level_m = -(config["terrain"]["gebco_max_depth_m"] + 0.5),
        # Resolved per-scenario (see scenario_params in 00_common.smk):
        # "default" replays config.yml's own sfincs.boundary_setup/surge
        # settings; named scenarios (config/scenarios.yml) override mode
        # to "compound" and set their own surge_rp/river_rp.
        forcing_mode       = lambda wildcards: scenario_params(wildcards.scenario)["mode"],
        design_rp_river_yr = lambda wildcards: scenario_params(wildcards.scenario)["river_rp"],
        design_rp_surge_yr = lambda wildcards: scenario_params(wildcards.scenario)["surge_rp"],
        compound_lag_hr    = config["sfincs"]["boundary_setup"]["compound"]["lag_hr"],
        flat_boundary_point_spacing_m = config["sfincs"]["boundary_setup"]["flat_boundary_point_spacing_m"],
        waterlevel_buffer_m = config["sfincs"]["boundary_setup"]["waterlevel_buffer_m"],
        outflow_buffer_m = config["sfincs"]["boundary_setup"]["outflow_buffer_m"],
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
        active_mask_enabled        = config["sfincs"]["grid"]["active_mask"]["enabled"],
        active_mask_elevation_buffer_m = config["sfincs"]["grid"]["active_mask"]["elevation_buffer_m"],
        # Only used by this rule's own quadtree fallback (regular grid +
        # modelled imports rule modelled_depth_estimation's own final weir
        # directly, no re-derivation) -- see river_depth_modelling's own header comment
        # in config.yml for why these live there.
        min_component_cells       = config["river_processing"]["river_depth_modelling"]["min_component_cells"],
        weir_par1                  = config["river_processing"]["river_depth_modelling"]["weir_par1"],
        weir_crest_junction_blend_m = config["river_processing"]["river_depth_modelling"]["weir_crest_junction_blend_m"],
        weir_freeboard_m           = config["river_processing"]["river_depth_modelling"]["freeboard_m"],
        river_crest_dilation_cells = config["river_processing"]["river_depth_modelling"]["river_crest_dilation_cells"],
    log:
        "logs/{basin_id}/{scenario}/13_build_sfincs.log"
    script:
        "../scripts/13_build_sfincs.py"
