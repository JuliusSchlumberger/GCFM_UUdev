# Rule: SFINCS-based river depth calibration -- an alternative to the
# empirical hydraulic-geometry depth estimate (rule empirical_depth_estimation,
# also numbered 10), selected via river_processing.depth_method == "modelled".
#
# Sibling alternative to rule empirical_depth_estimation -- exactly
# one of the two is ever DEFINED (guarded by the module-level `if` below,
# not just skipped at runtime), both writing the same unified output
# filename (river_network_depth_estimated.gpkg). Defining both
# unconditionally would make Snakemake raise AmbiguousRuleException for
# that shared output file.
#
# Runs after enforce_river_monotonicity (09): needs the *conditioned*
# elevation as the (un-excavated) calibration channel bed -- both the
# native-resolution version (subgrid's own fine sub-cell source, unchanged)
# and the shared SFINCS-grid-resolution version (rule enforce_river_monotonicity's
# second output, this rule's own sf.elevation.create() base layer, identical
# to what rule 13's production build uses). See 10_depth_estimation_modelled.py's
# module docstring for the full design.

if config["river_processing"]["depth_method"] == "modelled":

    rule modelled_depth_estimation:
        input:
            elevation_conditioned = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_conditioned.tif"),
            elevation_conditioned_sfincs_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_conditioned_sfincs_grid.tif"),
            # Max conditioned river centerline elevation (rule
            # enforce_river_monotonicity, 09) -- used to set the active-cell
            # mask's own elevation ceiling, matching rule 13's production
            # build (see active_mask params below).
            river_elevation_max = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_elevation_max.json"),
            clean_river_network      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
            river_forcing           = results_path("{basin_id}/preprocessing_inputs/forcing/river_forcing.nc"),
            protection_levels        = results_path("{basin_id}/preprocessing_inputs/domain/protection_levels.json"),
            grid_resolution          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_grid_resolution.json"),
            roughness                = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_roughness.tif"),
            land_polygons            = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
            domain_gpkg              = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
            spec_basins_meta         = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
            landuse                  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse.tif"),
            # Real, steady coastal baseline (mean sea level + SLR/MDT
            # correction) for the water-level boundary, see
            # 10_depth_estimation_modelled.py.
            surge_forcing            = results_path("{basin_id}/preprocessing_inputs/forcing/surge_forcing.nc"),
            # Sea/land classification (rule get_landuse, 05b) -- builds this
            # rule's own spatially-varying zsini (sea cells start at
            # baseline_m, matching 13_build_sfincs.py's own zsini.tif) rather
            # than a uniform dry start that produces a transient "boundary
            # flooding in" spike at coastal/mouth cells.
            sea_mask                 = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_sea_mask.tif"),
            # Points where a non-seed, non-mouth, non-bifurcation reach
            # crosses the delta polygon's own outline (rule clean_river_network) --
            # a genuine place flow exits the modelled network without
            # reaching a real coastal mouth (e.g. a distributary clipped by
            # the domain boundary in a complex, multi-channel delta).
            # Registered as a free outflow boundary below, matching rule
            # 13's own production build exactly -- always present (possibly
            # empty).
            delta_outflow_points     = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_outflow_points.gpkg"),
        output:
            depth_estimated_river_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_depth_estimated.gpkg"),
            # Same filenames rule empirical_depth_estimation writes -- see this
            # rule's own module docstring: rule 13 imports whichever sibling ran,
            # never re-derives its own burn from the flattened rivdph/
            # weir_crest_calibrated columns. Written from the converged
            # round's own already-computed burn (both resolutions already
            # exist internally every round; this just persists the
            # converged round's own copies under the canonical name).
            river_burned_dem              = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_burned_dem.tif"),
            river_burned_dem_sfincs_grid  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_burned_dem_sfincs_grid.tif"),
            # Modelled-mode-only production input -- the converged round's
            # own actual traced weir (seed-head/domain-edge closure,
            # coastal-probe correction, per-cell smoothing all baked in),
            # not re-derivable from any per-reach scalar. See rule 13's own
            # import of this file.
            coastal_protection_weir       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_coastal_protection_weir.gpkg"),
            # Canonical outputs -- copies of the LAST round's own numbered
            # files below (round n_correction_iterations), for downstream
            # consumers/convention that expect these fixed filenames.
            plot_calibration             = results_path("{basin_id}/preprocessing_inputs/visuals/10_calibration/river_depth.png"),
            plot_water_level_timeseries  = results_path("{basin_id}/preprocessing_inputs/visuals/10_calibration/water_level_timeseries.png"),
            plot_max_inundation          = results_path("{basin_id}/preprocessing_inputs/visuals/10_calibration/max_inundation.png"),
            animation_flood_progress     = results_path("{basin_id}/preprocessing_inputs/visuals/10_calibration/flood_animation.mp4"),
            plot_crest_gap_map           = results_path("{basin_id}/preprocessing_inputs/visuals/10_calibration/crest_gap_map.png"),
            # Per-round diagnostics (round 0 = isolated calibration, rounds
            # 1..n_correction_iterations = coupled-system corrections) are
            # DELIBERATELY NOT declared here (same convention as
            # calibration_state.csv, written directly under calib_root) --
            # early-stopping means not every round index actually runs, and
            # a round that never simulated should have NO file at all, not
            # an empty placeholder. Snakemake requires every DECLARED output
            # to exist after the rule finishes, which would force an empty
            # touch() for every skipped round if these were declared;
            # writing them directly (10_depth_estimation_modelled.py's own
            # round_visuals_dir) lets a skipped round have nothing, and the
            # final round's own real files are what the canonical outputs
            # above are copied from.
        params:
            calib_root  = lambda wildcards: results_path(f"{wildcards.basin_id}/preprocessing_inputs/depth_crest_calibration"),
            sfincs_exe  = config["sfincs"]["simulation"]["sfincs_exe"],
            timeout_s   = config["sfincs"]["simulation"]["timeout_s"],
            active_mask_enabled = config["sfincs"]["grid"]["active_mask"]["enabled"],
            active_mask_elevation_buffer_m = config["sfincs"]["grid"]["active_mask"]["elevation_buffer_m"],
            outflow_buffer_m = config["sfincs"]["boundary_setup"]["outflow_buffer_m"],
            flow_accumulation_iterations = config["river_processing"]["flow_accumulation"]["iterations"],
            # Power-law comparison column (rivdph_powerlaw) -- computed inline,
            # not read from rule empirical_depth_estimation's output.
            hg_c = config["river_processing"]["empirical_estimation"]["hydraulic_geometry"]["c"],
            hg_f = config["river_processing"]["empirical_estimation"]["hydraulic_geometry"]["f"],
            calibration_days             = config["river_processing"]["river_depth_modelling"]["calibration_days"],
            discharge_ramp_hours          = config["river_processing"]["river_depth_modelling"]["discharge_ramp_hours"],
            weir_crest_m                  = config["river_processing"]["river_depth_modelling"]["weir_crest_m"],
            weir_par1                     = config["river_processing"]["river_depth_modelling"]["weir_par1"],
            weir_crest_fraction           = config["river_processing"]["river_depth_modelling"]["weir_crest_fraction"],
            n_correction_iterations       = config["river_processing"]["river_depth_modelling"]["n_correction_iterations"],
            min_crest_increment_per_round_m = config["river_processing"]["river_depth_modelling"]["min_crest_increment_per_round_m"],
            weir_freeboard_m              = config["river_processing"]["river_depth_modelling"]["freeboard_m"],
            river_crest_dilation_cells    = config["river_processing"]["river_depth_modelling"]["river_crest_dilation_cells"],
            min_component_cells           = config["river_processing"]["river_depth_modelling"]["min_component_cells"],
            animation_fps                 = config["sfincs"]["sanity_checks"]["animation_fps"],
            # Same subgrid setup as the production model (rule 13/14), so
            # calibration measures depth against the same subgrid-resolved
            # bathymetry the production model actually experiences.
            include_subgrid    = config["sfincs"]["subgrid"]["enabled"],
            nr_subgrid_pixels  = config["sfincs"]["subgrid"]["nr_subgrid_pixels"],
            nr_levels          = config["sfincs"]["subgrid"]["nr_levels"],
            nrmax              = config["sfincs"]["subgrid"]["nrmax"],
        # Claims the whole --cores budget -- see rule run_spinup's (14) threads
        # comment for the full rationale (SFINCS itself can multi-thread, but
        # Snakemake can't parallelize within one run, so give it the whole
        # machine rather than let other jobs compete with it for CPU).
        threads: workflow.cores
        log:
            "logs/{basin_id}/10_depth_estimation_modelled.log"
        script:
            "../scripts/10_depth_estimation_modelled.py"
