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
            # Native resolution -- subgrid table only (needs sub-cell
            # detail, same as elevation). The main regular-grid "manning"
            # field and this rule's own zsini/weir/ocean_mask sections all
            # use roughness_on_grid/landuse_on_grid below instead (rule
            # grid_align_landuse, 09b) -- see that rule's own module
            # docstring for why every consumer of the landuse/roughness
            # classification now shares one canonical coarse-grid resample.
            roughness                = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_roughness.tif"),
            # Grid-aligned land mask (rule grid_align_landuse) -- plot
            # background only; the water-level boundary uses landuse_on_grid
            # itself (see src.raster.restrict_waterlevel_boundary_to_sea).
            land_mask_on_grid        = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
            domain_gpkg              = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
            spec_basins_meta         = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
            # Already rasterized directly onto this rule's own grid (rule
            # grid_align_landuse, 09b) -- read with zero further
            # reprojection for the weir/ocean_mask/zsini sections below.
            landuse_on_grid          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_on_grid.tif"),
            roughness_on_grid        = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_roughness_on_grid.tif"),
            # Real, steady coastal baseline (mean sea level + SLR/MDT
            # correction) for the water-level boundary, see
            # 10_depth_estimation_modelled.py.
            surge_forcing            = results_path("{basin_id}/preprocessing_inputs/forcing/surge_forcing.nc"),
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
            # weir_crest_calibrated columns. Burned from round 0's own
            # per-cell excavation depth (the same excavation rounds 1 and 2
            # simulated with).
            river_burned_dem              = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_burned_dem.tif"),
            river_burned_dem_sfincs_grid  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_burned_dem_sfincs_grid.tif"),
            # Modelled-mode-only production input -- the traced weir with
            # every edge's own crest from round 1's water level on its own
            # water side (the same weir round 2 verified), not re-derivable
            # from any per-reach scalar. See rule 13's own import of this
            # file.
            coastal_protection_weir       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_coastal_protection_weir.gpkg"),
            # THE single sea/land classification every downstream consumer
            # reads directly, zero further reprojection: this rule's own
            # zsini (13_build_sfincs_skeleton.py) AND flood-diagnostic
            # consumers downstream (rules 14/15/16/17, via
            # src.postprocessing's own sea_mask_path argument -- those
            # reproject this coarse, grid-aligned raster onto their own
            # subgrid-/cell-resolution output grid, which is well-defined
            # since subgrid is an exact integer subdivision of THIS grid,
            # sharing its origin/axes). Written directly at THIS RULE'S OWN
            # GRID resolution (no reprojection at all -- grid.transform/
            # grid.crs as-is): 13_build_sfincs_skeleton.py's own
            # sf.grid.data["dep"] is confirmed pixel-identical to this
            # rule's own calibration grid (same mmax/nmax/dx/dy/x0/y0, both
            # "auto-UTM"-fit from the same domain_gpkg + grid_resolution.
            # json), so that script builds its ini DataArray in-memory
            # directly on this file's own grid and passes it to
            # sf.initial_conditions.create() with ZERO resampling
            # (reproject_like on an identical grid is a confirmed true
            # no-op). There used to be a SEPARATE, native-resolution
            # sea_mask_corrected.tif just for the flood-diagnostic
            # consumers -- removed 2026-08-07b as pure duplication of this
            # same boolean (see this rule's own script for the full
            # rationale). Same unified filename rule empirical_depth_
            # estimation writes -- downstream rules never need
            # depth_method-conditional file selection.
            zsini_sea_cells               = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
            # Canonical outputs -- copies of round 2's (verification) own
            # numbered files below, for downstream consumers/convention that
            # expect these fixed filenames.
            plot_calibration             = results_path("{basin_id}/preprocessing_inputs/visuals/10_calibration/river_depth.png"),
            plot_water_level_timeseries  = results_path("{basin_id}/preprocessing_inputs/visuals/10_calibration/water_level_timeseries.png"),
            plot_max_inundation          = results_path("{basin_id}/preprocessing_inputs/visuals/10_calibration/max_inundation.png"),
            animation_flood_progress     = results_path("{basin_id}/preprocessing_inputs/visuals/10_calibration/flood_animation.mp4"),
            plot_crest_gap_map           = results_path("{basin_id}/preprocessing_inputs/visuals/10_calibration/crest_gap_map.png"),
            # Coastal dike position with unprotected_ocean_wetlands off vs
            # on, overlaid (whichever the config sets is the one simulated).
            plot_dike_positions          = results_path("{basin_id}/preprocessing_inputs/visuals/10_calibration/dike_positions_wetland_switch.png"),
            # Per-round diagnostics (round 0 = confined/un-excavated, round
            # 1 = confined/excavated, round 2 = verification with the real
            # crests) are not declared here (same convention as
            # calibration_state.csv, written directly under calib_root) --
            # 10_depth_estimation_modelled.py writes them to its own
            # round_visuals_dir; round 2's files are what the canonical
            # outputs above are copied from.
        params:
            calib_root  = lambda wildcards: results_path(f"{wildcards.basin_id}/preprocessing_inputs/depth_crest_calibration"),
            sfincs_exe  = config["sfincs"]["simulation"]["sfincs_exe"],
            timeout_s   = config["sfincs"]["simulation"]["timeout_s"],
            active_mask_enabled = config["sfincs"]["grid"]["active_mask"]["enabled"],
            active_mask_elevation_buffer_m = config["sfincs"]["grid"]["active_mask"]["elevation_buffer_m"],
            outflow_buffer_m = config["sfincs"]["boundary_setup"]["outflow_buffer_m"],
            # Same surge-station matching radius production uses (rule 13/14)
            # -- rounds 1-2 force the coastal protection storm tide at those
            # stations.
            waterlevel_buffer_m = config["sfincs"]["boundary_setup"]["waterlevel_buffer_m"],
            flow_accumulation_iterations = config["river_processing"]["flow_accumulation"]["iterations"],
            # Power-law comparison column (rivdph_powerlaw) -- computed inline,
            # not read from rule empirical_depth_estimation's output.
            hg_c = config["river_processing"]["empirical_estimation"]["hydraulic_geometry"]["c"],
            hg_f = config["river_processing"]["empirical_estimation"]["hydraulic_geometry"]["f"],
            calibration_days             = config["river_processing"]["river_depth_modelling"]["calibration_days"],
            discharge_ramp_hours          = config["river_processing"]["river_depth_modelling"]["discharge_ramp_hours"],
            weir_crest_m                  = config["river_processing"]["river_depth_modelling"]["weir_crest_m"],
            weir_par1                     = config["river_processing"]["river_depth_modelling"]["weir_par1"],
            excavation_fraction           = config["river_processing"]["river_depth_modelling"]["excavation_fraction"],
            weir_freeboard_m              = config["river_processing"]["river_depth_modelling"]["freeboard_m"],
            unprotected_ocean_wetlands    = config["river_processing"]["river_depth_modelling"]["unprotected_ocean_wetlands"],
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
