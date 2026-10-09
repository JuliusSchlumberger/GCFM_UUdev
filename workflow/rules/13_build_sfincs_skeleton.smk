# Rule: build the scenario-INDEPENDENT part of a basin's SFINCS model --
# grid, elevation, mask (active + waterlevel + outflow boundary), coastal
# protection weir, roughness, subgrid table, observation points. Runs ONCE
# per basin, not once per scenario (no {scenario} wildcard) -- none of these
# sections depend on forcing_mode/design_rp_river_yr/design_rp_surge_yr
# (confirmed by tracing every one of their own inputs back to basin-level,
# RP-independent sources: the basin's fixed grid resolution, the conditioned/burned DEM
# rasters, river_network_depth_estimated.gpkg's own bankfull_discharge/
# has_glofas columns, etc.).
#
# Split out of what used to be one monolithic build_sfincs rule so that
# changing a scenario's own RP (surge_rp/river_rp in config/scenarios.yml)
# does not force this expensive HydroMT build to redo -- rule build_sfincs
# (13_build_sfincs.smk) now LOADS this rule's own output in read mode,
# redirects subsequent writes to its own scenario directory
# (sf.root.set(...)), and writes ONLY the forcing-specific components on
# top, referencing this rule's geometry files via relative paths in its own
# hand-crafted sfincs.inp -- never re-writing/duplicating them. Rule
# run_spinup (14_run_spinup.smk) does the same, for its own forcing (RP=1 river, calm sea).
#
# zsini (the model's real initial-condition GRID LAYER) is built HERE (a
# pure function of baseline_m + a sea/land classification, neither of which
# varies by scenario) even though whether it's actually USED depends on
# forcing_mode (see build_sfincs's own initial-conditions section) --
# computing it once here and letting every consumer reference the same
# layer is simpler than recomputing an identical raster per scenario.
# zsini_sea_cells_on_grid.tif (rule modelled_depth_estimation) is written
# directly at THIS rule's own grid resolution (pixel-identical: both models
# are created from the shared grid definition, sfincs_grid.json) and read
# directly into an in-memory DataArray with ZERO further reprojection, so
# every raster the landuse/sea/roughness classification chain touches gets
# resampled onto the SFINCS grid exactly once, upstream (rule
# grid_align_landuse, 09b). A native-resolution round-trip was separately
# confirmed (before this classification chain existed) to leave roughly half of the
# corrected fringe cells still wet (two independent nearest-neighbor
# implementations don't invert each other cleanly, even on a confirmed
# pixel-identical grid) -- the same risk this now avoids everywhere, not
# just for the weir-protected fringe.

rule build_sfincs_skeleton:
    input:
        domain_gpkg       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        # THE grid definition (rule build_sfincs_grid, 08c) this model is
        # created from -- see src/grid.py.
        sfincs_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_sfincs_grid.json"),
        # Native-resolution CONDITIONED (post-monotonicity, rule
        # enforce_river_monotonicity) elevation -- NOT the raw merged DEM
        # (rule 05a's own {basin_id}_elevation_merged.tif). Named
        # "elevation_merged"/"local_elevation_merged" until 2026-08-04 even
        # though it always pointed at elevation_conditioned.tif -- a stale
        # name left over from before rule 09 existed, fixed for clarity
        # (no behavior change: this was already the conditioned file).
        elevation_conditioned = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_conditioned.tif"),
        # The burned river channel (rule modelled_depth_estimation, 10): on
        # the subgrid pixel grid for sf.subgrid.create() (no resampling --
        # the script raises if it is not exactly this model's own subgrid
        # pixel grid), and its per-cell mean for the main-grid "dep".
        river_burned_subgrid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_burned_subgrid.tif"),
        river_burned_dem_sfincs_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_burned_dem_sfincs_grid.tif"),
        elevation_conditioned_sfincs_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_conditioned_sfincs_grid.tif"),
        river_elevation_max = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_elevation_max.json"),
        coastal_protection_weir = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_coastal_protection_weir.gpkg"),
        # Native resolution -- subgrid table only (needs sub-cell detail,
        # same as elevation). The main regular-grid "manning" field uses
        # roughness_on_grid below instead.
        roughness         = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_roughness.tif"),
        # Grid-aligned land mask (rule grid_align_landuse) -- plot
        # background only; the water-level boundary uses landuse_on_grid
        # itself (see src.raster.restrict_waterlevel_boundary_to_sea).
        land_mask_on_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
        river_network     = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_depth_estimated.gpkg"),
        delta_outflow_points = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_outflow_points.gpkg"),
        # Already rasterized directly onto this model's own grid (rule
        # grid_align_landuse, 09b) -- read with zero further reprojection
        # for the main "manning" field and the weir diagnostics section.
        landuse_on_grid   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_on_grid.tif"),
        roughness_on_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_roughness_on_grid.tif"),
        # zsini's ONLY source (rule modelled_depth_estimation, 10) --
        # written directly at this rule's OWN grid resolution (confirmed
        # pixel-identical: same domain_gpkg + grid resolution feed both),
        # read directly into an in-memory zsini DataArray with ZERO further
        # reprojection. There used to be a native-resolution path here
        # (sea_mask_corrected.tif + HydroMT's own reproject_like) -- removed
        # 2026-08-07: a second independent nearest-neighbor pass over an
        # already-resampled coarse mask does not invert cleanly even on a
        # confirmed pixel-identical grid (~53% of corrected fringe cells
        # were still wet under the old two-step version).
        zsini_sea_cells   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
        # Only for baseline_m/coastal_protection_crest_m (zsini + weir crest
        # floor, both basin-level fixed fields) and bankfull_discharge/
        # has_glofas (observation point placement) -- NOT for building any
        # actual forcing timeseries, which stays entirely in build_sfincs/
        # run_spinup.
        surge_forcing     = results_path("{basin_id}/preprocessing_inputs/forcing/surge_forcing.nc"),
        river_forcing     = results_path("{basin_id}/preprocessing_inputs/forcing/river_forcing.nc"),
    output:
        sfincs_inp     = results_path("{basin_id}/sfincs_skeleton/sfincs.inp"),
        sfincs_subgrid = results_path("{basin_id}/sfincs_skeleton/sfincs_subgrid.nc"),
        sfincs_weir    = results_path("{basin_id}/sfincs_skeleton/sfincs.weir"),
        weir_gpkg      = results_path("{basin_id}/sfincs_skeleton/{basin_id}_coastal_protection_weir.gpkg"),
        zsini          = results_path("{basin_id}/sfincs_skeleton/zsini.tif"),
        plot_grid      = results_path("{basin_id}/preprocessing_inputs/visuals/sfincs_build/01_grid.png"),
        plot_elevation = results_path("{basin_id}/preprocessing_inputs/visuals/sfincs_build/02_elevation.png"),
        plot_mask      = results_path("{basin_id}/preprocessing_inputs/visuals/sfincs_build/03_mask.png"),
        plot_roughness = results_path("{basin_id}/preprocessing_inputs/visuals/sfincs_build/04_roughness.png"),
        plot_coastal_protection_weir = results_path("{basin_id}/preprocessing_inputs/visuals/sfincs_build/07_coastal_protection_weir.png"),
    params:
        resolution         = lambda wildcards: grid_resolution_m(wildcards.basin_id),
        nr_subgrid_pixels  = config["sfincs"]["subgrid"]["nr_subgrid_pixels"],
        nr_levels          = config["sfincs"]["subgrid"]["nr_levels"],
        nrmax              = config["sfincs"]["subgrid"]["nrmax"],
        inputs_dir         = lambda wildcards: results_path(f"{wildcards.basin_id}/preprocessing_inputs"),
        skeleton_root      = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        active_mask_enabled        = config["sfincs"]["grid"]["active_mask"]["enabled"],
        active_mask_elevation_buffer_m = config["sfincs"]["grid"]["active_mask"]["elevation_buffer_m"],
        outflow_buffer_m  = config["sfincs"]["boundary_setup"]["outflow_buffer_m"],
        n_top_crossings    = config["sfincs"]["observation_points"]["n_top_crossings"],
        n_per_crossing     = config["sfincs"]["observation_points"]["n_per_crossing"],
        max_downstream_hops = config["sfincs"]["observation_points"]["max_downstream_hops"],
    log:
        "logs/{basin_id}/13_build_sfincs_skeleton.log"
    script:
        "../scripts/13_build_sfincs_skeleton.py"
