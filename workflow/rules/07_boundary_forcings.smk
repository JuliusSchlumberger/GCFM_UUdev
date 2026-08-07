if scenario_params("default")["surge_rp"] is None:
    raise ValueError(
        "scenario 'default' has surge_rp=null -- rule get_boundary_forcings needs "
        "a real, tabulated surge_rp from it for its own diagnostic-preview RP "
        "(rp_level/07_surge_correction.png); give 'default' a real surge_rp."
    )

rule get_boundary_forcings:
    input:
        spec_basins_meta   = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg        = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        spec_river_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network.gpkg"),
        river_discharge = catalogue_path("river_discharge"),
        surge_data = catalogue_path("storm_tide_return_periods"),
        land_polygons = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
        grdc_data = catalogue_path("grdc_discharge"),
        mdt_data = catalogue_path("mdt_cnes_cls22"),
        slr_data = lambda wc: (
            catalogue_path("slr_ar6_regional")
            if config["boundary_forcings"]["surge"]["slr"]["enabled"]
            else []
        ),
        # Always read (rule get_protection_levels always runs/produces this)
        # -- coastal_rp_yr feeds the coastal_protection_weir crest (rule 13),
        # which always runs regardless of river_processing.empirical_estimation.modify_hydrograph;
        # only the separate riverine-side discharge correction stays gated on
        # that flag (see 07_get_boundary_forcings.py).
        protection_levels = results_path("{basin_id}/preprocessing_inputs/domain/protection_levels.json"),
    output:
        river_forcing = results_path("{basin_id}/preprocessing_inputs/forcing/river_forcing.nc"),
        surge_forcing = results_path("{basin_id}/preprocessing_inputs/forcing/surge_forcing.nc"),
        glofas_clip   = results_path("{basin_id}/preprocessing_inputs/forcing/glofas_clip.nc"),
        plot_map             = results_path("{basin_id}/preprocessing_inputs/visuals/07_forcing_locations.png"),
        plot_timeseries      = results_path("{basin_id}/preprocessing_inputs/visuals/07_forcing_timeseries.png"),
        plot_eva_diagnostics = results_path("{basin_id}/preprocessing_inputs/visuals/07_forcing_eva.png"),
        plot_bias_correction = directory(results_path("{basin_id}/preprocessing_inputs/visuals/07_bias_correction")),
        plot_surge_correction = results_path("{basin_id}/preprocessing_inputs/visuals/07_surge_correction.png"),
    params:
        # shared (surge + river forcing timeseries axis)
        lead_days = config["boundary_forcings"]["lead_days"],
        dt_hr = config["boundary_forcings"]["dt_hr"],
        # surge
        min_surge_stations = config["boundary_forcings"]["surge"]["min_stations"],
        max_surge_stations = config["boundary_forcings"]["surge"]["max_stations"],
        surge_dedupe_radius_km = config["boundary_forcings"]["surge"]["dedupe_radius_km"],
        # Diagnostic-preview RP only (rp_level/rp_level_raw columns, the
        # 07_surge_correction.png plot) -- NOT derived from whichever
        # scenario is actually requested (target_scenarios), which would
        # couple this basin-level rule's own output to the scenario axis
        # (see the river side's eva.rp_fl for the identical rationale).
        # Sourced from the "default" scenario's own surge_rp instead of a
        # separate config.yml constant, so there's a single source of truth
        # for "the representative RP" rather than two numbers that can drift
        # apart. The actual production surge boundary (rule 13) always uses
        # the ACTUAL requested scenario's own surge_rp, never this value.
        surge_return_period = scenario_params("default")["surge_rp"],
        search_radii_km = config["boundary_forcings"]["surge"]["search_radii_km"],
        surge_period_hr = config["boundary_forcings"]["surge"]["period_hr"],
        mdt_fallback_search_deg = config["datum_correction"]["fallback_search_deg"],
        # Deliberately excludes slr_m: the target global-mean SLR value must
        # NOT be a param of this rule, or changing it would bump
        # surge_forcing.nc's mtime and force rule 10's weir/depth calibration
        # and rule 13's skeleton build to rerun for no physical reason (they
        # only ever read the MDT-only baseline_m). slr_m is instead a param
        # of rule build_sfincs/run_spinup, applied at build time to
        # slr_fingerprint (see 07_get_boundary_forcings.py, src.surge).
        surge_slr = {k: v for k, v in config["boundary_forcings"]["surge"]["slr"].items() if k != "slr_m"},
        # river
        river_period_hr = config["boundary_forcings"]["river"]["period_hr"],
        glofas_buffer_deg = config["boundary_forcings"]["river"]["glofas_buffer_deg"],
        eva = config["boundary_forcings"]["river"]["eva"],
        # Diagnostic-only use (an informational "visible_on_grid" plot
        # column, doesn't gate anything -- see 07_get_boundary_forcings.py).
        # Can't use the rule 08b-computed, width-optimized resolution here:
        # rule clean_river_network (08) depends on THIS rule's river_forcing
        # output, and 08b depends on 08's bankfull_discharge_acc column --
        # 07 -> 08b -> 08 -> 07 would be a cycle. Rule build_sfincs (13, the
        # real consumer) uses the fully computed value; this just needs a
        # representative static default for an informational annotation.
        sfincs_resolution          = config["sfincs"]["grid"]["optimize_resolution"]["default_resolution_m"],
        glofas_search_radius_km    = config["boundary_forcings"]["river"]["glofas_search_radius_km"],
        glofas_min_mean_discharge  = config["boundary_forcings"]["river"]["glofas_min_mean_discharge"],
        bias_correction            = config["boundary_forcings"]["river"]["bias_correction"],
        modify_hydrograph          = config["river_processing"]["empirical_estimation"]["modify_hydrograph"],
    log:
        "logs/{basin_id}/07_boundary_forcings.log"
    script:
        "../scripts/07_get_boundary_forcings.py"
