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
        # The basin's own tide + surge shapes per return period (rule
        # storm_tide_hydrographs, 07a_storm_tide_hydrographs.smk), at the
        # same surge stations this rule selects.
        storm_tide_hydrographs = lambda wc: (
            results_path(f"{wc.basin_id}/preprocessing_inputs/forcing/storm_tide_hydrographs.nc")
            if config["boundary_forcings"]["surge"]["hydrograph"]["enabled"]
            else []
        ),
        land_polygons = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
        grdc_data = catalogue_path("grdc_discharge"),
        mdt_data = catalogue_path("mdt_cnes_cls22"),
        slr_data = lambda wc: (
            catalogue_path("slr_ar6_regional")
            if config["boundary_forcings"]["surge"]["slr"]["enabled"]
            else []
        ),
        # Always read (rule get_protection_levels always runs/produces this)
        # -- coastal_rp_yr feeds the coastal_protection_weir crest (rule 13).
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
        # Only the keys this rule reads -- NOT the whole block: it also holds
        # window_overrides_hr, so passing it whole would rerun this rule (and
        # every rule below it) for EVERY basin whenever one basin's window
        # changes.
        surge_hydrograph = {
            k: config["boundary_forcings"]["surge"]["hydrograph"][k]
            for k in ("enabled", "ramp_hours", "max_match_km")
        },
        # This basin's own event half-window (default or per-basin override,
        # see hydrograph_window_hr in 00_common.smk) -- the same value rule
        # storm_tide_hydrographs built the shapes with.
        hydrograph_window_hr = lambda wildcards: hydrograph_window_hr(wildcards.basin_id),
        mdt_fallback_search_deg = config["datum_correction"]["fallback_search_deg"],
        # slr_m (the target global-mean SLR value) deliberately lives in
        # config/scenarios.yml, not here: it must NOT be a param of this
        # rule, or changing it would bump surge_forcing.nc's mtime and force
        # rule 10's weir/depth calibration and rule 13's skeleton build to
        # rerun for no physical reason (they only ever read the MDT-only
        # baseline_m). slr_m is instead a per-scenario param of rule
        # build_sfincs/run_spinup, applied at build time to slr_fingerprint
        # (see 07_get_boundary_forcings.py, src.surge, scenario_params in
        # 00_common.smk).
        surge_slr = config["boundary_forcings"]["surge"]["slr"],
        # river
        river_period_hr = config["boundary_forcings"]["river"]["period_hr"],
        glofas_buffer_deg = config["boundary_forcings"]["river"]["glofas_buffer_deg"],
        eva = config["boundary_forcings"]["river"]["eva"],
        # Diagnostic-only use (an informational "visible_on_grid" plot
        # column, doesn't gate anything -- see 07_get_boundary_forcings.py).
        sfincs_resolution          = lambda wildcards: grid_resolution_m(wildcards.basin_id),
        glofas_search_radius_km    = config["boundary_forcings"]["river"]["glofas_search_radius_km"],
        glofas_min_mean_discharge  = config["boundary_forcings"]["river"]["glofas_min_mean_discharge"],
        bias_correction            = config["boundary_forcings"]["river"]["bias_correction"],
        # Only for this rule's own preview plot (07_forcing_timeseries.png) --
        # the scenario's real river event is built in rule build_sfincs.
        river_event                = config["boundary_forcings"]["river"]["event_hydrograph"],
    log:
        "logs/{basin_id}/07_boundary_forcings.log"
    script:
        "../scripts/07_get_boundary_forcings.py"
