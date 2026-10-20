if scenario_params("default")["surge_rp_yr"] is None:
    raise ValueError(
        "scenario 'default' has surge_rp_yr=null -- rule get_boundary_forcings needs "
        "a real, tabulated surge_rp_yr from it for its own diagnostic-preview RP "
        "(storm_tide_at_rp_m/07_surge_correction.png); give 'default' a real surge_rp_yr."
    )

rule get_boundary_forcings:
    input:
        spec_basins_meta   = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg        = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        spec_river_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network.gpkg"),
        river_discharge = catalogue_path("river_discharge"),
        surge_data = catalogue_path("storm_tide_return_periods"),
        surge_stations = results_path("{basin_id}/preprocessing_inputs/forcing/surge_stations.json"),
        # Tide + surge shapes per return period at the same surge stations.
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
        # surge_protection_rp_yr feeds the coastal_protection_weir crest (rule 13).
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
        max_width_fill_m = config["river_processing"]["width_qa"]["max_width_fill_m"],
        # shared surge + river timeseries axis
        lead_days = config["boundary_forcings"]["lead_days"],
        dt_hr = config["boundary_forcings"]["dt_hr"],
        # surge
        # Diagnostic-preview RP only (storm_tide_at_rp_m/storm_tide_at_rp_raw_m, 07_surge_correction.png),
        # taken from the "default" scenario so this basin-level rule stays off
        # the scenario axis; rule 13 uses the requested scenario's surge_rp_yr.
        surge_rp_yr = scenario_params("default")["surge_rp_yr"],
        surge_period_hr = config["boundary_forcings"]["surge"]["period_hr"],
        # Only the keys read here: the block also holds window_overrides_hr, and
        # passing it whole would rerun every basin when one basin's window changes.
        surge_hydrograph = {
            k: config["boundary_forcings"]["surge"]["hydrograph"][k]
            for k in ("enabled", "ramp_hr", "max_match_km")
        },
        # Same half-window rule storm_tide_hydrographs built the shapes with.
        hydrograph_window_hr = lambda wildcards: hydrograph_window_hr(wildcards.basin_id),
        mdt_fallback_search_deg = config["datum_correction"]["fallback_search_deg"],
        # slr_m is not a param here: changing it would rerun rule 10's
        # calibration and the skeleton build, which only read the MDT-only
        # calm_sea_level_m. slr_m is instead a per-scenario param of rule
        # build_sfincs/run_spinup, applied at build time to slr_fingerprint.
        surge_slr = config["boundary_forcings"]["surge"]["slr"],
        # river
        river_period_hr = config["boundary_forcings"]["river"]["period_hr"],
        glofas_buffer_deg = config["boundary_forcings"]["river"]["glofas_buffer_deg"],
        eva = config["boundary_forcings"]["river"]["eva"],
        # Diagnostic only: the "visible_on_grid" plot column.
        sfincs_resolution_m        = lambda wildcards: grid_resolution_m(wildcards.basin_id),
        glofas_search_radius_km    = config["boundary_forcings"]["river"]["glofas_search_radius_km"],
        glofas_min_mean_discharge_m3s = config["boundary_forcings"]["river"]["glofas_min_mean_discharge_m3s"],
        bias_correction            = config["boundary_forcings"]["river"]["bias_correction"],
        # Preview plot only; the scenario's river event is built in rule build_sfincs.
        river_event                = config["boundary_forcings"]["river"]["event_hydrograph"],
    log:
        "logs/{basin_id}/07_boundary_forcings.log"
    script:
        "../scripts/07_get_boundary_forcings.py"
