rule get_boundary_forcings:
    input:
        spec_basins_meta   = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        domain_gpkg        = results_path("{basin_id}/inputs/domain/{basin_id}_domain.gpkg"),
        spec_river_network = results_path("{basin_id}/inputs/domain/{basin_id}_river_network.gpkg"),
        river_discharge = catalogue_path("river_discharge"),
        surge_data = catalogue_path("storm_tide_return_periods"),
        land_polygons = results_path("{basin_id}/inputs/domain/{basin_id}_land_polygons.gpkg"),
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
        protection_levels = results_path("{basin_id}/inputs/domain/protection_levels.json"),
    output:
        river_forcing = results_path("{basin_id}/inputs/forcing/river_forcing.nc"),
        surge_forcing = results_path("{basin_id}/inputs/forcing/surge_forcing.nc"),
        glofas_clip   = results_path("{basin_id}/inputs/forcing/glofas_clip.nc"),
        plot_map             = results_path("{basin_id}/visuals/input_data/07_forcing_locations.png"),
        plot_timeseries      = results_path("{basin_id}/visuals/input_data/07_forcing_timeseries.png"),
        plot_eva_diagnostics = results_path("{basin_id}/visuals/input_data/07_forcing_eva.png"),
        plot_bias_correction = directory(results_path("{basin_id}/visuals/input_data/07_bias_correction")),
        plot_surge_correction = results_path("{basin_id}/visuals/input_data/07_surge_correction.png"),
    params:
        # shared (surge + river forcing timeseries axis)
        lead_days = config["boundary_forcings"]["lead_days"],
        dt_hr = config["boundary_forcings"]["dt_hr"],
        # surge
        min_surge_stations = config["boundary_forcings"]["surge"]["min_stations"],
        max_surge_stations = config["boundary_forcings"]["surge"]["max_stations"],
        surge_dedupe_radius_km = config["boundary_forcings"]["surge"]["dedupe_radius_km"],
        surge_return_period = config["boundary_forcings"]["surge"]["return_period"],
        search_radii_km = config["boundary_forcings"]["surge"]["search_radii_km"],
        surge_period_hr = config["boundary_forcings"]["surge"]["period_hr"],
        mdt_fallback_search_deg = config["datum_correction"]["fallback_search_deg"],
        surge_slr = config["boundary_forcings"]["surge"]["slr"],
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
