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
        protection_levels = lambda wc: (
            results_path(f"{wc.basin_id}/inputs/domain/protection_levels.json")
            if config["protection_levels"]["enabled"]
            else []
        ),
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
        # surge
        min_surge_stations = config["boundary_forcings"]["surge"]["min_stations"],
        max_surge_stations = config["boundary_forcings"]["surge"]["max_stations"],
        surge_dedupe_radius_km = config["boundary_forcings"]["surge"]["dedupe_radius_km"],
        surge_return_period = config["boundary_forcings"]["surge"]["return_period"],
        search_radii_km = config["boundary_forcings"]["surge"]["search_radii_km"],
        surge_period_hr = config["boundary_forcings"]["surge"]["period_hr"],
        surge_lead_days = config["boundary_forcings"]["surge"]["lead_days"],
        surge_dt_hr = config["boundary_forcings"]["surge"]["dt_hr"],
        mdt_variable = config["datum_correction"]["mdt_variable"],
        surge_vertical_correction = config["boundary_forcings"]["surge"]["vertical_correction"],
        surge_slr = config["boundary_forcings"]["surge"]["slr"],
        # river
        river_period_hr = config["boundary_forcings"]["river"]["period_hr"],
        river_lead_days = config["boundary_forcings"]["river"]["lead_days"],
        river_dt_hr = config["boundary_forcings"]["river"]["dt_hr"],
        glofas_buffer_deg = config["boundary_forcings"]["river"]["glofas_buffer_deg"],
        glofas_variable = config["boundary_forcings"]["river"]["glofas_variable"],
        eva = config["boundary_forcings"]["river"]["eva"],
        sfincs_resolution          = config["sfincs"]["grid"]["resolution"],
        sfincs_nr_subgridcells     = config["sfincs"]["subgrid"]["nr_subgrid_pixels"] if config["sfincs"]["subgrid"]["enabled"] == True else 0 ,
        glofas_search_radius_km    = config["boundary_forcings"]["river"]["glofas_search_radius_km"],
        glofas_min_mean_discharge  = config["boundary_forcings"]["river"]["glofas_min_mean_discharge"],
        width_column               = config["river_processing"]["width_column"],
        max_width_to_width_ratio   = config["river_processing"]["cleaning"]["max_width_to_width_ratio"],
        grdc_search_radius_km      = config["boundary_forcings"]["river"]["grdc_search_radius_km"],
        bias_correction            = config["boundary_forcings"]["river"]["bias_correction"],
        protection_levels_enabled  = config["protection_levels"]["enabled"],
    log:
        "logs/{basin_id}/07_boundary_forcings.log"
    script:
        "../scripts/07_get_boundary_forcings.py"
