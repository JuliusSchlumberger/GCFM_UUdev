rule get_boundary_forcings:
    input:
        spec_basins_meta   = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        domain_gpkg        = results_path("{basin_id}/inputs/domain/domain.gpkg"),
        spec_river_network = results_path("{basin_id}/inputs/domain/river_network.gpkg"),
        river_discharge = catalogue_path("river_discharge"),
        surge_data = catalogue_path("storm_tide_return_periods"),
        land_polygons = results_path("{basin_id}/inputs/domain/land_polygons.gpkg"),
    output:
        river_forcing = results_path("{basin_id}/inputs/forcing/river_forcing.nc"),
        surge_forcing = results_path("{basin_id}/inputs/forcing/surge_forcing.nc"),
        glofas_clip   = results_path("{basin_id}/inputs/forcing/glofas_clip.nc"),
        plot_map             = results_path("{basin_id}/inputs/plots/05a_forcing_locations.png"),
        plot_timeseries      = results_path("{basin_id}/inputs/plots/05b_forcing_timeseries.png"),
        plot_eva_diagnostics = results_path("{basin_id}/inputs/plots/05c_forcing_eva.png"),
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
        # river
        river_period_hr = config["boundary_forcings"]["river"]["period_hr"],
        river_lead_days = config["boundary_forcings"]["river"]["lead_days"],
        river_dt_hr = config["boundary_forcings"]["river"]["dt_hr"],
        glofas_buffer_deg = config["boundary_forcings"]["river"]["glofas_buffer_deg"],
        glofas_variable = config["boundary_forcings"]["river"]["glofas_variable"],
        eva = config["boundary_forcings"]["eva"],
        sfincs_resolution          = config["sfincs"]["grid"]["resolution"],
        sfincs_nr_subgridcells     = config["sfincs"]["subgrid"]["nr_subgrid_pixels"] if config["sfincs"]["subgrid"]["enabled"] == True else 0 ,
        glofas_search_radius_km    = config["boundary_forcings"]["river"]["glofas_search_radius_km"],
        glofas_min_mean_discharge  = config["boundary_forcings"]["river"]["glofas_min_mean_discharge"],
        width_column               = config["boundary_forcings"]["river"]["width_column"],
        dem_elev_filter_enabled    = config["boundary_forcings"]["river"]["dem_elevation_filter"]["enabled"],
        dem_elev_filter_buffer_m   = config["boundary_forcings"]["river"]["dem_elevation_filter"]["buffer_m"],
        dem_elev_clip_m            = config["topography"]["clip_elevation_m"][config["topography"]["source"]],
    log:
        "logs/{basin_id}/04_forcing.log"
    script:
        "../scripts/04_get_boundary_forcings.py"
