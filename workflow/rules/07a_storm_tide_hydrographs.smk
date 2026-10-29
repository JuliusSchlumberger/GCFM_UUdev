# Rules: storm-tide hydrograph components per basin, from the hourly GTSM
# reanalysis -- the average tide and one normalised surge shape per COAST-RP
# return period, at the basin's own surge stations. Consumed by rule
# get_boundary_forcings (07d) when boundary_forcings.surge.hydrograph.enabled.
#
#   select_surge_stations   domain + COAST-RP                  -> surge_stations.json
#   extract_gtsm_series     surge_stations.json + GTSM subset  -> gtsm_hourly.nc
#   storm_tide_hydrographs  gtsm_hourly.nc + COAST-RP          -> storm_tide_hydrographs.nc
#
# The GTSM subset holds every basin's stations, read once from the global
# monthly reanalysis by tools/subset_gtsm_stations.py. It is a param, not an
# input: rebuilding it for a new basin must not rerun the other basins.

rule select_surge_stations:
    input:
        spec_basins_meta = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        surge_data       = raw_input("storm_tide_return_periods"),
    output:
        surge_stations = results_path("{basin_id}/preprocessing_inputs/forcing/surge_stations.json"),
    params:
        min_surge_stations     = config["boundary_forcings"]["surge"]["min_stations"],
        max_surge_stations     = config["boundary_forcings"]["surge"]["max_stations"],
        surge_dedupe_radius_km = config["boundary_forcings"]["surge"]["dedupe_radius_km"],
        search_radii_km        = config["boundary_forcings"]["surge"]["search_radii_km"],
    threads: rule_threads("select_surge_stations")
    log:
        "logs/{basin_id}/07a_select_surge_stations.log"
    script:
        "../scripts/07a_select_surge_stations.py"


rule extract_gtsm_series:
    input:
        surge_stations = results_path("{basin_id}/preprocessing_inputs/forcing/surge_stations.json"),
    output:
        gtsm_hourly = results_path("{basin_id}/preprocessing_inputs/forcing/gtsm_hourly.nc"),
    params:
        gtsm_station_subset = catalogue_path("gtsm_station_subset"),
        max_match_km        = config["boundary_forcings"]["surge"]["hydrograph"]["max_match_km"],
    threads: rule_threads("extract_gtsm_series")
    log:
        "logs/{basin_id}/07b_extract_gtsm_series.log"
    script:
        "../scripts/07b_extract_gtsm_series.py"


rule storm_tide_hydrographs:
    input:
        gtsm_hourly = results_path("{basin_id}/preprocessing_inputs/forcing/gtsm_hourly.nc"),
        surge_data  = raw_input("storm_tide_return_periods"),
    output:
        hydrographs = results_path("{basin_id}/preprocessing_inputs/forcing/storm_tide_hydrographs.nc"),
        **analysis_outputs(plot=analysis_figure_path("07_forcing_storm_tide_shapes.png")),
    params:
        window_hr      = lambda wildcards: hydrograph_window_hr(wildcards.basin_id),
        min_events     = config["boundary_forcings"]["surge"]["hydrograph"]["min_events"],
        match_window_m = config["boundary_forcings"]["surge"]["hydrograph"]["match_window_m"],
        match_on       = config["boundary_forcings"]["surge"]["hydrograph"]["match_on"],
        max_match_km   = config["boundary_forcings"]["surge"]["hydrograph"]["max_match_km"],
    threads: rule_threads("storm_tide_hydrographs")
    log:
        "logs/{basin_id}/07c_storm_tide_hydrographs.log"
    script:
        "../scripts/07c_storm_tide_hydrographs.py"
