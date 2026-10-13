# Rules: storm-tide hydrograph components per basin, from the hourly GTSM
# reanalysis -- the average tide and one normalised surge shape per COAST-RP
# return period, at the basin's own surge stations. Consumed by rule
# get_boundary_forcings (07) when boundary_forcings.surge.hydrograph.enabled.
#
#   select_surge_stations   domain + COAST-RP           -> surge_stations.json
#   extract_gtsm_series     surge_stations.json + GTSM  -> gtsm_hourly.nc
#   storm_tide_hydrographs  gtsm_hourly.nc + COAST-RP   -> storm_tide_hydrographs.nc
#
# extract_gtsm_series is the expensive step (it has to read every monthly
# file of the global reanalysis, ~60 GB), so it is isolated behind
# surge_stations.json: a small text file whose content only changes when the
# basin's selected stations do. Snakemake compares the checksum of small
# inputs, so rewriting an identical surge_stations.json (a new delta-polygon
# file with this basin unchanged, a rerun of the domain rules) does NOT
# re-trigger the extraction. The GTSM folders are parameters, not inputs,
# for the same reason.

rule select_surge_stations:
    input:
        spec_basins_meta = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        surge_data       = catalogue_path("storm_tide_return_periods"),
    output:
        surge_stations = results_path("{basin_id}/preprocessing_inputs/forcing/surge_stations.json"),
    params:
        # the same four values rule get_boundary_forcings selects with
        min_surge_stations     = config["boundary_forcings"]["surge"]["min_stations"],
        max_surge_stations     = config["boundary_forcings"]["surge"]["max_stations"],
        surge_dedupe_radius_km = config["boundary_forcings"]["surge"]["dedupe_radius_km"],
        search_radii_km        = config["boundary_forcings"]["surge"]["search_radii_km"],
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
        gtsm_waterlevel_dir = catalogue_path("gtsm_total_water_level"),
        gtsm_surge_dir      = catalogue_path("gtsm_surge_residual"),
        max_match_km        = config["boundary_forcings"]["surge"]["hydrograph"]["max_match_km"],
    # Disk-bound, not CPU-bound: more readers do not help on a hard disk.
    threads: 2
    log:
        "logs/{basin_id}/07b_extract_gtsm_series.log"
    script:
        "../scripts/07b_extract_gtsm_series.py"


rule storm_tide_hydrographs:
    input:
        gtsm_hourly = results_path("{basin_id}/preprocessing_inputs/forcing/gtsm_hourly.nc"),
        surge_data  = catalogue_path("storm_tide_return_periods"),
    output:
        hydrographs = results_path("{basin_id}/preprocessing_inputs/forcing/storm_tide_hydrographs.nc"),
        plot        = results_path("{basin_id}/preprocessing_inputs/visuals/07c_storm_tide_hydrographs.png"),
    params:
        window_hr      = lambda wildcards: hydrograph_window_hr(wildcards.basin_id),
        min_events     = config["boundary_forcings"]["surge"]["hydrograph"]["min_events"],
        match_window_m = config["boundary_forcings"]["surge"]["hydrograph"]["match_window_m"],
        match_on       = config["boundary_forcings"]["surge"]["hydrograph"]["match_on"],
        max_match_km   = config["boundary_forcings"]["surge"]["hydrograph"]["max_match_km"],
    log:
        "logs/{basin_id}/07c_storm_tide_hydrographs.log"
    script:
        "../scripts/07c_storm_tide_hydrographs.py"
