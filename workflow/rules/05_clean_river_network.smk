rule clean_river_network:
    input:
        spec_basins_meta   = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        domain_gpkg        = results_path("{basin_id}/inputs/domain/domain.gpkg"),
        spec_river_network = results_path("{basin_id}/inputs/domain/river_network.gpkg"),
        river_forcing      = results_path("{basin_id}/inputs/forcing/river_forcing.nc"),
        land_polygons           = results_path("{basin_id}/inputs/domain/land_polygons.gpkg"),
        specific_basins    = results_path("{basin_id}/inputs/domain/intersecting_basins.gpkg")
    output:
        clean_river_network = results_path("{basin_id}/inputs/domain/river_network_clean.gpkg"),
        plot_clean_network     = results_path("{basin_id}/inputs/plots/06a_river_clean.png"),
        plot_discharge_network = results_path("{basin_id}/inputs/plots/06b_river_discharge.png"),
    params:
        discharge_variable = config["river_processing"]["depth"]["discharge_variable"],
        flow_accumulation_iterations = config["river_processing"]["depth"]["flow_accumulation_iterations"],
        min_width_m = config["river_processing"]["depth"]["min_width_m"],
    log:
        "logs/{basin_id}/05_river_clean.log"
    script:
        "../scripts/05_clean_river_network.py"
