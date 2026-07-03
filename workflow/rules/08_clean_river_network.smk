rule clean_river_network:
    input:
        spec_basins_meta   = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        domain_gpkg        = results_path("{basin_id}/inputs/domain/{basin_id}_domain.gpkg"),
        spec_river_network = results_path("{basin_id}/inputs/domain/{basin_id}_river_network.gpkg"),
        river_forcing      = results_path("{basin_id}/inputs/forcing/river_forcing.nc"),
        land_polygons           = results_path("{basin_id}/inputs/domain/{basin_id}_land_polygons.gpkg"),
        specific_basins    = results_path("{basin_id}/inputs/domain/{basin_id}_intersecting_basins.gpkg")
    output:
        clean_river_network = results_path("{basin_id}/inputs/domain/{basin_id}_river_network_clean.gpkg"),
        plot_clean_network     = results_path("{basin_id}/visuals/input_data/08_river_clean.png"),
        plot_discharge_network = results_path("{basin_id}/visuals/input_data/08_river_discharge.png"),
    params:
        discharge_variable = config["river_processing"]["flow_accumulation"]["discharge_variable"],
        flow_accumulation_iterations = config["river_processing"]["flow_accumulation"]["iterations"],
        min_width_m = config["river_processing"]["hydraulic_geometry"]["min_width_m"],
        max_width_to_width_ratio = config["river_processing"]["cleaning"]["max_width_to_width_ratio"],
        width_column = config["river_processing"]["width_column"],
    log:
        "logs/{basin_id}/08_clean_river_network.log"
    script:
        "../scripts/08_clean_river_network.py"
