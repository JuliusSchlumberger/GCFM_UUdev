rule clean_river_network:
    input:
        spec_basins_meta   = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg        = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        spec_river_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network.gpkg"),
        delta_polygon      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_polygon.gpkg"),
        river_forcing      = results_path("{basin_id}/preprocessing_inputs/forcing/river_forcing.nc"),
        land_polygons      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
    output:
        clean_river_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
        delta_outflow_points = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_outflow_points.gpkg"),
        **analysis_outputs(
            plot_clean_network     = analysis_figure_path("08_river_network_clean.png"),
            plot_discharge_network = analysis_figure_path("08_river_discharge.png"),
        ),
    params:
        discharge_variable = config["river_processing"]["flow_accumulation"]["discharge_variable"],
        max_width_fill_m   = config["river_processing"]["width_qa"]["max_width_fill_m"],
        canal_routing      = config["river_processing"]["canal_routing"],
    threads: rule_threads("clean_river_network")
    log:
        "logs/{basin_id}/08_clean_river_network.log"
    script:
        "../scripts/08_clean_river_network.py"
