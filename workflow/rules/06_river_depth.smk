rule add_river_depth:
    input:
        spec_basins_meta    = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        domain_gpkg         = results_path("{basin_id}/inputs/domain/domain.gpkg"),
        clean_river_network = results_path("{basin_id}/inputs/domain/river_network_clean.gpkg"),
        land_polygons            = results_path("{basin_id}/inputs/domain/land_polygons.gpkg"),
    output:
        processed_river_network = results_path("{basin_id}/inputs/river_network_processed.gpkg"),
        plot_river_depth                   = results_path("{basin_id}/inputs/plots/07a_river_depth.png"),
        plot_hydraulic_relations           = results_path("{basin_id}/inputs/plots/07b_river_hydraulics.png"),
        plot_river_network_width_discharge = results_path("{basin_id}/inputs/plots/07c_river_width_q.png"),
        plot_longitudinal_profile          = results_path("{basin_id}/inputs/plots/07d_river_profile.png"),
    params:
        discharge_threshold = config["river_processing"]["depth"]["discharge_threshold"],
        min_width_m = config["river_processing"]["depth"]["min_width_m"],
        hg_a = config["river_processing"]["depth"]["hydraulic_geometry"]["a"],
        hg_b = config["river_processing"]["depth"]["hydraulic_geometry"]["b"],
        hg_c = config["river_processing"]["depth"]["hydraulic_geometry"]["c"],
        hg_f = config["river_processing"]["depth"]["hydraulic_geometry"]["f"],
    log:
        "logs/{basin_id}/06_river_depth.log"
    script:
        "../scripts/06_add_river_depth.py"
