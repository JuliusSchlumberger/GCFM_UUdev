rule add_river_depth:
    """
    Power-law (Leopold-Maddock) hydraulic depth. Rule 09b (add_estuarine_depth)
    ALWAYS runs immediately after this rule and blends in estuarine depth near
    the coast -- 09a/09b together form one sequential depth-estimation chain,
    not a fork. Rule 10 (condition_elevation) is the actual independent branch,
    parallel to the 09a->09b chain; both feed rule 11 (burn_river_bed).

    TODO: bifurcation discharge splitting (accumulate_discharge in
    src/river_network.py) weights each downstream arm by width * angle_factor.
    This has not been sensitivity-tested against alternatives -- see the note
    in config.yml's river_processing.hydraulic_geometry section.
    """
    input:
        spec_basins_meta    = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        domain_gpkg         = results_path("{basin_id}/inputs/domain/{basin_id}_domain.gpkg"),
        clean_river_network = results_path("{basin_id}/inputs/domain/{basin_id}_river_network_clean.gpkg"),
        land_polygons            = results_path("{basin_id}/inputs/domain/{basin_id}_land_polygons.gpkg"),
    output:
        processed_river_network = results_path("{basin_id}/inputs/domain/{basin_id}_river_network_processed.gpkg"),
        plot_river_depth                   = results_path("{basin_id}/visuals/input_data/09a_river_depth.png"),
        plot_river_network_width_discharge = results_path("{basin_id}/visuals/input_data/09a_river_width_q.png"),
    params:
        discharge_threshold = config["river_processing"]["hydraulic_geometry"]["discharge_threshold"],
        min_width_m = config["river_processing"]["hydraulic_geometry"]["min_width_m"],
        hg_a = config["river_processing"]["hydraulic_geometry"]["a"],
        hg_b = config["river_processing"]["hydraulic_geometry"]["b"],
        hg_c = config["river_processing"]["hydraulic_geometry"]["c"],
        hg_f = config["river_processing"]["hydraulic_geometry"]["f"],
    log:
        "logs/{basin_id}/09a_river_depth.log"
    script:
        "../scripts/09a_add_river_depth.py"
