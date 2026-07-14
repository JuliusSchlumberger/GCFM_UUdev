rule add_estuarine_depth:
    """
    ALWAYS runs immediately after add_river_depth (09a) -- this is an
    unconditional sequence, not a fork: 09a/09b together produce
    river_network_estuarine.gpkg, the one "final" network every downstream
    rule uses, and this rule's hydraulic-relations plot, which can show the
    estuarine/fluvial/blend depth breakdown.

    Rule 10 (condition_elevation) is the actual independent branch -- it
    depends on 09a's output too, but not on this rule's. Rule 11
    (burn_river_bed) is the merge point that needs both this rule's
    estuarine-adjusted network and rule 10's conditioned elevation.

    When river_processing.estuarine_depth.enabled = false: output is a
    straight copy of river_network_processed.gpkg and the plot shows
    power-law depths only (same visual content as rule 09a's own depth plot).

    When enabled: the Nienhuis et al. (2018) tidal parameters are used to
    apply the Leuven et al. (2018) depth model for reaches with
    dist_out <= L_e, with a linear blend around the tidal limit, and the
    plot gains per-reach depth-source markers plus a correlation subplot.
    """
    input:
        processed_river_network = results_path("{basin_id}/inputs/domain/{basin_id}_river_network_processed.gpkg"),
        delta_polygon           = results_path("{basin_id}/inputs/domain/{basin_id}_delta_polygon.gpkg"),
        nienhuis                = catalogue_path("nienhuis_delta_characteristics"),
    output:
        estuarine_river_network  = results_path("{basin_id}/inputs/domain/{basin_id}_river_network_estuarine.gpkg"),
        plot_hydraulic_relations = results_path("{basin_id}/visuals/input_data/09b_river_hydraulics.png"),
    params:
        enabled              = config["river_processing"]["estuarine_depth"]["enabled"],
        max_match_dist_km    = config["river_processing"]["estuarine_depth"]["max_match_dist_km"],
        obrien_C             = config["river_processing"]["estuarine_depth"]["obrien_C"],
        obrien_alpha         = config["river_processing"]["estuarine_depth"]["obrien_alpha"],
        convergence_ratio_k  = config["river_processing"]["estuarine_depth"]["convergence_ratio_k"],
        blend_fraction       = config["river_processing"]["estuarine_depth"]["blend_fraction"],
        min_depth_m          = config["river_processing"]["estuarine_depth"]["min_depth_m"],
    log:
        "logs/{basin_id}/09b_estuarine_depth.log"
    script:
        "../scripts/09b_estuarine_depth.py"
