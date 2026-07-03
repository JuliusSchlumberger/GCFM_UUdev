rule test_upstream_boundary:
    """
    Check that river boundary forcing points are far enough upstream from the
    river mouths to avoid interference from the surge wave propagating inland.

    Wave propagation distance = T_eff × (celerity − v_river)
    where T_eff = effective_period_fraction × surge_period_hr × 3600 s,
    celerity = sqrt(g × (depth_mouth + surge_amplitude)), and
    v_river  = max(min_river_velocity_ms, Q_bf / (W × D)).

    Diagnostic-only side branch: only needs rule 09's (add_river_depth) and
    07's (get_boundary_forcings) outputs, and nothing downstream depends on
    this rule's output -- it does NOT gate build_sfincs (13). Numbered after
    the rule-10/11 river-depth-refinement branch purely because it's
    convenient to run last among the "preprocess" validation checks, not
    because anything requires it to run after them.
    """
    input:
        spec_basins_meta        = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        domain_gpkg             = results_path("{basin_id}/inputs/domain/{basin_id}_domain.gpkg"),
        processed_river_network = results_path("{basin_id}/inputs/domain/{basin_id}_river_network_processed.gpkg"),
        river_forcing           = results_path("{basin_id}/inputs/forcing/river_forcing.nc"),
        surge_forcing           = results_path("{basin_id}/inputs/forcing/surge_forcing.nc"),
        land_polygons           = results_path("{basin_id}/inputs/domain/{basin_id}_land_polygons.gpkg"),
        spec_landuse            = results_path("{basin_id}/inputs/domain/{basin_id}_landuse.tif"),
    output:
        plot_upstream_check = results_path("{basin_id}/visuals/input_data/12_upstream_boundary_check.png"),
    params:
        effective_period_fraction = config["testing"]["upstream_boundary_check"]["effective_period_fraction"],
        min_depth_at_mouth_m      = config["testing"]["upstream_boundary_check"]["min_depth_at_mouth_m"],
        min_river_velocity_ms     = config["testing"]["upstream_boundary_check"]["min_river_velocity_ms"],
        surge_period_hr           = config["boundary_forcings"]["surge"]["period_hr"],
    log:
        "logs/{basin_id}/12_upstream_boundary_check.log"

    script:
        "../scripts/12_test_upstream_boundary.py"
