rule test_upstream_boundary:
    """
    Check that river boundary forcing points are far enough upstream from the
    river mouths to avoid interference from the surge wave propagating inland.

    Wave propagation distance = T_eff × (celerity − v_river)
    where T_eff = effective_period_fraction × surge_period_hr × 3600 s,
    celerity = sqrt(g × (depth_mouth + surge_amplitude)), and
    v_river  = max(min_river_velocity_ms, Q_bf / (W × D)).
    """
    input:
        spec_basins_meta        = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        domain_gpkg             = results_path("{basin_id}/inputs/domain/domain.gpkg"),
        processed_river_network = results_path("{basin_id}/inputs/river_network_processed.gpkg"),
        river_forcing           = results_path("{basin_id}/inputs/forcing/river_forcing.nc"),
        surge_forcing           = results_path("{basin_id}/inputs/forcing/surge_forcing.nc"),
        land_polygons           = results_path("{basin_id}/inputs/domain/land_polygons.gpkg"),
    output:
        plot_upstream_check = results_path("{basin_id}/inputs/plots/08b_upstream_boundary_check.png"),
    params:
        effective_period_fraction = config["boundary_forcings"]["upstream_boundary_check"]["effective_period_fraction"],
        min_depth_at_mouth_m      = config["boundary_forcings"]["upstream_boundary_check"]["min_depth_at_mouth_m"],
        min_river_velocity_ms     = config["boundary_forcings"]["upstream_boundary_check"]["min_river_velocity_ms"],
        surge_period_hr           = config["boundary_forcings"]["surge"]["period_hr"],
    log:
        "logs/{basin_id}/07_upstream_boundary_check.log"

    script:
        "../scripts/07_test_upstream_boundary.py"


rule test_discharge_comparison:
    """Compare seed-reach bankfull discharge against the Lin et al. rivers_ge30m Q2."""
    input:
        spec_basins_meta    = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        domain_gpkg         = results_path("{basin_id}/inputs/domain/domain.gpkg"),
        clean_river_network = results_path("{basin_id}/inputs/domain/river_network_clean.gpkg"),
        river_forcing       = results_path("{basin_id}/inputs/forcing/river_forcing.nc"),
        glofas_clip         = results_path("{basin_id}/inputs/forcing/glofas_clip.nc"),
        rivers_lin          = testing_catalogue_path("rivers_ge30m"),
    output:
        plot_discharge_comparison = results_path("{basin_id}/inputs/plots/08a_discharge_check.png"),
    params:
        lin_layer         = "rivers_ge30m",
        buffer_deg        = 0.5,
        min_stream_order  = config["testing"]["min_stream_order"],
        max_match_dist_km       = config["testing"]["max_match_dist_km"],
        glofas_search_radius_km = config["testing"]["glofas_search_radius_km"],
        glofas_variable   = config["boundary_forcings"]["river"]["glofas_variable"],
        eva               = config["boundary_forcings"]["eva"],
    log:
        "logs/{basin_id}/07_discharge_check.log"
    script:
        "../scripts/07_test_discharge_comparison.py"
