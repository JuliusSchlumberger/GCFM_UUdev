rule test_upstream_boundary:
    """
    Check that river boundary forcing points are far enough upstream from the
    river mouths to avoid interference from the surge wave propagating inland.

    Final distance = min(kinematic distance, attenuation distance):
      kinematic distance   = T_eff × (celerity − v_river), a time-bounded
        estimate of how far the wave front could travel in T_eff.
      attenuation distance = how far upstream (marching reach-by-reach,
        following the mainstem at confluences) until the surge amplitude
        decays below amplitude_threshold_fraction of its own starting value,
        via friction-based exponential decay (Lorentz-linearised, channel
        Manning's n from channel_manning_n).
    Each mouth uses its OWN nearest CoastRP station's rp_level as surge
    amplitude (not the domain-wide max). depth_mouth (rivdph) and v_river
    (Q_bf / (W × D)) are used as calculated, with no config floor; a mouth
    with no usable calculated depth or velocity is skipped.

    Diagnostic-only side branch: only needs rule 09b's (add_estuarine_depth)
    and 07's (get_boundary_forcings) outputs, and nothing downstream depends
    on this rule's output -- it does NOT gate build_sfincs (13). Numbered
    after the rule-10/11 river-depth-refinement branch purely because it's
    convenient to run last among the "preprocess" validation checks, not
    because anything requires it to run after them.

    Uses river_network_estuarine.gpkg (rule 09b's output, the FINAL
    hybrid-blended rivdph), not river_network_processed.gpkg (rule 09a's
    power-law-only rivdph) -- matches the network 13_build_sfincs.py and
    rule burn_river_bed actually use. Using the wrong one previously gave
    wildly inconsistent mouth depths (e.g. 1.89-17.80 m across 5 mouths of
    the same delta, vs. ~5.04 m for all of them once correctly estuarine
    -blended).
    """
    input:
        spec_basins_meta        = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        domain_gpkg             = results_path("{basin_id}/inputs/domain/{basin_id}_domain.gpkg"),
        river_network_estuarine = results_path("{basin_id}/inputs/domain/{basin_id}_river_network_estuarine.gpkg"),
        river_forcing           = results_path("{basin_id}/inputs/forcing/river_forcing.nc"),
        surge_forcing           = results_path("{basin_id}/inputs/forcing/surge_forcing.nc"),
        land_polygons           = results_path("{basin_id}/inputs/domain/{basin_id}_land_polygons.gpkg"),
        spec_landuse            = results_path("{basin_id}/inputs/domain/{basin_id}_landuse.tif"),
    output:
        plot_upstream_check = results_path("{basin_id}/visuals/input_data/12_upstream_boundary_check.png"),
    params:
        effective_period_fraction    = config["testing"]["upstream_boundary_check"]["effective_period_fraction"],
        channel_manning_n            = config["testing"]["upstream_boundary_check"]["channel_manning_n"],
        amplitude_threshold_fraction = config["testing"]["upstream_boundary_check"]["amplitude_threshold_fraction"],
        surge_period_hr              = config["boundary_forcings"]["surge"]["period_hr"],
    log:
        "logs/{basin_id}/12_upstream_boundary_check.log"

    script:
        "../scripts/12_test_upstream_boundary.py"


rule test_bifurcation_calibration_options:
    """
    Compares discharge partitioning at every bifurcation between the
    ORIGINAL and MODIFIED SWORD river networks, rebuilt directly from raw
    SWORD (mirrors rules get_river_network (06) + clean_river_network (08)
    rather than reading their already-built output) so both sources can be
    compared side by side. One figure per bifurcation, 1 or 2 panels
    depending on whether the manual width corrections actually touched that
    bifurcation's neighborhood -- see 12b_bifurcation_calibration_options.py
    for the full comparison methodology.

    Diagnostic-only side branch: only needs rule 07's river_forcing.nc (for
    seed crossings/discharges) and the raw SWORD sources themselves (read
    directly from the data catalogue, not rule 06's output) -- nothing
    downstream depends on this rule's output, and it does NOT gate
    build_sfincs (13).
    """
    input:
        spec_basins_meta       = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        domain_gpkg            = results_path("{basin_id}/inputs/domain/{basin_id}_domain.gpkg"),
        delta_polygon          = results_path("{basin_id}/inputs/domain/{basin_id}_delta_polygon.gpkg"),
        river_forcing          = results_path("{basin_id}/inputs/forcing/river_forcing.nc"),
        river_network_original = catalogue_path("river_network_original"),
        river_network          = catalogue_path("river_network"),
    output:
        plot_dir = directory(results_path("{basin_id}/visuals/bifurcation_calibration_options")),
    params:
        n_iterations       = config["river_processing"]["flow_accumulation"]["iterations"],
        min_width_m        = config["river_processing"]["hydraulic_geometry"]["min_width_m"],
        discharge_variable = config["river_processing"]["flow_accumulation"]["discharge_variable"],
    log:
        "logs/{basin_id}/12b_bifurcation_calibration_options.log"
    script:
        "../scripts/12b_bifurcation_calibration_options.py"
