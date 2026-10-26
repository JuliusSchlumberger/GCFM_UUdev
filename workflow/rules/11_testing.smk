rule test_upstream_boundary:
    """
    Check that river forcing points lie beyond the inland reach of the surge wave.

    Distance = min(kinematic T_eff × (c − v_river), attenuation): the upstream
    distance, along the mainstem, until Lorentz-linearised friction decay
    (channel_manning_n) brings the amplitude below amplitude_threshold_fraction.
    Each mouth uses its own nearest CoastRP station's storm_tide_at_rp_m; river_depth_m and
    Q_bf / (W × D) are used as calculated, no floor. Diagnostic only; does not
    gate build_sfincs.
    """
    input:
        spec_basins_meta        = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg             = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        river_network_depth_estimated = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_depth_estimated.gpkg"),
        river_forcing           = results_path("{basin_id}/preprocessing_inputs/forcing/river_forcing.nc"),
        surge_forcing           = results_path("{basin_id}/preprocessing_inputs/forcing/surge_forcing.nc"),
        # Plot background.
        land_mask_on_grid       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
    output:
        plot_upstream_check = results_path("{basin_id}/preprocessing_inputs/visuals/11_upstream_boundary_check.png"),
    params:
        effective_period_fraction    = config["testing"]["upstream_boundary_check"]["effective_period_fraction"],
        channel_manning_n            = config["river_processing"]["channel_manning_n"],
        amplitude_threshold_fraction = config["testing"]["upstream_boundary_check"]["amplitude_threshold_fraction"],
        surge_rp_yr                  = config["testing"]["upstream_boundary_check"]["surge_rp_yr"],
        surge_period_hr              = config["boundary_forcings"]["surge"]["period_hr"],
    threads: rule_threads("test_upstream_boundary")
    log:
        "logs/{basin_id}/11_upstream_boundary_check.log"

    script:
        "../scripts/11_test_upstream_boundary.py"


rule test_bifurcation_calibration_options:
    """
    Plot bifurcation discharge partitioning without vs with the manual SWORD widths.

    Rebuilds both networks from raw SWORD as rules 06 + 08 do; one figure per
    bifurcation. Diagnostic only; does not gate build_sfincs.
    """
    input:
        spec_basins_meta       = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg            = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        delta_polygon          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_polygon.gpkg"),
        river_forcing          = results_path("{basin_id}/preprocessing_inputs/forcing/river_forcing.nc"),
        # Figure background; this rule runs before the SFINCS grid exists.
        land_polygons          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
        sword_regions          = raw_region_inputs("river_network"),
        width_corrections      = SWORD_WIDTH_CORRECTIONS,
    output:
        plot_dir = directory(results_path("{basin_id}/preprocessing_inputs/visuals/11b_bifurcation_calibration_options")),
    params:
        discharge_variable = config["river_processing"]["flow_accumulation"]["discharge_variable"],
    threads: rule_threads("test_bifurcation_calibration_options")
    log:
        "logs/{basin_id}/11b_bifurcation_calibration_options.log"
    script:
        "../scripts/11b_bifurcation_calibration_options.py"
