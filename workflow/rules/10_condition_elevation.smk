rule enforce_river_monotonicity:
    """
    Modifies the DEM (not the river network) -- this rule conditions the
    ELEVATION RASTER, using the river network purely as a guide for where and
    in what order to walk. Enforces a monotonically non-increasing DEM
    elevation profile along every river centerline in the downstream
    direction: any centerline pixel whose DEM elevation exceeds the running
    minimum from upstream is lowered to that minimum, producing
    elevation_conditioned.tif.

    Operates on the FULLY-PROCESSED river network (river_network_processed.gpkg,
    rule 09a's output) rather than the raw or cleaned network -- deliberately,
    so conditioning only touches reaches that survive rule 09a's discharge
    filter and will actually be part of the final model; and on the raw
    merged DEM (elevation_merged.tif, rule 05a's output). This is why the
    rule can't run any earlier than it does, despite operating on elevation:
    it needs the FINAL reach set, which isn't known until rule 09a completes.

    Only scheduled when river_processing.conditioning.enabled = true.
    Independent of rule 09b (add_estuarine_depth) -- both only depend on 09a's
    output, neither depends on the other. Downstream rule build_sfincs (13)
    picks up elevation_conditioned.tif via a config-conditional input lambda;
    rule 11 (burn_river_bed) is the merge point needing both this rule's
    conditioned elevation and 09b's estuarine-adjusted network.
    """
    input:
        elevation_merged       = results_path("{basin_id}/inputs/domain/{basin_id}_elevation_merged.tif"),
        river_network          = results_path("{basin_id}/inputs/domain/{basin_id}_river_network_processed.gpkg"),
    output:
        elevation_conditioned  = results_path("{basin_id}/inputs/domain/{basin_id}_elevation_conditioned.tif"),
        plot_conditioning      = results_path("{basin_id}/visuals/input_data/10_condition_elevation.png"),
    log:
        "logs/{basin_id}/10_condition_elevation.log"
    script:
        "../scripts/10_condition_elevation.py"
