rule get_protection_levels:
    """
    Identify the dominant (largest-area) existing flood-protection standard
    (FLOPROS, joined to WRI Aqueduct's geogunit_107 admin units) inside the
    delta polygon, separately for riverine and coastal hazards, and produce
    a diagnostic map of the protection return period in the area.

    Depends only on the delta polygon (rule split_delta_polygons, 01), not
    the model domain/river network — it is not "static terrain data" and
    doesn't need to be grouped with elevation/landuse/roughness (rules 05a-05c).
    The plot's own background (2026-08-06: landuse-derived, not OSM anymore
    -- see 04_get_protection_levels.py's own comment) is vectorized directly
    from the raw global landuse catalogue source, deliberately NOT from rule
    get_land_polygons' (03) own per-basin output, to avoid pulling in the
    model-domain dependency chain (02/03) this rule otherwise avoids entirely.
    Always runs and always produces its outputs; river_processing.empirical_estimation.modify_hydrograph
    (consumed by rule get_boundary_forcings, 07, empirical depth_method only)
    only gates whether the identified protection level is actually subtracted
    from the forcing timeseries. Rule modelled_depth_estimation (10, modelled
    depth_method) always consumes this rule's riverine_rp_yr when finite,
    independent of that toggle.
    """
    input:
        specific_delta  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_polygon.gpkg"),
        flopros_table   = catalogue_path("protection_levels_flopros"),
        geogunit_raster = catalogue_path("wri_geogunit_107"),
        geogunit_list   = catalogue_path("wri_geogunit_107_list"),
        global_landuse  = catalogue_path("land_use"),
    output:
        protection_levels = results_path("{basin_id}/preprocessing_inputs/domain/protection_levels.json"),
        plot_protection   = results_path("{basin_id}/preprocessing_inputs/visuals/04_protection_levels.png"),
    params:
        default_rp_yr = config["flopros_range"]["default_rp_yr"],
        max_rp_yr     = config["flopros_range"]["max_rp_yr"],
    log:
        "logs/{basin_id}/04_protection_levels.log"
    script:
        "../scripts/04_get_protection_levels.py"
