rule get_protection_levels:
    """
    Identify the dominant (largest-area) existing flood-protection standard
    (FLOPROS, joined to WRI Aqueduct's geogunit_107 admin units) inside the
    delta polygon, separately for riverine and coastal hazards, and produce
    a diagnostic map of the protection return period in the area.

    Depends only on the delta polygon (rule split_delta_polygons, 01), not
    the model domain/river network — it is not "static terrain data" and
    doesn't need to be grouped with elevation/landuse/roughness (rules 05a-05c).
    Always runs and always produces its outputs; protection_levels.enabled
    (consumed by rule get_boundary_forcings, 07) only gates whether the
    identified protection level is actually subtracted from the forcing
    timeseries.
    """
    input:
        specific_delta  = results_path("{basin_id}/inputs/domain/{basin_id}_delta_polygon.gpkg"),
        flopros_table   = catalogue_path("protection_levels_flopros"),
        geogunit_raster = catalogue_path("wri_geogunit_107"),
        geogunit_list   = catalogue_path("wri_geogunit_107_list"),
        osm_land        = catalogue_path("osm_land"),
    output:
        protection_levels = results_path("{basin_id}/inputs/domain/protection_levels.json"),
        plot_protection   = results_path("{basin_id}/visuals/input_data/04_protection_levels.png"),
    params:
        default_rp_yr = config["protection_levels"]["default_rp_yr"],
        max_rp_yr     = config["protection_levels"]["max_rp_yr"],
    log:
        "logs/{basin_id}/04_protection_levels.log"
    script:
        "../scripts/04_get_protection_levels.py"
