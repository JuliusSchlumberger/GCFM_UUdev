rule get_protection_levels:
    """
    Find the dominant flood-protection standard in the delta, per hazard.

    Largest-area FLOPROS standard (joined to WRI Aqueduct geogunit_107 admin
    units) for riverine and coastal hazards. Depends only on the delta polygon.
    surge_protection_rp_yr feeds the coastal
    protection crest (rule 07d); rule 11 uses river_protection_rp_yr when finite.
    """
    input:
        specific_delta  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_polygon.gpkg"),
        flopros_table   = raw_input("protection_levels_flopros"),
        geogunit_raster = raw_input("wri_geogunit_107"),
        geogunit_list   = raw_input("wri_geogunit_107_list"),
    output:
        protection_levels = results_path("{basin_id}/preprocessing_inputs/domain/protection_levels.json"),
    params:
        default_rp_yr = config["flopros_range"]["default_rp_yr"],
        max_rp_yr     = config["flopros_range"]["max_rp_yr"],
    threads: rule_threads("get_protection_levels")
    log:
        "logs/{basin_id}/03_protection_levels.log"
    script:
        "../scripts/03_get_protection_levels.py"
