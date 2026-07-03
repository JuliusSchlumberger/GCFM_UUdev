rule determine_model_domain:
    input:
        specific_delta = results_path("{basin_id}/inputs/domain/{basin_id}_delta_polygon.gpkg"),
        river_basins   = catalogue_path("river_basins"),
    output:
        specific_basins = results_path("{basin_id}/inputs/domain/{basin_id}_intersecting_basins.gpkg"),
        domain_gpkg     = results_path("{basin_id}/inputs/domain/{basin_id}_domain.gpkg"),
        spec_basins_meta = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
    params:
        mode           = config["domain"]["mode"],
        buffer_m       = config["domain"]["buffer_m"],
        delta_buffer_m = config["domain"]["delta_buffer_m"],
        target_crs     = config["domain"]["target_crs"],
    log:
        "logs/{basin_id}/02_determine_model_domain.log"
    script:
        "../scripts/02_determine_model_domain.py"
