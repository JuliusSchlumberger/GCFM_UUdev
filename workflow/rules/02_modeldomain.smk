rule determine_model_domain:
    input:
        specific_delta = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_polygon.gpkg"),
    output:
        domain_gpkg     = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        spec_basins_meta = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
    params:
        delta_buffer_m = config["domain"]["delta_buffer_m"],
        # Orientation of the model grid's rectangle, fitted here once so the
        # clipping bbox is guaranteed to cover the whole grid.
        rotated        = lambda wildcards: grid_rotated(wildcards.basin_id),
    log:
        "logs/{basin_id}/02_determine_model_domain.log"
    script:
        "../scripts/02_determine_model_domain.py"
