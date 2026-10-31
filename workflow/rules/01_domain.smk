# Rule: the basin's delta polygon from the catalogue's delta polygons, its UTM
# CRS and the clipping bbox every later rule reads.

rule determine_model_domain:
    input:
        delta_polygons = raw_input("delta_polygons"),
    output:
        specific_delta   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_polygon.gpkg"),
        domain_gpkg      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        spec_basins_meta = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
    params:
        delta_buffer_m = config["domain"]["delta_buffer_m"],
    threads: rule_threads("determine_model_domain")
    log:
        "logs/{basin_id}/01_determine_model_domain.log"
    script:
        "../scripts/01_determine_model_domain.py"
