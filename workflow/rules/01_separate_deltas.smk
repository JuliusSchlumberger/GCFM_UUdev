rule split_delta_polygons:
    input:
        delta_polygons = raw_input("delta_polygons")
    output:
        specific_delta = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_polygon.gpkg")
    threads: rule_threads("split_delta_polygons")
    log:
        "logs/{basin_id}/01_separate_deltas.log"
    script:
        "../scripts/01_separate_deltas.py"
