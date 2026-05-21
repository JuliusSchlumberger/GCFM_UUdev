rule split_delta_polygons:
    input:
        delta_polygons = lambda w: catalogue_path("delta_polygons")
    output:
        specific_delta = results_path("{basin_id}/inputs/delta_polygon.gpkg")
    log:
        "logs/{basin_id}/separate_deltas.log"
    script:
        "../scripts/separate_deltas.py"
