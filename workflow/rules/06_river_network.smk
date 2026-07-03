rule get_river_network:
    input:
        spec_basins_meta     = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        domain_gpkg          = results_path("{basin_id}/inputs/domain/{basin_id}_domain.gpkg"),
        global_river_network = catalogue_path("river_network"),
        land_polygons        = results_path("{basin_id}/inputs/domain/{basin_id}_land_polygons.gpkg"),
        elevation_merged     = results_path("{basin_id}/inputs/domain/{basin_id}_elevation_merged.tif"),
        spec_landuse         = results_path("{basin_id}/inputs/domain/{basin_id}_landuse.tif"),
    output:
        spec_river_network = results_path("{basin_id}/inputs/domain/{basin_id}_river_network.gpkg"),
        plot_river_network = results_path("{basin_id}/visuals/input_data/06_river_network.png"),
    log:
        "logs/{basin_id}/06_river_network.log"
    script:
        "../scripts/06_get_river_network.py"
