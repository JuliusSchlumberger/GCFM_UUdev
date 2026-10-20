# Rules 06/06b: the basin's SWORD river network, clipped, and its diagnostic map.
#
# Split to avoid a cycle: the clipped network is the barrier for rule
# prepare_landuse's (02b) sea flood fill, so the sea stops at river mouths
# instead of running up every channel, while the plot needs rule 03's land
# mask, built from 02b. Order: 02 -> 06 -> 02b -> 03 -> 06b.

rule get_river_network:
    """
    Clip the global (modified) SWORD network to the domain bbox, unprocessed.
    """
    input:
        spec_basins_meta     = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        global_river_network = catalogue_path("river_network"),
    output:
        spec_river_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network.gpkg"),
    log:
        "logs/{basin_id}/06_river_network.log"
    script:
        "../scripts/06_get_river_network.py"


rule plot_river_network:
    """
    Plot the clipped network over rule get_land_polygons' (03) land mask.
    """
    input:
        spec_basins_meta   = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg        = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        spec_river_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network.gpkg"),
        land_polygons      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
    output:
        plot_river_network = results_path("{basin_id}/preprocessing_inputs/visuals/06_river_network.png"),
    log:
        "logs/{basin_id}/06b_plot_river_network.log"
    script:
        "../scripts/06b_plot_river_network.py"
