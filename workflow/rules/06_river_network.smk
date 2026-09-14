# Rules 06/06b: the basin's SWORD river network, clipped, and its diagnostic map.
#
# Deliberately SPLIT (2026-09-14). The clipped network is an INPUT of rule
# prepare_landuse (02b): ESA WorldCover has no open-sea class, and the sea
# is derived by flood-filling its water class inward from the domain border
# with the river network as a barrier, so that the sea stops at the river
# mouths instead of running up every channel. The plot, on the other hand,
# draws the network over rule get_land_polygons' (03) own land mask -- and
# 03 vectorizes 02b's output. Clip and plot in one rule would therefore be
# a cycle (02b -> 03 -> 06 -> 02b); with the clip free of the land mask the
# order is simply 02 -> 06 -> 02b -> 03 -> 06b.

rule get_river_network:
    """
    Clip the global (modified) SWORD network to the domain bbox and write it
    straight out -- no topology/width processing, that is rule
    clean_river_network (08).
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
    Diagnostic map of the clipped network over rule get_land_polygons' (03)
    land mask -- the only part of rule 06 that depends on the land mask, and
    therefore the only part that has to run after rules 02b/03.
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
