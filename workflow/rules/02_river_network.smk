# Rule: the basin's SWORD river network, clipped to the domain bbox. It runs before
# rule prepare_landuse (05), whose sea flood fill uses it as a barrier so the sea
# stops at river mouths instead of running up every channel.
# analysis/plot_river_network.py maps it.

rule get_river_network:
    """
    Clip SWORD v17c (all regions) to the domain bbox and apply the manual widths.
    """
    input:
        spec_basins_meta     = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        sword_regions        = raw_region_inputs("river_network"),
        width_corrections    = SWORD_WIDTH_CORRECTIONS,
    output:
        spec_river_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network.gpkg"),
    threads: rule_threads("get_river_network")
    resources:
        raw_io=1,
    log:
        "logs/{basin_id}/02_river_network.log"
    script:
        "../scripts/02_get_river_network.py"
