rule get_land_polygons:
    """Clip the global OSM land polygons to the basin model domain.

    Runs right after the domain is determined (rule 02) — every later static-
    data step (elevation, landuse, river network) takes this file as an
    input, so it has to exist before any of them, not alongside/after them.

    The clipped file is used by build_sfincs (rule 13) to define waterlevel
    boundary cells: active-domain edge cells NOT covered by land become mask=2
    (waterlevel boundary), which is exactly the coastal/open-water perimeter.
    """
    input:
        spec_basins_meta = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        domain_gpkg      = results_path("{basin_id}/inputs/domain/{basin_id}_domain.gpkg"),
        osm_land         = catalogue_path("osm_land"),
    output:
        land_polygons = results_path("{basin_id}/inputs/domain/{basin_id}_land_polygons.gpkg"),
    log:
        "logs/{basin_id}/03_land_polygons.log"
    script:
        "../scripts/03_get_land_polygons.py"
