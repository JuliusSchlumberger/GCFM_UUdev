rule get_land_polygons:
    """Vectorize the global landuse raster (landuse != 200, i.e. not sea),
    clipped to the basin model domain, native WGS84.

    Runs right after the domain is determined (rule 02) — every later static-
    data step (elevation, landuse, river network) takes this file as an
    input, so it has to exist before any of them, not alongside/after them.
    Only needs the domain + the raw global landuse catalogue source, NOT
    elevation_merged.tif -- rule get_elevation (05a) itself is one of this
    rule's own consumers (for its own diagnostic plots' background), so
    depending on 05a's own output here would be circular.

    2026-08-06: previously clipped OSM land polygons instead -- see this
    rule's own script docstring (03_get_land_polygons.py) and CHANGELOG for
    why that was dropped. The output file's own name/shape/role is
    unchanged, so every downstream consumer (every diagnostic plot's
    background, plus the two real, functional exclude_polygon uses in
    10_depth_estimation_modelled.py/13_build_sfincs_skeleton.py's own
    waterlevel boundary mask) needed no changes at all.
    """
    input:
        spec_basins_meta = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        global_landuse   = catalogue_path("land_use"),
    output:
        land_polygons = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
    log:
        "logs/{basin_id}/03_land_polygons.log"
    script:
        "../scripts/03_get_land_polygons.py"
