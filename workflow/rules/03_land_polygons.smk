rule get_land_polygons:
    """Vectorize the basin's own land-use raster (landuse != 200, i.e. not
    sea; rule prepare_landuse, 02b), native WGS84.

    Runs right after the land-use source is prepared (rule 02b) — every later
    static-data step (elevation, landuse, river network) takes this file as an
    input, so it has to exist before any of them, not alongside/after them.
    Does NOT need elevation_merged.tif -- rule get_elevation (05a) itself is
    one of this rule's own consumers (for its own diagnostic plots'
    background), so depending on 05a's own output here would be circular.

    Figure background only, and only for figures made before the SFINCS
    grid exists (rules 04-09, 11b) -- everything from rule
    grid_align_landuse (09b) on uses the grid-aligned land mask traced from
    landuse_on_grid.tif instead (2026-09-11; see 03_get_land_polygons.py).

    2026-08-06: previously clipped OSM land polygons instead -- see this
    rule's own script docstring and CHANGELOG for why that was dropped.
    """
    input:
        spec_basins_meta = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        landuse_source   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_source.tif"),
    output:
        land_polygons = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
    log:
        "logs/{basin_id}/03_land_polygons.log"
    script:
        "../scripts/03_get_land_polygons.py"
