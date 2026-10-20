rule get_land_polygons:
    """Vectorize the basin's land-use raster (landuse != 200, not sea), native WGS84.

    Runs right after the land-use source is prepared (rule 02b) — every later
    static-data step (elevation, landuse, river network) takes this file as an
    input, so it has to exist before any of them, not alongside/after them.

    Figure background only, for figures made before the SFINCS grid exists
    (rules 04-09, 11b); from rule grid_align_landuse (09b) on, the grid-aligned
    land mask is used.
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
