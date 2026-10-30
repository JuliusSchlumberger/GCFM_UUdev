# The per-basin land-use classification every later rule reads, in pipeline
# codes (source classes + 200 = open sea), selected by config landuse.source:
#   copernicus_lc100  LC100, 100 m (catalogue 'land_use'): already has 200 = open sea.
#   esa_worldcover    WorldCover 2021, 10 m (catalogue 'land_use_esa_worldcover'):
#                     no sea class (sea, lagoons, rivers and lakes are all 80), so
#                     200 is derived against LC100's sea class. Tiles must first be
#                     indexed with tools/build_esa_worldcover_vrt.py.

rule prepare_landuse:
    """
    Clip the selected land-use source to the domain bbox in pipeline codes.

    Source classes + 200 = open sea, 255 = nodata; WGS84 at the source's own
    resolution. Rule get_protection_levels (04) reads raw LC100 instead.
    """
    input:
        spec_basins_meta = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        # Always required: the product (lc100) or the sea reference (esa_worldcover).
        global_landuse   = raw_input("land_use"),
        worldcover       = (
            raw_input("land_use_esa_worldcover")
            if config["landuse"]["source"] == "esa_worldcover" else []
        ),
        # Barrier for the sea fill: this basin's clipped network (rule 02).
        river_network    = (
            results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network.gpkg")
            if config["landuse"]["source"] == "esa_worldcover" else []
        ),
        # Elevation guard on the sea fill: high water bodies are never sea.
        fathomdem_tiles  = raw_input("fathomdem") if config["landuse"]["source"] == "esa_worldcover" else [],
    output:
        landuse_source = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_source.tif"),
        # Land (landuse != 200) as polygons: background of the figures made before
        # the SFINCS grid exists; nothing computational reads it.
        land_polygons  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
    params:
        source                     = config["landuse"]["source"],
        river_barrier_width_factor = config["landuse"]["river_barrier_width_factor"],
        river_barrier_min_width_m  = config["landuse"]["river_barrier_min_width_m"],
        sea_max_elevation_m        = config["landuse"]["sea_max_elevation_m"],
    threads: rule_threads("prepare_landuse")
    log:
        "logs/{basin_id}/05_prepare_landuse.log"
    script:
        "../scripts/05_prepare_landuse.py"
