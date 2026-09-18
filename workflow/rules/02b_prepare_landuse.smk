# Rule: the per-basin land-use classification every later rule reads.
#
# Runs right after the domain is determined (rule 02) and before rule
# get_land_polygons (03), which vectorizes this rule's output. Selects
# between the two sources with config landuse.source and writes ONE raster
# in PIPELINE CODES -- the source's own classes plus 200 for open sea --
# so nothing downstream needs source-conditional logic:
#
#   copernicus_lc100  LC100, 100 m (catalogue 'land_use'): already carries
#                     200 = open sea, so this rule only windows it.
#   esa_worldcover    WorldCover 2021, 10 m (catalogue
#                     'land_use_esa_worldcover'): has NO sea class (sea,
#                     lagoons, rivers and lakes are all class 80), so 200
#                     is DERIVED against LC100's own sea class -- which is
#                     why the LC100 input below is unconditional. See
#                     workflow/src/landuse.py for the rule and
#                     tools/build_esa_worldcover_vrt.py for the mosaic the
#                     tiles have to be indexed into first.

rule prepare_landuse:
    """
    Clip the selected land-use source to the basin's domain bbox and write it
    in pipeline codes (source classes + 200 = open sea, 255 = nodata), WGS84,
    at the source's own resolution.

    THE single land-use classification of the pipeline: rule
    get_land_polygons (03) vectorizes it for figure backgrounds, rule
    get_landuse (05b) resamples it (by MODE -- classes are categorical) onto
    the elevation grid and derives sea_mask.tif from its 200, and rule
    get_roughness (05c) AREA-AVERAGES Manning's n from it (never from an
    upscaled class raster; see src.landuse.aggregate_manning).

    Rule get_protection_levels (04) is deliberately NOT a consumer: its plot
    background is vectorized from the raw LC100 source over a wider window
    than this basin's domain, keeping that rule independent of the model
    domain chain (see its own docstring).
    """
    input:
        spec_basins_meta = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        # Always required: the product itself in copernicus_lc100 mode, the
        # sea reference for deriving 200 in esa_worldcover mode.
        global_landuse   = catalogue_path("land_use"),
        worldcover       = (
            catalogue_path("land_use_esa_worldcover")
            if config["landuse"]["source"] == "esa_worldcover" else []
        ),
        # Barrier for the sea fill: THIS basin's own clipped network (rule
        # get_river_network, 06 -- which is why that rule's plot lives in a
        # separate rule, see 06_river_network.smk), never the raw global
        # SWORD source.
        river_network    = (
            results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network.gpkg")
            if config["landuse"]["source"] == "esa_worldcover" else []
        ),
    output:
        landuse_source = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_source.tif"),
    params:
        source                     = config["landuse"]["source"],
        river_barrier_width_factor = config["landuse"]["river_barrier_width_factor"],
        river_barrier_min_width_m  = config["landuse"]["river_barrier_min_width_m"],
    log:
        "logs/{basin_id}/02b_prepare_landuse.log"
    script:
        "../scripts/02b_prepare_landuse.py"
