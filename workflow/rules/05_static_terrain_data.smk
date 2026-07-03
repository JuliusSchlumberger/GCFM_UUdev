# Rules 05a-05c: static terrain/landcover raster prep. A genuinely linear
# chain (elevation -> landuse/zsini -> roughness), each step reprojecting
# onto the previous step's exact grid -- unlike the rest of the pipeline,
# sub-lettering within one number is appropriate here since there's no fork.

rule get_elevation:
    """
    Build the merged elevation product for a basin.

    Pipeline inside 05a_get_elevation.py:
      1. Merge FathomDEM tiles.
      2. DEM EGM2008 → GOCO06s (mandatory).
      3. Clip GEBCO to domain UTM grid; re-reference to GOCO06s by
         subtracting the MDT (mandatory).
      4. Hard merge: FathomDEM wherever valid, GEBCO everywhere else (no
         land-polygon mask, no gradient blend).
      5. Write elevation_merged.tif.
      6. Diagnostic elevation + datum-correction maps.

    zsini.tif is built in rule get_landuse (05b), which already needs the
    land/water-body distinction for its own reprojection step.
    """
    input:
        spec_basins_meta        = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        domain_gpkg             = results_path("{basin_id}/inputs/domain/{basin_id}_domain.gpkg"),
        global_topography_tiles = catalogue_path("fathomdem"),
        global_bathymetry       = catalogue_path("coastal_bathymetry"),
        land_polygons           = results_path("{basin_id}/inputs/domain/{basin_id}_land_polygons.gpkg"),
        # Datum-correction inputs (mandatory — FathomDEM's native EGM2008 datum
        # must not be blended with GEBCO/COAST-RP/MDT, which share GOCO06s).
        goco06s_gfc = catalogue_path("goco06s"),
        egm2008_gfc = catalogue_path("egm2008_geoid"),
        mdt         = catalogue_path("mdt_cnes_cls22"),
    output:
        elevation_merged = results_path("{basin_id}/inputs/domain/{basin_id}_elevation_merged.tif"),
        plot_elevation   = results_path("{basin_id}/visuals/input_data/05a_elevation.png"),
    params:
        work_res_m          = config["terrain"]["work_res_m"],
        mdt_variable        = config["datum_correction"]["mdt_variable"],
        mdt_load_margin_deg = config["datum_correction"]["mdt_load_margin_deg"],
    log:
        "logs/{basin_id}/05a_elevation.log"
    script:
        "../scripts/05a_get_elevation.py"


rule get_landuse:
    """Reprojected onto elevation_merged.tif's exact UTM grid (rule get_elevation,
    05a) so landuse/roughness share one pixel grid with elevation/zsini instead of
    each being independently reprojected by every downstream consumer.
    Also builds zsini.tif: land polygons rasterised on that same grid give the
    initial land/sea split (0 m at sea, nodata on land), then any cell with
    landuse==200 (permanent water body) is overridden to the sea initial value."""
    input:
        spec_basins_meta = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        domain_gpkg      = results_path("{basin_id}/inputs/domain/{basin_id}_domain.gpkg"),
        global_landuse   = catalogue_path("land_use"),
        land_polygons    = results_path("{basin_id}/inputs/domain/{basin_id}_land_polygons.gpkg"),
        elevation_merged = results_path("{basin_id}/inputs/domain/{basin_id}_elevation_merged.tif"),
    output:
        spec_landuse = results_path("{basin_id}/inputs/domain/{basin_id}_landuse.tif"),
        zsini        = results_path("{basin_id}/inputs/domain/{basin_id}_zsini.tif"),
        plot_landuse = results_path("{basin_id}/visuals/input_data/05b_landuse.png"),
        plot_zsini   = results_path("{basin_id}/visuals/input_data/05b_zsini.png"),
    log:
        "logs/{basin_id}/05b_landuse.log"
    script:
        "../scripts/05b_get_landuse.py"


rule get_roughness:
    input:
        spec_basins_meta      = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        domain_gpkg           = results_path("{basin_id}/inputs/domain/{basin_id}_domain.gpkg"),
        spec_landuse          = results_path("{basin_id}/inputs/domain/{basin_id}_landuse.tif"),
        matching_lu_roughness = catalogue_path("lu_to_roughness_lookup"),
        land_polygons              = results_path("{basin_id}/inputs/domain/{basin_id}_land_polygons.gpkg"),
    output:
        spec_roughness = results_path("{basin_id}/inputs/domain/{basin_id}_roughness.tif"),
        plot_roughness = results_path("{basin_id}/visuals/input_data/05c_roughness.png"),
    log:
        "logs/{basin_id}/05c_roughness.log"
    script:
        "../scripts/05c_get_roughness.py"
