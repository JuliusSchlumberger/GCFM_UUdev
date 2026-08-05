# Rules 05a-05c: static terrain/landcover raster prep. A genuinely linear
# chain (elevation -> landuse/zsini -> roughness), each step reprojecting
# onto the previous step's exact grid -- unlike the rest of the pipeline,
# sub-lettering within one number is appropriate here since there's no fork.

rule get_elevation:
    """
    Build the merged elevation product for a basin.

    Pipeline inside 05a_get_elevation.py:
      1. Merge FathomDEM tiles.
      1b. Clip FathomDEM to ocean=NaN using the DeltaDTM validity mask
          (src.raster.clip_ocean_from_topo) — FathomDEM is terrestrial and
          reports spurious shallow "elevation" over open water instead of
          nodata; this lets step 4's merge fall back to GEBCO there.
      2. DEM EGM2008 → GOCO06s (mandatory).
      3. Clip GEBCO to domain UTM grid; re-reference to GOCO06s by
         subtracting the MDT (mandatory).
      3b. Clamp GEBCO depths to terrain.gebco_max_depth_m below sea level --
          mitigates SFINCS's CFL-driven time step shrinking in genuinely
          deep offshore water it isn't modelling open-ocean dynamics for.
      4. Hard merge: FathomDEM wherever valid, GEBCO everywhere else (no
         land-polygon mask, no gradient blend).
      5. Write elevation_merged.tif.
      6. Diagnostic elevation + datum-correction maps.

    The sea/land classification (sea_mask.tif) is built in rule get_landuse
    (05b), which already needs the land/water-body distinction for its own
    reprojection step. The actual initial water level (zsini.tif) is built
    later, in rule build_sfincs (13), once baseline_m is known.
    """
    input:
        spec_basins_meta        = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg             = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        global_topography_tiles = catalogue_path("fathomdem"),
        global_bathymetry       = catalogue_path("coastal_bathymetry"),
        deltadtm_mask           = catalogue_path("deltadtm_mask"),
        land_polygons           = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
        # Datum-correction inputs (mandatory — FathomDEM's native EGM2008 datum
        # must not be blended with GEBCO/COAST-RP/MDT, which share GOCO06s).
        goco06s_gfc = catalogue_path("goco06s"),
        egm2008_gfc = catalogue_path("egm2008_geoid"),
        mdt         = catalogue_path("mdt_cnes_cls22"),
    output:
        elevation_merged = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_merged.tif"),
        plot_elevation   = results_path("{basin_id}/preprocessing_inputs/visuals/05a_elevation.png"),
    params:
        mdt_load_margin_deg = config["datum_correction"]["mdt_load_margin_deg"],
        gebco_max_depth_m   = config["terrain"]["gebco_max_depth_m"],
    log:
        "logs/{basin_id}/05a_elevation.log"
    script:
        "../scripts/05a_get_elevation.py"


rule get_landuse:
    """Reprojected onto elevation_merged.tif's exact UTM grid (rule get_elevation,
    05a) so landuse/roughness share one pixel grid with elevation/zsini instead of
    each being independently reprojected by every downstream consumer.
    Also builds sea_mask.tif: land polygons rasterised on that same grid give
    the land/sea classification (1.0 at sea, nodata on land), then any cell
    with landuse==200 (permanent water body) is overridden to sea. This is a
    pure classification, not the initial water level itself -- rule
    build_sfincs (13) turns it into zsini.tif once baseline_m is known."""
    input:
        spec_basins_meta = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        global_landuse   = catalogue_path("land_use"),
        land_polygons    = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
        elevation_merged = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_merged.tif"),
    output:
        spec_landuse  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse.tif"),
        sea_mask      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_sea_mask.tif"),
        plot_landuse  = results_path("{basin_id}/preprocessing_inputs/visuals/05b_landuse.png"),
        plot_sea_mask = results_path("{basin_id}/preprocessing_inputs/visuals/05b_sea_mask.png"),
    log:
        "logs/{basin_id}/05b_landuse.log"
    script:
        "../scripts/05b_get_landuse.py"


rule get_roughness:
    input:
        spec_basins_meta      = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg           = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        spec_landuse          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse.tif"),
        matching_lu_roughness = catalogue_path("lu_to_roughness_lookup"),
        land_polygons              = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
    output:
        spec_roughness = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_roughness.tif"),
        plot_roughness = results_path("{basin_id}/preprocessing_inputs/visuals/05c_roughness.png"),
    log:
        "logs/{basin_id}/05c_roughness.log"
    script:
        "../scripts/05c_get_roughness.py"
