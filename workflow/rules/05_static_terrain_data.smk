# Rules 05a-05c: static terrain/landcover rasters, a linear chain
# (elevation -> landuse/zsini -> roughness) on one shared grid.

rule get_elevation:
    """
    Build the merged elevation product for a basin.

    Pipeline inside 05a_get_elevation.py:
      1. Merge FathomDEM tiles.
      1b. Set FathomDEM to NaN on sea cells (land use 200): it is
          terrestrial and reports spurious shallow "elevation" over water,
          so step 4 falls back to GEBCO there.
      2. DEM EGM2008 → GOCO06s (mandatory).
      3. Clip GEBCO to domain UTM grid; re-reference MSL -> GOCO06s by
         adding the MDT (mandatory; H_GOCO06s = H_MSL + MDT).
      3b. Clamp GEBCO between terrain.gebco_max_depth_m (deep water shrinks
          SFINCS's CFL time step) and, on sea cells, terrain.gebco_min_depth_m
          (too-shallow coastal artefacts; creeks GEBCO cannot resolve) below
          local mean sea level.
      4. Merge: FathomDEM wherever valid, GEBCO everywhere else, with a
         linear blend from FathomDEM to GEBCO over terrain.coast_blend_m
         seaward of the coastline (open water only, not narrow creeks).
      5. Write elevation_merged.tif.
      6. Diagnostic elevation + datum-correction maps.

    sea_mask.tif is built in rule get_landuse (05b). The actual initial water
    level (zsini.tif) is built later, in rule build_sfincs_skeleton, from
    zsini_sea_cells_on_grid.tif (rule 10).
    """
    input:
        spec_basins_meta        = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg             = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        global_topography_tiles = raw_input("fathomdem"),
        global_bathymetry       = raw_input("coastal_bathymetry"),
        # Sea/land: land use 200 = sea, the criterion zsini, weir tracing and
        # the water-level boundary also use.
        landuse_source          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_source.tif"),
        land_polygons           = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
        # Datum-correction inputs (mandatory — FathomDEM's native EGM2008 datum
        # must not be blended with GEBCO/COAST-RP/MDT, which share GOCO06s).
        goco06s_gfc = raw_input("goco06s"),
        egm2008_gfc = raw_input("egm2008_geoid"),
        mdt         = raw_input("mdt_cnes_cls22"),
    output:
        elevation_merged = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_merged.tif"),
        plot_elevation   = results_path("{basin_id}/preprocessing_inputs/visuals/05a_elevation.png"),
    params:
        mdt_load_margin_deg = config["datum_correction"]["mdt_load_margin_deg"],
        gebco_max_depth_m   = config["terrain"]["gebco_max_depth_m"],
        gebco_min_depth_m   = config["terrain"]["gebco_min_depth_m"],
        coast_blend_m       = config["terrain"]["coast_blend_m"],
    threads: rule_threads("get_elevation")
    resources:
        raw_io=1,
    log:
        "logs/{basin_id}/05a_elevation.log"
    script:
        "../scripts/05a_get_elevation.py"


rule get_landuse:
    """Resample the land-use raster onto elevation_merged.tif's UTM grid and build sea_mask.tif.

    Resampled by mode, not nearest-neighbour: the 10 m WorldCover source is
    finer than the ~30 m grid, and nearest would keep one arbitrary pixel of
    ~9. For 100 m LC100 it is upsampling, where both agree.
    sea_mask.tif = landuse == 200 only (1.0 at sea, nodata elsewhere), the same
    criterion as src.protection_weir's ocean_mask, so no cell starts wet
    outside the weir. land_polygons is used for plot backgrounds only.
    sea_mask.tif is a pure classification, not the initial water level
    itself; zsini.tif is built by rule build_sfincs_skeleton from
    zsini_sea_cells_on_grid.tif (rule 10)."""
    input:
        spec_basins_meta = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        landuse_source   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_source.tif"),
        land_polygons    = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
        elevation_merged = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_merged.tif"),
    output:
        spec_landuse  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse.tif"),
        sea_mask      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_sea_mask.tif"),
        plot_landuse  = results_path("{basin_id}/preprocessing_inputs/visuals/05b_landuse.png"),
        plot_sea_mask = results_path("{basin_id}/preprocessing_inputs/visuals/05b_sea_mask.png"),
    threads: rule_threads("get_landuse")
    log:
        "logs/{basin_id}/05b_landuse.log"
    script:
        "../scripts/05b_get_landuse.py"


rule get_roughness:
    """Area-average Manning's n from the native land-use source onto elevation_merged.tif's grid.

    Not reclassified from the upscaled landuse.tif, which would give a cell
    that is 60% water and 40% built-up the dominant class' 0.02 (basin 2433835,
    70 m cells: 30% hold more than one class, and in 43% of those the two
    differ by more than 0.01 in n). hydromt also samples this raster per
    subgrid pixel (7 m at nr_subgrid_pixels=10). Aggregation:
    config landuse.roughness_aggregation (src.landuse.aggregate_manning).
    """
    input:
        spec_basins_meta      = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg           = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        # Native-resolution source, not landuse.tif.
        landuse_source        = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_source.tif"),
        # Grid reference only.
        elevation_merged      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_merged.tif"),
        matching_lu_roughness = raw_input("lu_to_roughness_lookup"),
        land_polygons              = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
        # The channel takes channel_manning_n (src.river_burn.channel_polygons).
        river_network         = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
    output:
        spec_roughness = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_roughness.tif"),
        plot_roughness = results_path("{basin_id}/preprocessing_inputs/visuals/05c_roughness.png"),
    params:
        roughness_aggregation = config["landuse"]["roughness_aggregation"],
        channel_manning_n     = config["river_processing"]["channel_manning_n"],
    threads: rule_threads("get_roughness")
    log:
        "logs/{basin_id}/05c_roughness.log"
    script:
        "../scripts/05c_get_roughness.py"
