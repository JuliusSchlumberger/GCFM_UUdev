# Rule: FathomDEM + GEBCO merged on one UTM grid, and the land use on that grid.

rule get_elevation:
    """
    Build the merged elevation product for a basin.

    Pipeline inside 06_get_elevation.py:
      1. Merge FathomDEM tiles.
      1b. Resample the land use onto this grid (landuse.tif, mode) and set
          FathomDEM to NaN on its sea cells (land use 200): it is
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
      6. Diagnostic elevation, land-use and datum-correction maps.

    The initial water level (zsini.tif) is built later, in rule
    build_sfincs_skeleton, from zsini_sea_cells_on_grid.tif (rule 11).
    """
    input:
        spec_basins_meta        = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg             = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        global_topography_tiles = raw_input("fathomdem"),
        global_bathymetry       = raw_input("coastal_bathymetry"),
        # Sea/land: land use 200 = sea, the criterion zsini, weir tracing and
        # the water-level boundary also use.
        landuse_source          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_source.tif"),
        # Datum-correction inputs (mandatory — FathomDEM's native EGM2008 datum
        # must not be blended with GEBCO/COAST-RP/MDT, which share GOCO06s).
        goco06s_gfc = raw_input("goco06s"),
        egm2008_gfc = raw_input("egm2008_geoid"),
        mdt         = raw_input("mdt_cnes_cls22"),
    output:
        elevation_merged = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_merged.tif"),
        # Land use on this grid (mode); its class 200 also blanks FathomDEM over sea.
        spec_landuse     = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse.tif"),
        # Datum corrections, drawn by rule analysis_preprocess: geoid offset
        # (EGM2008 - GOCO06s), raw MDT over the ocean, and the corrections
        # applied to FathomDEM and GEBCO on this rule's UTM grid.
        geoid_offset     = diagnostics_path("geoid_offset.tif"),
        mdt_ocean        = diagnostics_path("mdt_ocean.tif"),
        datum_fathomdem  = diagnostics_path("datum_correction_fathomdem.tif"),
        datum_gebco      = diagnostics_path("datum_correction_gebco.tif"),
    params:
        mdt_load_margin_deg = config["datum_correction"]["mdt_load_margin_deg"],
        gebco_max_depth_m   = config["terrain"]["gebco_max_depth_m"],
        gebco_min_depth_m   = config["terrain"]["gebco_min_depth_m"],
        coast_blend_m       = config["terrain"]["coast_blend_m"],
    threads: rule_threads("get_elevation")
    resources:
        raw_io=1,
    log:
        "logs/{basin_id}/06_elevation.log"
    script:
        "../scripts/06_get_elevation.py"
