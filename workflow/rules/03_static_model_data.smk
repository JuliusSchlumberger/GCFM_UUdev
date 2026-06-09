rule get_elevation:
    """
    Build the merged elevation product for a basin.

    Pipeline inside 03a_get_elevation.py:
      1. Merge on-land tiles.
      2. Clip GEBCO to domain UTM grid.
      3. Rasterise land polygons; fill land NaN with impassable-barrier value.
      4. Hard-boundary merge (DiluviumDEM or DeltaDTM on-land, GEBCO in ocean).
      5. Write elevation_merged.tif + zsini.tif.
      6. Diagnostic elevation + zsini maps.

    Optional (topography.blend.enabled: true):
      Gradient-blended elevation_gradient.tif is written as a side effect
      (not a tracked Snakemake output) for visual comparison only.
    """
    input:
        spec_basins_meta        = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        domain_gpkg             = results_path("{basin_id}/inputs/domain/domain.gpkg"),
        global_topography_tiles = lambda wc: catalogue_path(
            "diluvium_dem" if config["topography"]["source"] == "DiluviumDEM"
            else "delta_dtm"
        ),
        global_bathymetry       = catalogue_path("coastal_bathymetry"),
        land_polygons           = results_path("{basin_id}/inputs/domain/land_polygons.gpkg"),
        # Datum-correction inputs — only required when source==DiluviumDEM and enabled.
        # Lambda returns [] when not needed so Snakemake skips the dependency check.
        goco06s_gfc = lambda wc: (
            catalogue_path("goco06s")
            if config["topography"]["source"] == "DiluviumDEM"
            and config["topography"]["datum_correction"]["enabled"]
            else []
        ),
        egm2008_gfc = lambda wc: (
            catalogue_path("egm2008_geoid")
            if config["topography"]["source"] == "DiluviumDEM"
            and config["topography"]["datum_correction"]["enabled"]
            else []
        ),
        mdt = lambda wc: (
            catalogue_path("mdt_cnes_cls22")
            if config["topography"]["source"] == "DiluviumDEM"
            and config["topography"]["datum_correction"]["enabled"]
            else []
        ),
    output:
        elevation_merged = results_path("{basin_id}/inputs/domain/elevation_merged.tif"),
        zsini            = results_path("{basin_id}/inputs/domain/zsini.tif"),
        plot_elevation   = results_path("{basin_id}/inputs/plots/01_elevation.png"),
        plot_zsini       = results_path("{basin_id}/inputs/plots/01c_zsini.png"),
    params:
        work_res_m        = config["topography"]["blend"]["work_res_m"],
        blend_buffer_m    = config["topography"]["blend"]["blend_buffer_m"],
        clip_elevation_m  = config["topography"]["clip_elevation_m"][config["topography"]["source"]],
        land_barrier_val  = config["topography"]["land_barrier_elevation"],
        blend_enabled     = config["topography"]["blend"]["enabled"],
        topo_source       = config["topography"]["source"],
        datum_correction_enabled  = config["topography"]["datum_correction"]["enabled"],
        mdt_variable              = config["topography"]["datum_correction"]["mdt_variable"],
        mdt_load_margin_deg       = config["topography"]["datum_correction"]["mdt_load_margin_deg"],
    log:
        "logs/{basin_id}/03a_elevation.log"
    script:
        "../scripts/03a_get_elevation.py"


rule get_landuse:
    input:
        spec_basins_meta = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        domain_gpkg      = results_path("{basin_id}/inputs/domain/domain.gpkg"),
        global_landuse   = catalogue_path("land_use"),
        land_polygons         = results_path("{basin_id}/inputs/domain/land_polygons.gpkg"),
    output:
        spec_landuse = results_path("{basin_id}/inputs/domain/landuse.tif"),
        plot_landuse = results_path("{basin_id}/inputs/plots/02_landuse.png"),
    log:
        "logs/{basin_id}/03b_landuse.log"
    script:
        "../scripts/03b_get_landuse.py"


rule get_roughness:
    input:
        spec_basins_meta      = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        domain_gpkg           = results_path("{basin_id}/inputs/domain/domain.gpkg"),
        spec_landuse          = results_path("{basin_id}/inputs/domain/landuse.tif"),
        matching_lu_roughness = catalogue_path("lu_to_roughness_lookup"),
        land_polygons              = results_path("{basin_id}/inputs/domain/land_polygons.gpkg"),
    output:
        spec_roughness = results_path("{basin_id}/inputs/domain/roughness.tif"),
        plot_roughness = results_path("{basin_id}/inputs/plots/03_roughness.png"),
    log:
        "logs/{basin_id}/03c_roughness.log"
    script:
        "../scripts/03c_get_roughness.py"


rule get_river_network:
    input:
        spec_basins_meta     = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        domain_gpkg          = results_path("{basin_id}/inputs/domain/domain.gpkg"),
        global_river_network = catalogue_path("river_network"),
        land_polygons        = results_path("{basin_id}/inputs/domain/land_polygons.gpkg"),
        elevation_merged     = results_path("{basin_id}/inputs/domain/elevation_merged.tif"),
    output:
        spec_river_network = results_path("{basin_id}/inputs/domain/river_network.gpkg"),
        plot_river_network = results_path("{basin_id}/inputs/plots/04_river_network.png"),
    log:
        "logs/{basin_id}/03d_river_network.log"
    script:
        "../scripts/03d_get_river_network.py"


rule get_land_polygons:
    """Clip the global OSM land polygons to the basin model domain.

    The clipped file is used by build_sfincs (rule 08) to define waterlevel
    boundary cells: active-domain edge cells NOT covered by land become mask=2
    (waterlevel boundary), which is exactly the coastal/open-water perimeter.
    """
    input:
        spec_basins_meta = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        domain_gpkg      = results_path("{basin_id}/inputs/domain/domain.gpkg"),
        osm_land         = catalogue_path("osm_land"),
    output:
        land_polygons = results_path("{basin_id}/inputs/domain/land_polygons.gpkg"),
    log:
        "logs/{basin_id}/03e_get_land_polygons.log"
    script:
        "../scripts/03e_get_land_polygons.py"
