# Rule: the land surface after the river network is cleaned. On the elevation
# grid: land use with the river channel set to water (landuse_channels.tif) and
# Manning's n at the land-use source's resolution with channel_manning_n in the
# channel (roughness.tif, for the subgrid tables). On the SFINCS grid, in
# SfincsModel's own orientation (row 0 on the origin side), so the rasters line
# up cell for cell with every model's sf.grid.data: land use by mode, Manning's n
# area-averaged from the source, and the land mask for figure backgrounds.
# The channel is src.river_burn.channel_polygons, the area rule 11 burns the bed into.

rule grid_align_landuse:
    input:
        spec_basins_meta      = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg           = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        sfincs_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_sfincs_grid.json"),
        landuse                = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse.tif"),
        # Grid reference of landuse_channels.tif and roughness.tif.
        elevation_merged       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_merged.tif"),
        # Manning's n is area-averaged from the native source.
        landuse_source         = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_source.tif"),
        matching_lu_roughness  = raw_input("lu_to_roughness_lookup"),
        # The channel takes channel_manning_n (src.river_burn.channel_polygons).
        river_network          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
    output:
        # On the elevation grid: the channel set to water (rule 10 reads landuse.tif,
        # whose islands it must see).
        landuse_channels         = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_channels.tif"),
        # Elevation grid subdivided to the source's pixel size: subgrid tables only.
        spec_roughness           = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_roughness.tif"),
        landuse_on_grid          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_on_grid.tif"),
        roughness_on_grid        = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_roughness_on_grid.tif"),
        # Land mask (land use != 200), WGS84, cell-aligned: figure background.
        land_mask_on_grid        = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
    params:
        water_code            = config["landuse"]["water_code"],
        roughness_aggregation = config["landuse"]["roughness_aggregation"],
        channel_manning_n     = config["river_processing"]["channel_manning_n"],
        resolution_m = lambda wildcards: grid_resolution_m(wildcards.basin_id),
    threads: rule_threads("grid_align_landuse")
    log:
        "logs/{basin_id}/09_land_surface.log"
    script:
        "../scripts/09_land_surface.py"
