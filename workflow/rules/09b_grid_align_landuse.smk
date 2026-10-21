# Rule: resample landuse.tif (rule get_landuse, 05b, native resolution)
# onto the shared SFINCS regular grid exactly ONCE (mode), and area-average
# roughness onto the same grid from the native land-use source.
#
# Stored in SfincsModel's own orientation (row 0 on the origin side), so the
# rasters line up cell for cell with every model's sf.grid.data and rules 10
# and 13 assign sf.grid.data["dep"] coords onto them without reprojection.
# DEM/subgrid stay at native resolution.

rule grid_align_landuse:
    input:
        spec_basins_meta      = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg           = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        sfincs_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_sfincs_grid.json"),
        landuse                = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse.tif"),
        # Roughness is area-averaged from the native source, as in rule 05c.
        landuse_source         = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_source.tif"),
        matching_lu_roughness  = catalogue_path("lu_to_roughness_lookup"),
    output:
        landuse_on_grid          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_on_grid.tif"),
        roughness_on_grid        = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_roughness_on_grid.tif"),
        # Land mask (land use != 200), WGS84, cell-aligned: figure background.
        land_mask_on_grid        = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
        plot_landuse_on_grid     = results_path("{basin_id}/preprocessing_inputs/visuals/09b_landuse_on_grid.png"),
        plot_roughness_on_grid   = results_path("{basin_id}/preprocessing_inputs/visuals/09b_roughness_on_grid.png"),
    params:
        roughness_aggregation = config["landuse"]["roughness_aggregation"],
        resolution_m = lambda wildcards: grid_resolution_m(wildcards.basin_id),
    log:
        "logs/{basin_id}/09b_grid_align_landuse.log"
    script:
        "../scripts/09b_grid_align_landuse.py"
