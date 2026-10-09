# Rule: resample landuse.tif (rule get_landuse, 05b, native resolution)
# onto the shared SFINCS regular grid exactly ONCE, then build roughness
# directly from that single coarse raster.
#
# The grid is the shared grid definition (sfincs_grid.json, rule
# build_sfincs_grid, 08c -- axis-aligned or rotated, see src/grid.py),
# stored in SfincsModel's own orientation (row 0 on the origin side), the
# same definition every SfincsModel in this pipeline is created from -- so
# these rasters line up cell for cell with every model's sf.grid.data.
#
# landuse_on_grid.tif is the ONLY sea/land/roughness classification any
# downstream consumer resamples from from here on -- rules
# modelled_depth_estimation (10, weir tracing +
# zsini sea-cell classification) and build_sfincs_skeleton (13, roughness +
# weir diagnostics) all read it directly, zero further reprojection, and
# directly assign their own SfincsModel's sf.grid.data["dep"] coords onto
# it -- which only works because this rule now stores it in that same
# SfincsModel-native orientation.
#
# DEM/subgrid stays native resolution -- this rule only touches the
# categorical landuse/roughness/sea classification chain.

rule grid_align_landuse:
    input:
        spec_basins_meta      = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg           = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        # THE grid definition (rule build_sfincs_grid, 08c) -- see src/grid.py.
        sfincs_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_sfincs_grid.json"),
        landuse                = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse.tif"),
        # Roughness is area-averaged from the SOURCE at its own resolution,
        # not reclassified from landuse_on_grid's dominant class -- same
        # reasoning as rule get_roughness (05c); see src.landuse.
        landuse_source         = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_source.tif"),
        matching_lu_roughness  = catalogue_path("lu_to_roughness_lookup"),
    output:
        landuse_on_grid          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_on_grid.tif"),
        roughness_on_grid        = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_roughness_on_grid.tif"),
        # Land mask (land use != 200) traced from landuse_on_grid, WGS84,
        # cell-aligned with the model -- the land background of every
        # figure of model output from here on (see src.plots' docstring).
        land_mask_on_grid        = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
        plot_landuse_on_grid     = results_path("{basin_id}/preprocessing_inputs/visuals/09b_landuse_on_grid.png"),
        plot_roughness_on_grid   = results_path("{basin_id}/preprocessing_inputs/visuals/09b_roughness_on_grid.png"),
    params:
        roughness_aggregation = config["landuse"]["roughness_aggregation"],
        resolution = lambda wildcards: grid_resolution_m(wildcards.basin_id),
    log:
        "logs/{basin_id}/09b_grid_align_landuse.log"
    script:
        "../scripts/09b_grid_align_landuse.py"
