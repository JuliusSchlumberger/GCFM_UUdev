# Rule: resample landuse.tif (rule get_landuse, 05b, native resolution)
# onto the shared SFINCS regular grid exactly ONCE, then build roughness
# directly from that single coarse raster.
#
# Builds its own grid via create_grid_from_region(domain_gpkg,
# grid_resolution.json) + the SAME y-ascending flip SfincsModel.grid.
# create_from_region() applies internally, rather than reading
# sfincs_grid.json (rule build_sfincs_grid, 08c) directly -- sfincs_grid.json
# is stored in that function's own RAW (y-descending, GDAL-standard)
# orientation, deliberately NOT flipped (rule 08c avoids instantiating a
# SfincsModel at all, to skip its scratch-directory side effect) -- opposite
# of every actual SfincsModel's own sf.grid.data convention. See
# 09b_grid_align_landuse.py's own module docstring for the full mechanism
# (found 2026-08-07e after a real pipeline run: zsini came out "rotated"
# and implausibly small because of exactly this row-order mismatch).
#
# landuse_on_grid.tif is the ONLY sea/land/roughness classification any
# downstream consumer resamples from from here on -- rules
# modelled_depth_estimation/empirical_depth_estimation (10, weir tracing +
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
        landuse                = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse.tif"),
        grid_resolution        = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_grid_resolution.json"),
        matching_lu_roughness  = catalogue_path("lu_to_roughness_lookup"),
        land_polygons          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
    output:
        landuse_on_grid          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_on_grid.tif"),
        roughness_on_grid        = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_roughness_on_grid.tif"),
        plot_landuse_on_grid     = results_path("{basin_id}/preprocessing_inputs/visuals/09b_landuse_on_grid.png"),
        plot_roughness_on_grid   = results_path("{basin_id}/preprocessing_inputs/visuals/09b_roughness_on_grid.png"),
    log:
        "logs/{basin_id}/09b_grid_align_landuse.log"
    script:
        "../scripts/09b_grid_align_landuse.py"
