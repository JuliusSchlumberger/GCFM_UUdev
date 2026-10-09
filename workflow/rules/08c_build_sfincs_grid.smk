# Rule: subdivide the basin's grid frame into the SFINCS model's regular
# grid and persist it ({basin_id}_sfincs_grid.json: x0/y0/dx/dy/mmax/nmax/
# rotation/epsg plus the equivalent affine transform and shape).
#
# This file is THE grid definition: rule enforce_river_monotonicity (09),
# rule grid_align_landuse (09b), rule modelled_depth_estimation (10) and
# rule build_sfincs_skeleton (13) all read it -- the two SfincsModels are
# created from it (src.grid.create_model_grid), nothing refits the grid.
#
# The grid frame comes from rule determine_model_domain (02,
# domain_bbox.json's grid_frame): a rotated rectangle when
# sfincs.grid.rotated, otherwise the axis-aligned grid is fitted here to the
# domain polygon's bounds, snapped to the resolution. The resolution is the
# basin's own fixed main-grid dx (sfincs.grid.resolution_m /
# resolution_overrides_m, see grid_resolution_m in 00_common.smk).

rule build_sfincs_grid:
    input:
        domain_gpkg = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        spec_basins_meta = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
    output:
        sfincs_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_sfincs_grid.json"),
    params:
        resolution = lambda wildcards: grid_resolution_m(wildcards.basin_id),
    log:
        "logs/{basin_id}/08c_build_sfincs_grid.log"
    script:
        "../scripts/08c_build_sfincs_grid.py"
