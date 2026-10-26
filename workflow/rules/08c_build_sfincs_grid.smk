# Rule: write the SFINCS regular grid definition ({basin_id}_sfincs_grid.json:
# x0/y0/dx/dy/mmax/nmax/rotation/epsg plus affine transform and shape).
#
# The single grid definition read by rules 09, 09b, 10 and 13; every
# SfincsModel is created from it and nothing refits the grid. The frame is
# domain_bbox.json's grid_frame when sfincs.grid.rotated, else the domain
# bounds snapped to the resolution.

rule build_sfincs_grid:
    input:
        domain_gpkg = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        spec_basins_meta = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
    output:
        sfincs_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_sfincs_grid.json"),
    params:
        resolution_m = lambda wildcards: grid_resolution_m(wildcards.basin_id),
    threads: rule_threads("build_sfincs_grid")
    log:
        "logs/{basin_id}/08c_build_sfincs_grid.log"
    script:
        "../scripts/08c_build_sfincs_grid.py"
