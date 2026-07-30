# Rule: build the SFINCS model's regular grid, and persist its exact
# transform/shape/CRS so every later step that needs a SFINCS-resolution
# raster (rule enforce_river_monotonicity's coarse-conditioned-DEM
# resample, rule modelled_depth_estimation's depth calibration, rule 13's
# production build) targets the identical grid.
#
# Only the REGULAR grid is shared -- quadtree refinement (rule 13's own
# quadtree branch) is unrelated to calibration (which never uses quadtree,
# see its own NotImplementedError guard) and is built separately inside
# 13_build_sfincs.py when sfincs.grid.quadtree.enabled.
#
# create_from_region is a deterministic function of (domain polygon,
# resolution, crs rule) -- rule modelled_depth_estimation and rule 13 each call it themselves to
# get a populated sf.grid.data, reproducing the identical grid this rule
# builds; this rule's own output ({basin_id}_sfincs_grid.json) exists so
# rule 10's resample step (a plain rasterio consumer, not a full
# SfincsModel) has a transform/shape/CRS target without needing to spin up
# hydromt just to read it.

rule build_sfincs_grid:
    input:
        domain_gpkg = results_path("{basin_id}/inputs/domain/{basin_id}_domain.gpkg"),
        grid_resolution = results_path("{basin_id}/inputs/domain/{basin_id}_grid_resolution.json"),
    output:
        sfincs_grid = results_path("{basin_id}/inputs/domain/{basin_id}_sfincs_grid.json"),
    params:
        # Throwaway root -- SfincsModel needs a real root path even though
        # nothing is ever written here (sf.write() is never called; only
        # the in-memory grid's own transform/shape/CRS are read out).
        sfincs_grid_root = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_grid"),
    log:
        "logs/{basin_id}/08c_build_sfincs_grid.log"
    script:
        "../scripts/08c_build_sfincs_grid.py"
