# Rule: build the SFINCS model's regular grid, and persist its exact
# transform/shape/CRS so every later step that needs a SFINCS-resolution
# raster (rule enforce_river_monotonicity's coarse-conditioned-DEM
# resample, rule modelled_depth_estimation's depth calibration, rule 13's
# production build) targets the identical grid.
#
# create_from_region is a deterministic function of (domain polygon,
# resolution, crs rule) -- rule modelled_depth_estimation and rule 13 each call it themselves to
# get a populated sf.grid.data, reproducing the identical grid this rule
# builds; this rule's own output ({basin_id}_sfincs_grid.json) exists so
# rule 10's resample step (a plain rasterio consumer, not a full
# SfincsModel) has a transform/shape/CRS target without needing to spin up
# hydromt just to read it. The script itself calls
# hydromt.model.processes.create_grid_from_region directly (the same
# standalone function SfincsModel.grid.create_from_region delegates to
# internally) rather than instantiating a SfincsModel, so no Model root
# directory is ever created.

rule build_sfincs_grid:
    input:
        domain_gpkg = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        grid_resolution = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_grid_resolution.json"),
    output:
        sfincs_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_sfincs_grid.json"),
    log:
        "logs/{basin_id}/08c_build_sfincs_grid.log"
    script:
        "../scripts/08c_build_sfincs_grid.py"
