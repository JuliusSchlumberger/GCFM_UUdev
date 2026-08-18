# Rule: analysis-ready flood products per scenario, from rule 16's finished
# event run. Cheap postprocessing only -- never re-runs SFINCS. The GeoTIFF is
# the input for flood-source attribution and adaptation measures later.

rule compute_flood_metrics:
    input:
        sfincs_map_nc = results_path("{basin_id}/runs/{scenario}/sfincs/sfincs_map.nc"),
        # Raw landuse -- compute_risk_metrics' own urban_code (50) check,
        # unaffected by the weir's sea/land correction either way.
        landuse       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse.tif"),
        # Corrected sea/land classification (see rule run_spinup's own
        # comment) -- cells the final weir protects are cleared to "land",
        # not masked as open sea in these flood metrics.
        sea_mask      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
        delta_polygon = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_polygon.gpkg"),
    output:
        flood_map_tif = results_path("{basin_id}/runs/{scenario}/visuals/max_flood_depth.tif"),
        metrics_csv   = results_path("{basin_id}/runs/{scenario}/metrics/flood_metrics.csv"),
    params:
        sfincs_root     = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{wildcards.scenario}/sfincs"),
        skeleton_root   = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        hmin          = config["metrics"]["hmin"],
        urban_code      = config["metrics"]["urban_landuse_code"],
        include_subgrid = config["sfincs"]["subgrid"]["enabled"],
    log:
        "logs/{basin_id}/{scenario}/17_flood_metrics.log"
    script:
        "../scripts/17_flood_metrics.py"
