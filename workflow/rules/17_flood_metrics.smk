# Rule: analysis-ready flood products per scenario, from rule 16's finished
# event run. Cheap postprocessing only -- never re-runs SFINCS. The GeoTIFF is
# the input for flood-source attribution and adaptation measures later.

rule compute_flood_metrics:
    input:
        sfincs_map_nc = results_path("{basin_id}/scenarios/{scenario}/sfincs/sfincs_map.nc"),
        landuse       = results_path("{basin_id}/inputs/domain/{basin_id}_landuse.tif"),
        delta_polygon = results_path("{basin_id}/inputs/domain/{basin_id}_delta_polygon.gpkg"),
    output:
        flood_map_tif = results_path("{basin_id}/scenarios/{scenario}/metrics/max_flood_depth.tif"),
        metrics_csv   = results_path("{basin_id}/scenarios/{scenario}/metrics/flood_metrics.csv"),
    params:
        sfincs_root     = lambda wildcards: results_path(f"{wildcards.basin_id}/scenarios/{wildcards.scenario}/sfincs"),
        hmin          = config["metrics"]["hmin"],
        urban_code      = config["metrics"]["urban_landuse_code"],
        include_subgrid = config["sfincs"]["subgrid"]["enabled"],
    log:
        "logs/{basin_id}/{scenario}/17_flood_metrics.log"
    script:
        "../scripts/17_flood_metrics.py"
