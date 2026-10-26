# Rule: max flood depth GeoTIFF and risk metrics per scenario from rule 16's
# event run; the GeoTIFF feeds attribution and adaptation.

rule compute_flood_metrics:
    input:
        sfincs_map_nc = results_path("{basin_id}/runs/{scenario}/sfincs/sfincs_map.nc"),
        # Raw landuse for the urban_code (50) check.
        landuse       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_channels.tif"),
        land_bed_level = results_path("{basin_id}/sfincs_skeleton/land_bed_level.nc"),
        delta_polygon = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_polygon.gpkg"),
    output:
        flood_map_tif = results_path("{basin_id}/runs/{scenario}/visuals/max_flood_depth.tif"),
        metrics_csv   = results_path("{basin_id}/runs/{scenario}/metrics/flood_metrics.csv"),
    params:
        sfincs_root     = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{wildcards.scenario}/sfincs"),
        hmin          = config["metrics"]["hmin"],
        urban_code      = config["metrics"]["urban_landuse_code"],
    threads: rule_threads("compute_flood_metrics")
    log:
        "logs/{basin_id}/{scenario}/17_flood_metrics.log"
    script:
        "../scripts/17_flood_metrics.py"
