# Flood attribution: classify flooding by source (river/coast). The matching
# river-only and coastal-only scenarios must be defined in scenarios.yml.
# Not yet tested on a rotated grid (sfincs.grid.rotated defaults to true); only
# rules 02-17 were validated, on basin 620947. On a rotated grid dep_subgrid.tif,
# max_flood_depth.tif and attribution_mask.tif are rotated GeoTIFFs: check the
# first run, or set sfincs.grid.rotated_overrides: {<basin_id>: false}.

rule attribution_mask:
    input:
        river_tif    = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{attribution_counterparts(wildcards.scenario)[0]}/visuals/max_flood_depth.tif"),
        coastal_tif  = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{attribution_counterparts(wildcards.scenario)[1]}/visuals/max_flood_depth.tif"),
        # Spin-up (RP=1 river, calm sea) has no max_flood_depth.tif; the script
        # derives one from sfincs_map.nc as rule 17 does.
        spinup_map_nc = results_path("{basin_id}/spin_up/sfincs_map.nc"),
        sea_mask      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
        skeleton_inp  = results_path("{basin_id}/sfincs_skeleton/sfincs.inp"),  # mod_ref, for the diagnostic plot's own basemap
        # Plot background.
        land_mask_on_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
        # Unadapted flood map, source of baseline_excess_volume.
        baseline_flood_map = results_path("{basin_id}/runs/{scenario}/visuals/max_flood_depth.tif"),
    output:
        attribution_mask_tif = results_path("{basin_id}/runs/{scenario}/attribution_mask.tif"),
        attribution_mask_png = results_path("{basin_id}/runs/{scenario}/attribution_mask.png"),
        # classes=(1,3), i.e. everything but pure-coastal; read by rules 18a and
        # 18b for water_retention sizing.
        baseline_excess_volume = results_path("{basin_id}/runs/{scenario}/baseline_excess_volume.json"),
    params:
        skeleton_root   = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        spin_up_root    = lambda wildcards: results_path(f"{wildcards.basin_id}/spin_up"),
        hmin = config["metrics"]["hmin"],
    log: "logs/{basin_id}/runs/{scenario}/18c_attribution_mask.log"
    script: "../scripts/18c_attribution_mask.py"
