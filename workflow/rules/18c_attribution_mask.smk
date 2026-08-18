# flood attribution - check where flooding is coming from - river/ coast
# ensure the attirbution scenarios are listed in the scenarios run

rule attribution_mask:
    input:
        river_tif    = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{attribution_counterparts(wildcards.scenario)[0]}/visuals/max_flood_depth.tif"),
        coastal_tif  = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{attribution_counterparts(wildcards.scenario)[1]}/visuals/max_flood_depth.tif"),
        # spin-up's own RP=1 (both drivers) run -- no max_flood_depth.tif of its
        # own (rule run_spinup, 14, only ever writes sfincs_map.nc + validation
        # PNGs), so this script derives its depth raster from sfincs_map.nc
        # itself, same way rule compute_flood_metrics (17) does for every
        # scenario's own visuals/max_flood_depth.tif.
        spinup_map_nc = results_path("{basin_id}/spin_up/sfincs_map.nc"),
        sea_mask      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
        skeleton_inp  = results_path("{basin_id}/sfincs_skeleton/sfincs.inp")   # mod_ref, for the diagnostic plot's own basemap
    output:
        attribution_mask_tif = results_path("{basin_id}/runs/{scenario}/attribution_mask.tif"),
        attribution_mask_png = results_path("{basin_id}/runs/{scenario}/attribution_mask.png"),
    params:
        skeleton_root   = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        spin_up_root    = lambda wildcards: results_path(f"{wildcards.basin_id}/spin_up"),
        include_subgrid = config["sfincs"]["subgrid"]["enabled"],
        hmin = config["metrics"]["hmin"],
    log: "logs/{basin_id}/runs/{scenario}/18c_attribution_mask.log"
    script: "../scripts/18c_attribution_mask.py"
