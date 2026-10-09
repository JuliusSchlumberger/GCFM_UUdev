# flood attribution - check where flooding is coming from - river/ coast
# ensure the attirbution scenarios are listed in the scenarios run
# NOT YET TESTED ON A ROTATED GRID (2026-10-09): sfincs.grid.rotated is now
# true by default, and the attribution rule below has not been run on a rotated model grid yet --
# only the baseline chain (rules 02-17) was validated, on basin 620947. On a
# rotated grid the model rasters (dep_subgrid.tif, max_flood_depth.tif,
# attribution_mask.tif) are rotated GeoTIFFs: check the first run's numbers
# and figures, or keep a basin axis-aligned with
# sfincs.grid.rotated_overrides: {<basin_id>: false}. See src/grid.py and
# CHANGELOG.md 2026-10-09.

rule attribution_mask:
    input:
        river_tif    = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{attribution_counterparts(wildcards.scenario)[0]}/visuals/max_flood_depth.tif"),
        coastal_tif  = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{attribution_counterparts(wildcards.scenario)[1]}/visuals/max_flood_depth.tif"),
        # spin-up's own (RP=1 river, calm sea) run -- no max_flood_depth.tif of its
        # own (rule run_spinup, 14, only ever writes sfincs_map.nc + validation
        # PNGs), so this script derives its depth raster from sfincs_map.nc
        # itself, same way rule compute_flood_metrics (17) does for every
        # scenario's own visuals/max_flood_depth.tif.
        spinup_map_nc = results_path("{basin_id}/spin_up/sfincs_map.nc"),
        sea_mask      = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
        skeleton_inp  = results_path("{basin_id}/sfincs_skeleton/sfincs.inp"),  # mod_ref, for the diagnostic plot's own basemap
        # Grid-aligned land mask (rule grid_align_landuse) -- the plot's background.
        land_mask_on_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
        # this scenario's own UNCONTROLLED (no adaptation) flood map -- source
        # for baseline_excess_volume below, same run compute_flood_metrics (17)
        # already writes for every scenario
        baseline_flood_map = results_path("{basin_id}/runs/{scenario}/visuals/max_flood_depth.tif"),
    output:
        attribution_mask_tif = results_path("{basin_id}/runs/{scenario}/attribution_mask.tif"),
        attribution_mask_png = results_path("{basin_id}/runs/{scenario}/attribution_mask.png"),
        # baseline_excess_volume.json: computed HERE (once per basin x scenario,
        # classes=(1,3) i.e. everything but pure-coastal) so both
        # adapt_apply_pre (18a) and adapt_metrics_post (18b) can read the SAME
        # fixed reference number for water_retention's own baseline_excess_volume,
        # without either touching attribution_mask_tif directly.
        baseline_excess_volume = results_path("{basin_id}/runs/{scenario}/baseline_excess_volume.json"),
    params:
        skeleton_root   = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        spin_up_root    = lambda wildcards: results_path(f"{wildcards.basin_id}/spin_up"),
        hmin = config["metrics"]["hmin"],
    log: "logs/{basin_id}/runs/{scenario}/18c_attribution_mask.log"
    script: "../scripts/18c_attribution_mask.py"
