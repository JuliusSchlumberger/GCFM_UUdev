# Rule: compute a per-basin SFINCS main-grid resolution (see
# src/grid_resolution.py and 08b_optimize_grid_resolution.py docstrings),
# instead of using a single fixed sfincs.grid.resolution value for every
# basin. Runs after clean_river_network (08) -- needs its
# bankfull_discharge_acc column -- and before build_sfincs (13), which reads
# this rule's output JSON. get_boundary_forcings (07) cannot use this rule's
# output for its own diagnostic-only resolution annotation: rule 08 depends
# on 07's river_forcing output, and this rule depends on 08's
# bankfull_discharge_acc column, so 07 -> this rule -> 08 -> 07 would be a
# cycle; 07 uses the static sfincs.grid.optimize_resolution.
# default_resolution_m value instead (see 07_get_boundary_forcings.py).

rule optimize_grid_resolution:
    input:
        domain_gpkg          = results_path("{basin_id}/inputs/domain/{basin_id}_domain.gpkg"),
        spec_basins_meta     = results_path("{basin_id}/inputs/domain/domain_bbox.json"),
        clean_river_network  = results_path("{basin_id}/inputs/domain/{basin_id}_river_network_clean.gpkg"),
    output:
        grid_resolution = results_path("{basin_id}/inputs/domain/{basin_id}_grid_resolution.json"),
    params:
        enabled                 = config["sfincs"]["grid"]["optimize_resolution"]["enabled"],
        default_resolution_m    = config["sfincs"]["grid"]["optimize_resolution"]["default_resolution_m"],
        target_cells_per_width  = config["sfincs"]["grid"]["optimize_resolution"]["target_cells_per_width"],
        max_active_cells        = config["sfincs"]["grid"]["optimize_resolution"]["max_active_cells"],
    log:
        "logs/{basin_id}/08b_optimize_grid_resolution.log"
    script:
        "../scripts/08b_optimize_grid_resolution.py"
