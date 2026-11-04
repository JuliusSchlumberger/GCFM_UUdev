# Rule: derive the adaptation measure lines (coastal/river levees, mouth
# closure, mouth pump) of a basin from its skeleton's baseline weir.
# Strategies reference them as `locations: auto/<name>`
# (config/adaptation_strategies.yml), which makes them inputs of the adapt
# rules; nothing else requests them. figures/60_adapt_lines.png is for review.
# The lines go to the basin's folder under adaptation_data_root (see
# adaptation_lines_dir), next to its hand-made measure files.
_LINES_DIR = adaptation_lines_dir("{basin_id}")

rule build_adaptation_lines:
    input:
        # Dependency on the skeleton; the script reads its grid (mask) and weir.
        sfincs_inp      = results_path("{basin_id}/sfincs_skeleton/sfincs.inp"),
        sfincs_weir     = results_path("{basin_id}/sfincs_skeleton/sfincs.weir"),
        sfincs_grid     = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_sfincs_grid.json"),
        landuse_on_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_on_grid.tif"),
        river_network   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
    output:
        **{name: f"{_LINES_DIR}/{name}.geojson" for name in ADAPTATION_LINE_NAMES},
        gpkg       = f"{_LINES_DIR}/adaptation_lines.gpkg",
        plot       = figure_path("60_adapt_lines.png"),
        build_info = f"{_LINES_DIR}/build_info.json",
    params:
        skeleton_root = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        # Same settings as rule 11's weir builder, so the labels match its weir.
        min_component_cells        = config["river_processing"]["river_depth_modelling"]["min_component_cells"],
        unprotected_ocean_wetlands = config["river_processing"]["river_depth_modelling"]["unprotected_ocean_wetlands"],
        closure                    = config["adaptation"]["lines"],
    threads: rule_threads("build_adaptation_lines")
    log:
        "logs/{basin_id}/12c_build_adaptation_lines.log"
    script:
        "../scripts/12c_build_adaptation_lines.py"
