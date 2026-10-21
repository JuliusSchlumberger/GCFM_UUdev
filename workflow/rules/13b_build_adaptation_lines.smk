# Rule: derive the adaptation measure lines (coastal/river levees, mouth
# closure, mouth pump) of a basin from its skeleton's baseline weir.
# Strategies reference them as `locations: auto/<name>`
# (config/adaptation_strategies.yml), which makes them inputs of the adapt
# rules; nothing else requests them. adaptation_lines.png is for review.

rule build_adaptation_lines:
    input:
        # Dependency on the skeleton; the script also reads its gis/mask.tif.
        sfincs_inp      = results_path("{basin_id}/sfincs_skeleton/sfincs.inp"),
        sfincs_weir     = results_path("{basin_id}/sfincs_skeleton/sfincs.weir"),
        sfincs_grid     = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_sfincs_grid.json"),
        landuse_on_grid = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_on_grid.tif"),
        river_network   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
    output:
        **{name: results_path(f"{{basin_id}}/adaptation_lines/{name}.geojson")
           for name in ADAPTATION_LINE_NAMES},
        gpkg       = results_path("{basin_id}/adaptation_lines/adaptation_lines.gpkg"),
        plot       = results_path("{basin_id}/adaptation_lines/adaptation_lines.png"),
        build_info = results_path("{basin_id}/adaptation_lines/build_info.json"),
    params:
        skeleton_root = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        # Same settings as rule 10's weir builder, so the labels match its weir.
        min_component_cells        = config["river_processing"]["river_depth_modelling"]["min_component_cells"],
        unprotected_ocean_wetlands = config["river_processing"]["river_depth_modelling"]["unprotected_ocean_wetlands"],
        closure                    = config["adaptation"]["lines"],
    log:
        "logs/{basin_id}/13b_build_adaptation_lines.log"
    script:
        "../scripts/13b_build_adaptation_lines.py"
