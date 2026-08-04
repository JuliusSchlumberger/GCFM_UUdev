# Rule: empirical (formula-driven, no simulation) river channel depth
# estimate -- power-law (Leopold-Maddock) hydraulic depth, optionally
# refined with the Nienhuis/O'Brien estuarine depth model near the coast.
# Only DEFINED (not just skipped) when river_processing.depth_method ==
# "empirical" -- guarded by the module-level `if` below rather than a
# Snakemake wildcard/output-based mechanism, since this rule and its
# sibling alternative (modelled_depth_estimation, also numbered 10) write
# the SAME unified output filename (river_network_depth_estimated.gpkg); defining
# both unconditionally would make Snakemake raise AmbiguousRuleException
# for that file. Exactly one of the two is ever registered, so every
# downstream rule (13) needs no depth_method-conditional file selection.
# The estuarine blend is not independent of the power-law depth step -- it
# always runs immediately after it, over the same reach set.

if config["river_processing"]["depth_method"] == "empirical":

    rule empirical_depth_estimation:
        input:
            spec_basins_meta    = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
            domain_gpkg         = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
            clean_river_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
            land_polygons       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
            delta_polygon       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_polygon.gpkg"),
            nienhuis            = catalogue_path("nienhuis_delta_characteristics"),
            # Every basin gets a conditioned elevation and its own burned DEM
            # (see 10_depth_estimation_empirical.py's module docstring).
            elevation_conditioned = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_elevation_conditioned.tif"),
            sfincs_grid            = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_sfincs_grid.json"),
        output:
            depth_estimated_river_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_depth_estimated.gpkg"),
            river_burned_dem              = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_burned_dem.tif"),
            river_burned_dem_sfincs_grid  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_burned_dem_sfincs_grid.tif"),
            plot_river_depth                   = results_path("{basin_id}/preprocessing_inputs/visuals/10_river_depth.png"),
            plot_river_network_width_discharge = results_path("{basin_id}/preprocessing_inputs/visuals/10_river_width_q.png"),
            plot_hydraulic_relations            = results_path("{basin_id}/preprocessing_inputs/visuals/10_river_hydraulics.png"),
        params:
            hg_c = config["river_processing"]["empirical_estimation"]["hydraulic_geometry"]["c"],
            hg_f = config["river_processing"]["empirical_estimation"]["hydraulic_geometry"]["f"],
            estuarine_enabled    = config["river_processing"]["empirical_estimation"]["estuarine_depth"]["enabled"],
            max_match_dist_km    = config["river_processing"]["empirical_estimation"]["estuarine_depth"]["max_match_dist_km"],
            obrien_C             = config["river_processing"]["empirical_estimation"]["estuarine_depth"]["obrien_C"],
            obrien_alpha         = config["river_processing"]["empirical_estimation"]["estuarine_depth"]["obrien_alpha"],
            convergence_ratio_k  = config["river_processing"]["empirical_estimation"]["estuarine_depth"]["convergence_ratio_k"],
            blend_fraction       = config["river_processing"]["empirical_estimation"]["estuarine_depth"]["blend_fraction"],
            min_depth_m          = config["river_processing"]["empirical_estimation"]["estuarine_depth"]["min_depth_m"],
        log:
            "logs/{basin_id}/10_depth_estimation_empirical.log"
        script:
            "../scripts/10_depth_estimation_empirical.py"
