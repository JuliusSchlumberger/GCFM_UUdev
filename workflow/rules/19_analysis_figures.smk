# Rules: the diagnostic figures (config analysis: true), drawn only from outputs
# and diagnostic files the production rules already wrote, so switching
# analysis on or off never reruns a production rule. Figures go to
# {basin_id}/figures/analysis/; rule sanity_checks (13b) draws the spin-up's.

_HYDROGRAPHS = config["boundary_forcings"]["surge"]["hydrograph"]["enabled"]


rule analysis_preprocess:
    input:
        spec_basins_meta       = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg            = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        land_polygons          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_polygons.gpkg"),
        # 03 protection levels
        delta_polygon          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_delta_polygon.gpkg"),
        protection_levels      = results_path("{basin_id}/preprocessing_inputs/domain/protection_levels.json"),
        flopros_table          = raw_input("protection_levels_flopros"),
        geogunit_raster        = raw_input("wri_geogunit_107"),
        global_landuse         = raw_input("land_use"),
        # 06 datum corrections
        geoid_offset           = diagnostics_path("geoid_offset.tif"),
        mdt_ocean              = diagnostics_path("mdt_ocean.tif"),
        datum_fathomdem        = diagnostics_path("datum_correction_fathomdem.tif"),
        datum_gebco            = diagnostics_path("datum_correction_gebco.tif"),
        # 07 boundary forcing
        hydrographs            = (
            results_path("{basin_id}/preprocessing_inputs/forcing/storm_tide_hydrographs.nc")
            if _HYDROGRAPHS else []
        ),
        surge_stations         = diagnostics_path("surge_stations.gpkg"),
        crossings              = diagnostics_path("river_crossings.gpkg"),
        bias_correction        = diagnostics_path("bias_correction.npz"),
        river_forcing          = results_path("{basin_id}/preprocessing_inputs/forcing/river_forcing.nc"),
        glofas_clip            = results_path("{basin_id}/preprocessing_inputs/forcing/glofas_clip.nc"),
        # 08 river network, 10 water surface
        spec_river_network     = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network.gpkg"),
        clean_river_network    = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
        surface_profile_points = diagnostics_path("river_surface_profile_points.csv"),
        surface_profile_paths  = diagnostics_path("river_surface_profile_paths.csv"),
    output:
        plot_protection        = analysis_figure_path("03_protection_levels.png"),
        plot_geoid_offset      = analysis_figure_path("06_datum_geoid_offset.png"),
        plot_mdt_ocean         = analysis_figure_path("06_datum_mdt_ocean.png"),
        plot_datum_fathomdem   = analysis_figure_path("06_datum_fathomdem_correction.png"),
        plot_datum_gebco       = analysis_figure_path("06_datum_gebco_correction.png"),
        **({"plot_storm_tide_shapes": analysis_figure_path("07_forcing_storm_tide_shapes.png")}
           if _HYDROGRAPHS else {}),
        plot_map               = analysis_figure_path("07_forcing_locations.png"),
        plot_eva_diagnostics   = analysis_figure_path("07_forcing_river_eva.png"),
        # One figure per GRDC-corrected GloFAS cell, named by crossing and cell.
        plot_bias_correction   = directory(analysis_figure_path("07_forcing_bias_correction")),
        plot_surge_correction  = analysis_figure_path("07_forcing_surge_correction.png"),
        plot_clean_network     = analysis_figure_path("08_river_network_clean.png"),
        plot_discharge_network = analysis_figure_path("08_river_discharge.png"),
        plot_water_surface     = analysis_figure_path("10_river_surface.png"),
    params:
        eva              = config["boundary_forcings"]["river"]["eva"],
        max_width_fill_m = config["river_processing"]["width_qa"]["max_width_fill_m"],
    threads: rule_threads("analysis_preprocess")
    log:
        "logs/{basin_id}/19a_analysis_preprocess.log"
    script:
        "../scripts/19a_analysis_preprocess.py"


def _calibration_round_inputs():
    inputs = {}
    for r in CALIB_ROUNDS:
        inputs[f"water_level_r{r}"] = diagnostics_path(f"calibration_water_level_r{r}.csv")
        inputs[f"map_r{r}"] = results_path(
            f"{{basin_id}}/preprocessing_inputs/depth_crest_calibration/round{r}/sfincs_map.nc"
        )
    return inputs


def _calibration_round_figures():
    figures = {}
    for r in CALIB_ROUNDS:
        figures[f"plot_water_level_r{r}"] = analysis_figure_path(f"11_calib_water_level__calib_r{r}.png")
        figures[f"animation_r{r}"] = analysis_figure_path(f"11_calib_flood_progression__calib_r{r}.mp4")
    return figures


rule analysis_calibration:
    input:
        spec_basins_meta        = results_path("{basin_id}/preprocessing_inputs/domain/domain_bbox.json"),
        domain_gpkg             = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        land_mask_on_grid       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
        landuse_on_grid         = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_on_grid.tif"),
        clean_river_network     = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
        coastal_protection_weir = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_coastal_protection_weir.gpkg"),
        depth_segments          = diagnostics_path("river_depth_segments.gpkg"),
        model_region            = diagnostics_path("model_region.gpkg"),
        discharge_points        = diagnostics_path("calibration_discharge_points.gpkg"),
        dike_wetlands_off       = diagnostics_path("dike_wetlands_off.gpkg"),
        dike_wetlands_on        = diagnostics_path("dike_wetlands_on.gpkg"),
        ocean_wetland_mask      = diagnostics_path("ocean_wetland_mask.tif"),
        crest_on_grid           = diagnostics_path("crest_on_grid.tif"),
        calibration_rounds      = diagnostics_path("calibration_rounds.json"),
        ocean_mask_on_grid      = results_path("{basin_id}/preprocessing_inputs/depth_crest_calibration/ocean_mask_on_grid.tif"),
        **_calibration_round_inputs(),
    output:
        plot_river_network_depth         = analysis_figure_path("11_calib_river_depth_network.png"),
        plot_river_network_forward_check = analysis_figure_path("11_calib_river_forward_check.png"),
        plot_crest_gap_map               = analysis_figure_path("11_calib_crest_gap.png"),
        plot_dike_positions              = analysis_figure_path("11_calib_dike_position_wetlands.png"),
        **_calibration_round_figures(),
    params:
        design_discharge           = config["river_processing"]["river_depth_modelling"]["bed"]["design_discharge"],
        channel_manning_n          = config["river_processing"]["channel_manning_n"],
        unprotected_ocean_wetlands = config["river_processing"]["river_depth_modelling"]["unprotected_ocean_wetlands"],
        animation_fps              = config["sfincs"]["sanity_checks"]["animation_fps"],
    threads: rule_threads("analysis_calibration")
    log:
        "logs/{basin_id}/19b_analysis_calibration.log"
    script:
        "../scripts/19b_analysis_calibration.py"


rule analysis_run_event:
    input:
        sfincs_map_nc       = results_path("{basin_id}/runs/{scenario}/sfincs/sfincs_map.nc"),
        land_bed_level      = results_path("{basin_id}/sfincs_skeleton/land_bed_level.nc"),
        land_mask_on_grid   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
        clean_river_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
        domain_gpkg         = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
    output:
        plot_inundation_ratio = analysis_figure_path("15_run_inundation_ratio__{scenario}.png"),
    params:
        run_root               = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{wildcards.scenario}/sfincs"),
        run_label              = lambda wildcards: run_label(wildcards.scenario),
        min_inundation_depth_m = config["sfincs"]["sanity_checks"]["min_inundation_depth_m"],
    threads: rule_threads("analysis_run_event")
    log:
        "logs/{basin_id}/runs/{scenario}/19c_analysis_run_event.log"
    script:
        "../scripts/13b_sanity_checks.py"
