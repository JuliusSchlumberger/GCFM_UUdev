# Rule: run the scenario's SFINCS event from build_sfincs's sfincs.inp, then the
# rule 15 diagnostics for this run plus a per-timestep flooded-area/volume CSV.

rule run_event:
    input:
        sfincs_inp           = results_path("{basin_id}/runs/{scenario}/sfincs/sfincs.inp"),
        # Only when spin-up is enabled; build_sfincs sets rstfile to match.
        rstart               = lambda wildcards: (
            results_path(f"{wildcards.basin_id}/spin_up/" + RST_FNAME)
            if config["sfincs"]["spinup"]["enabled"] else []
        ),
        # Plot background.
        land_mask_on_grid    = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_land_mask_on_grid.gpkg"),
        # Corrected sea/land classification; weir-protected cells are land.
        sea_mask             = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
        domain_gpkg          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        clean_river_network  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
    output:
        sfincs_map_nc              = results_path("{basin_id}/runs/{scenario}/sfincs/sfincs_map.nc"),
        plot_inundation_ratio      = results_path("{basin_id}/runs/{scenario}/visuals/01_inundation_ratio.png"),
        animation_flood_progress   = results_path("{basin_id}/runs/{scenario}/visuals/02_flood_animation.mp4"),
        flood_timeseries_csv       = results_path("{basin_id}/runs/{scenario}/visuals/flood_timeseries.csv"),
    params:
        sfincs_root                = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{wildcards.scenario}/sfincs"),
        # The subgrid reference raster lives in the skeleton; sfincs_root only references it.
        skeleton_root              = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        **run_event_params(),
    # Whole --cores budget, as rule run_spinup.
    threads: workflow.cores
    log:
        "logs/{basin_id}/runs/{scenario}/16_run_event.log"
    script:
        "../scripts/16_run_event.py"
