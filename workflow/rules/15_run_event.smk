# Rule: run the scenario's SFINCS event from build_sfincs's sfincs.inp; with
# analysis, also its flood animation and per-timestep flooded-area/volume CSV.

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
        land_bed_level       = results_path("{basin_id}/sfincs_skeleton/land_bed_level.nc"),
        domain_gpkg          = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_domain.gpkg"),
        clean_river_network  = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
    output:
        sfincs_map_nc              = results_path("{basin_id}/runs/{scenario}/sfincs/sfincs_map.nc"),
        # Diagnostics (analysis only): the time series rasterizes every map frame
        # onto the subgrid (~12 min per run); nothing downstream reads either.
        **({
            "animation_flood_progress": figure_path("50_run_flood_progression__{scenario}.mp4"),
            "flood_timeseries_csv": results_path("{basin_id}/runs/{scenario}/visuals/flood_timeseries.csv"),
        } if ANALYSIS else {}),
    params:
        sfincs_root                = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{wildcards.scenario}/sfincs"),
        **run_event_params(),
        run_label                  = lambda wildcards: run_label(wildcards.scenario),
    threads: rule_threads("run_event")
    log:
        "logs/{basin_id}/runs/{scenario}/15_run_event.log"
    script:
        "../scripts/15_run_event.py"
