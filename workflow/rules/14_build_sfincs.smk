# Rule: write a scenario's forcing (sfincs.bzs/.bnd/.dis/.src) and sfincs.inp on
# top of the basin skeleton. Depends on the skeleton's sfincs.inp so geometry
# changes rerun it; changing a scenario's RP reruns neither the skeleton nor
# run_spinup.

rule build_sfincs:
    input:
        skeleton_inp      = results_path("{basin_id}/sfincs_skeleton/sfincs.inp"),
        river_network     = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_depth_estimated.gpkg"),
        surge_forcing     = results_path("{basin_id}/preprocessing_inputs/forcing/surge_forcing.nc"),
        river_forcing     = results_path("{basin_id}/preprocessing_inputs/forcing/river_forcing.nc"),
        # Only when spin-up is enabled.
        rstart = lambda wildcards: (
            results_path(f"{wildcards.basin_id}/spin_up/" + RST_FNAME)
            if config["sfincs"]["spinup"]["enabled"] else []
        ),
    output:
        sfincs_inp = results_path("{basin_id}/runs/{scenario}/sfincs/sfincs.inp"),
    params:
        **build_sfincs_params(),
        skeleton_root      = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
        sfincs_root        = lambda wildcards: results_path(f"{wildcards.basin_id}/runs/{wildcards.scenario}/sfincs"),
        spin_up_root       = lambda wildcards: results_path(f"{wildcards.basin_id}/spin_up"),
    threads: rule_threads("build_sfincs")
    log:
        "logs/{basin_id}/runs/{scenario}/14_build_sfincs.log"
    script:
        "../scripts/14_build_sfincs.py"
