# Rule: peak travel times from the main inflow and from the sea boundary to the
# main mouth (src.event_timing), read by build_sfincs for the compound lag and
# the event-run length.

rule event_timing:
    input:
        river_bed_nodes = results_path("{basin_id}/preprocessing_inputs/visuals/11_calibration/river_bed/{basin_id}_river_bed_nodes.csv"),
        walled_travel   = results_path("{basin_id}/preprocessing_inputs/visuals/11_calibration/river_bed/{basin_id}_walled_travel.csv"),
        river_network   = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_depth_estimated.gpkg"),
        river_forcing   = results_path("{basin_id}/preprocessing_inputs/forcing/river_forcing.nc"),
        # Its gis/dep.tif and gis/mask.tif carry the sea leg.
        skeleton_inp    = results_path("{basin_id}/sfincs_skeleton/sfincs.inp"),
    output:
        event_timing = results_path("{basin_id}/event_timing/{basin_id}_event_timing.csv"),
    params:
        basin_id          = lambda wildcards: wildcards.basin_id,
        channel_manning_n = config["river_processing"]["channel_manning_n"],
        ramp_hr           = config["river_processing"]["river_depth_modelling"]["ramp_hr"],
        wave_regime_rT    = config["river_processing"]["river_depth_modelling"]["wave_regime_rT"],
        froude_cap        = config["river_processing"]["river_depth_modelling"]["bed"]["froude_cap"],
    threads: rule_threads("event_timing")
    log:
        "logs/{basin_id}/12d_event_timing.log"
    script:
        "../scripts/12d_event_timing.py"
