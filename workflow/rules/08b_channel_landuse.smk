# Rule: land use with the river channel set to water. The channel is each cleaned
# reach buffered by width / 2 (src.river_burn.channel_polygons), the same area
# rule 10 burns the bed into; sea pixels keep class 200. Read instead of
# landuse.tif by every rule after 09, so built-up or crop pixels in the channel
# count neither as exposed nor as dry land. Rule 09 keeps the original, whose
# islands it must see to skip them.

rule channel_landuse:
    input:
        landuse       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse.tif"),
        river_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_clean.gpkg"),
    output:
        landuse_channels = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_landuse_channels.tif"),
    params:
        water_code = config["landuse"]["water_code"],
    threads: rule_threads("channel_landuse")
    log:
        "logs/{basin_id}/08b_channel_landuse.log"
    script:
        "../scripts/08b_channel_landuse.py"
