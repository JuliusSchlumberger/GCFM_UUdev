# Rule: burn the zbed_anchors.gpkg river-bed profile directly into a
# channel-only DEM at native (fine) resolution.
#
# Only actually scheduled when river_processing.burn_rivers.enabled = true
# (implicit via rule build_sfincs's input lambda, same pattern rule
# burn_river_bed itself already uses for conditioning.enabled — no explicit
# guard needed here). Requires conditioning.enabled = true (validated in
# 00_common.smk), since it needs zbed_anchors.gpkg.
#
# Why this exists: hydromt_sfincs's own burn_river_rect (called via
# subgrid_component.create(river_list=[...]) in 13_build_sfincs.py) processes
# the subgrid domain tile-by-tile, matching each tile's local (clipped) river
# line against the GLOBAL, un-clipped zbed_anchors points with no distance
# cutoff — confirmed (basin 4267691, Mississippi headwater reach) to turn a
# near-flat ~11.68 m zbed_anchors profile into a ~12.6-14.8 m wavy,
# non-monotonic burned bed level that doesn't track the source data. Burning
# it ourselves, once per reach using only that reach's own points and its own
# full centerline (see src/river_burn.py), avoids the bug outright. The
# output is fed to hydromt_sfincs as an ADDITIONAL, higher-priority
# elevation_list entry (13_build_sfincs.py), with elevation_merged /
# elevation_conditioned remaining the fallback for everywhere else (ocean,
# floodplain, gaps) — so this raster only needs to cover the buffered
# channel network itself, not the whole domain.

rule burn_river_dem:
    input:
        zbed_anchors    = results_path("{basin_id}/inputs/domain/{basin_id}_zbed_anchors.gpkg"),
        river_network   = results_path("{basin_id}/inputs/domain/{basin_id}_river_network_estuarine.gpkg"),
        domain_gpkg     = results_path("{basin_id}/inputs/domain/{basin_id}_domain.gpkg"),
        elevation_merged = results_path("{basin_id}/inputs/domain/{basin_id}_elevation_merged.tif"),
        global_topography_tiles = catalogue_path("fathomdem"),
        goco06s_gfc = catalogue_path("goco06s"),
        egm2008_gfc = catalogue_path("egm2008_geoid"),
    output:
        river_burned_dem = results_path("{basin_id}/inputs/domain/{basin_id}_river_burned_dem.tif"),
        plot_river_burn  = results_path("{basin_id}/visuals/input_data/11b_river_burn.png"),
    log:
        "logs/{basin_id}/11b_river_burn.log"
    script:
        "../scripts/11b_burn_river_dem.py"
