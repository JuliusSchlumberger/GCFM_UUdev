"""Classify a basin x scenario flood map by dominant source and size water retention.

Run by rule attribution_mask. Derives the spin-up max flood depth (as rule
compute_flood_metrics does) into the output folder, since run_spinup writes
none.

baseline_excess_volume (m3) = excess volume of classes (1, 3), river and
compound, in the unadapted flood map; class 4 (spin-up permanent water) is
excluded so measures are never sized against ordinary baseflow. Written to
baseline_excess_volume.json for water_retention sizing in rules
adapt_apply_pre and adapt_metrics_post.
"""

import json
from pathlib import Path

from src.attribution_plot import generate_attribution_maps
from src.log import setup_logging
from src.postprocessing import compute_excess_volume, compute_max_inundation, read_land_bed_level
from src.raster import FLOAT_GTIFF_COMPRESSION, FLOAT_NODATA

log = setup_logging(snakemake.log[0])

skeleton_root = Path(snakemake.params.skeleton_root)
spin_up_root  = Path(snakemake.params.spin_up_root)
threshold = float(snakemake.params.hmin)

out_folder = Path(snakemake.output.attribution_mask_tif).parent
out_folder.mkdir(parents=True, exist_ok=True)

da_hmax_spinup, _ = compute_max_inundation(
    spin_up_root, read_land_bed_level(snakemake.input.land_bed_level),
    hmin=threshold,
)
if da_hmax_spinup is None:
    raise RuntimeError(
        f"zsmax/bed level unavailable in {snakemake.input.spinup_map_nc} -- "
        "did the spin-up (rule run_spinup) finish?"
    )
spinup_tif = out_folder / "spinup_max_flood_depth.tif"
da_hmax_spinup.raster.to_raster(
    str(spinup_tif), dtype="float32", nodata=FLOAT_NODATA, **FLOAT_GTIFF_COMPRESSION
)
log.info(f"Spin-up max flood depth written: {spinup_tif}")

log.info(
    f"Building attribution mask for scenario={snakemake.wildcards.scenario!r} "
    f"(river={snakemake.input.river_tif}, coastal={snakemake.input.coastal_tif}, "
    f"spinup={spinup_tif})"
)

generate_attribution_maps(
    base_root=skeleton_root,
    attribution_runs=[(
        snakemake.wildcards.scenario,
        Path(snakemake.input.river_tif),
        Path(snakemake.input.coastal_tif),
        spinup_tif,
        out_folder,
    )],
    land_mask_path=snakemake.input.land_mask_on_grid,
    threshold=threshold,
)

baseline_excess_volume = compute_excess_volume(
    flood_map_path=snakemake.input.baseline_flood_map,
    attribution_mask_path=snakemake.output.attribution_mask_tif,
    classes=(1, 3),
)
with open(snakemake.output.baseline_excess_volume, "w") as fh:
    json.dump({"baseline_excess_volume": baseline_excess_volume}, fh)
log.info(f"baseline_excess_volume={baseline_excess_volume:.0f} m3 written: {snakemake.output.baseline_excess_volume}")
