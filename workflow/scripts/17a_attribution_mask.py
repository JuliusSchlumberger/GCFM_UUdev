"""Classify a basin x scenario flood map by dominant source and size water retention.

Run by rule attribution_mask. Class 4 is the water the embanked system holds
at its protection standard: the max flood depth of rule 11's last crest
calibration round (design discharge and coastal storm tide, present-day sea
level), derived into the output folder as rule compute_flood_metrics does.

baseline_excess_volume (m3) = flood volume of classes (1, 3), river and
compound, in the unadapted flood map; class 4 is excluded, so measures are
sized against the water beyond what the dikes hold. Written to
baseline_excess_volume.json for water_retention sizing in rules
adapt_apply_pre and adapt_metrics_post.
"""

import json
from pathlib import Path

from src.attribution_plot import generate_attribution_maps
from src.log import setup_logging
from src.postprocessing import (
    compute_excess_volume,
    compute_max_inundation,
    read_land_bed_level,
)
from src.raster import FLOAT_GTIFF_COMPRESSION, FLOAT_NODATA

log = setup_logging(snakemake.log[0])

skeleton_root = Path(snakemake.params.skeleton_root)
reference_root = Path(snakemake.params.reference_root)
threshold = float(snakemake.params.hmin)

out_folder = Path(snakemake.output.attribution_mask_tif).parent
out_folder.mkdir(parents=True, exist_ok=True)

da_hmax_reference, _ = compute_max_inundation(
    reference_root,
    read_land_bed_level(snakemake.input.land_bed_level),
    hmin=threshold,
)
if da_hmax_reference is None:
    raise RuntimeError(
        f"zsmax/bed level unavailable in {snakemake.input.reference_map_nc} -- "
        "did rule modelled_depth_estimation finish?"
    )
reference_tif = out_folder / "calibration_max_flood_depth.tif"
da_hmax_reference.raster.to_raster(
    str(reference_tif), dtype="float32", nodata=FLOAT_NODATA, **FLOAT_GTIFF_COMPRESSION
)
log.info(f"Class 4 reference max flood depth written: {reference_tif} (from {reference_root})")

log.info(
    f"Building attribution mask for scenario={snakemake.wildcards.scenario!r} "
    f"(river={snakemake.input.river_tif}, coastal={snakemake.input.coastal_tif}, "
    f"reference={reference_tif})"
)

generate_attribution_maps(
    base_root=skeleton_root,
    attribution_runs=[
        (
            snakemake.wildcards.scenario,
            Path(snakemake.input.river_tif),
            Path(snakemake.input.coastal_tif),
            reference_tif,
            out_folder,
            Path(snakemake.output.attribution_mask_png),
        )
    ],
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
log.info(
    f"baseline_excess_volume={baseline_excess_volume:.0f} m3 written: {snakemake.output.baseline_excess_volume}"
)
