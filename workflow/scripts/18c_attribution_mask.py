"""
18c_attribution_mask.py -- thin wrapper: classifies THIS basin x scenario's
flood map by dominant source, calling src.attribution_plot.generate_attribution_maps
for exactly one (label, river_tif, coastal_tif, spinup_tif, out_folder)
tuple -- not the original script's hardcoded 4-scenario batch call.

spinup_tif isn't produced by rule run_spinup (14) itself -- it only ever
writes sfincs_map.nc + validation PNGs, no max_flood_depth.tif -- so this
script derives it here, the same way rule compute_flood_metrics (17) builds
every scenario's own visuals/max_flood_depth.tif, and caches it in this
scenario's own attribution output folder (cheap to redo; avoids touching
rule run_spinup's own outputs for every basin x scenario attribution run).
"""

from pathlib import Path

from src.attribution_plot import generate_attribution_maps
from src.log import setup_logging
from src.postprocessing import compute_max_inundation

log = setup_logging(snakemake.log[0])

skeleton_root = Path(snakemake.params.skeleton_root)
spin_up_root  = Path(snakemake.params.spin_up_root)
threshold = float(snakemake.params.hmin)

out_folder = Path(snakemake.output.attribution_mask_tif).parent
out_folder.mkdir(parents=True, exist_ok=True)

da_hmax_spinup, _ = compute_max_inundation(
    spin_up_root, skeleton_root, snakemake.input.sea_mask,
    hmin=threshold, include_subgrid=bool(snakemake.params.include_subgrid),
)
if da_hmax_spinup is None:
    raise RuntimeError(
        f"zsmax/bed level unavailable in {snakemake.input.spinup_map_nc} -- "
        "did the spin-up (rule run_spinup) finish?"
    )
spinup_tif = out_folder / "spinup_max_flood_depth.tif"
da_hmax_spinup.raster.to_raster(str(spinup_tif), dtype="float32", nodata=-9999.0)
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
    threshold=threshold,
)
