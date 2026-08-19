"""
18b_adapt_post.py -- Postprocessing adaptation method: apply a strategy's
measures directly to the baseline scenario's already-computed max flood
depth raster (cheap, approximate) instead of rerunning SFINCS.

Each measure in adaptation_method_post.dispatch_rules is FILE-based: it
reads flood_map_path + scenario_root/attribution_mask.tif, writes a new
max_flood_depth.tif under output_dir, and returns {"out_raster": str}. This
script threads that path from one measure to the next -- the first measure
reads the baseline's own max_flood_depth.tif, every later measure reads the
PREVIOUS measure's own output. Since output_dir is the same for every
measure in the strategy, every one of them writes to the same
output_dir/max_flood_depth.tif -- which IS output.flood_map_tif -- so the
final chained path already sits where it needs to be, no separate copy step.
"""

from pathlib import Path

import pandas as pd
import rioxarray as rxr

from src.adaptation_method_post import dispatch_rules
from src.log import setup_logging
from src.plots import plot_inundation_check
from src.postprocessing import compute_max_inundation, compute_risk_metrics

log = setup_logging(snakemake.log[0])

# Params -----------------------------------------------------------------------
strategy_def         = snakemake.params.strategy_def
measures_def         = snakemake.params.measures_def
adaptation_root      = Path(snakemake.params.adaptation_root)
baseline_sfincs_root = Path(snakemake.params.baseline_sfincs_root)
skeleton_root        = Path(snakemake.params.skeleton_root)
hmin                 = float(snakemake.params.hmin)
urban_code           = int(snakemake.params.urban_code)
include_subgrid      = snakemake.params.include_subgrid

# snakemake.input.baseline_sfincs_map_nc / .attribution_mask_tif / .measure_data
# are Snakemake-only dependency tracking -- read implicitly by
# compute_max_inundation(baseline_sfincs_root, ...) and by each measure's own
# scenario_root/attribution_mask.tif lookup below, never opened directly here.

output_dir = Path(snakemake.output.metrics_csv).parent
output_dir.mkdir(parents=True, exist_ok=True)

# apply each measure in the strategy, in order, threading the flood map
# path -- and, for retreat, the landuse path -- from one measure's own
# output to the next
flood_map_path = str(snakemake.input.baseline_flood_map_tif)
landuse_path = str(snakemake.input.landuse)

for measure_type, raw_params in strategy_def["measures"].items():
    measure_def = measures_def[measure_type]
    resolved = {
        k: (str(adaptation_root / v) if k in ("locations", "dep_subgrid") and isinstance(v, str) else v)
        for k, v in raw_params.items()
    }
    log.info(f"Applying measure {measure_type} with params {resolved}")

    result = dispatch_rules(
        measure_type,
        flood_map_path=flood_map_path,
        scenario_root=str(baseline_sfincs_root),
        measure_def=measure_def,
        catalog_path=None,  # TODO - remove? accepted but never used/forwarded by dispatch_rules
        output_dir=str(output_dir),
        landuse_path=landuse_path,
        method="postprocessing",
        strategy_measures=strategy_def["measures"],
        **resolved,
    )
    flood_map_path = result["out_raster"]
    landuse_path = result.get("landuse_path", landuse_path)  # only retreat sets this

# Metrics on the final chained raster 
# Only da_hmax ever gets persisted to disk -- da_dep (the land-domain
# reference grid compute_risk_metrics also needs) has to be re-derived the
# same way rule 17 does.
_, da_dep = compute_max_inundation(
    baseline_sfincs_root,
    skeleton_root,
    snakemake.input.sea_mask,
    hmin=hmin,
    include_subgrid=include_subgrid,
)
if da_dep is None:
    raise RuntimeError("bed level unavailable — did the baseline event run (rule 16) finish?")

# masked=True is required here: the ported measures write nodata via
# rasterio with a literal fill value (e.g. -9999), and compute_risk_metrics
# relies on da_hmax.notnull() to tell flooded from dry cells -- without
# masking, every dry (nodata) cell would read as "flooded".
da_hmax = rxr.open_rasterio(flood_map_path, masked=True).squeeze(drop=True)

metrics = compute_risk_metrics(
    da_hmax,
    da_dep,
    landuse_path,  # original landuse.tif, unless retreat reclassified it above
    snakemake.input.delta_polygon,
    urban_code=urban_code,
)

# Inundation ratio plot -- mirrors rule 16's own "Check 1", but built from
# this method's own final adapted raster (da_hmax/da_dep above) instead of
# a live SFINCS run's zsmax: the post method never reruns SFINCS, so there
# is no zsmax to source it from. plot_inundation_check itself only needs a
# da_hmax DataArray, so it's reused unmodified.
n_land    = int(da_dep.notnull().sum().item())
n_flooded = int(da_hmax.notnull().sum().item())
plot_inundation_check(
    da_hmax, hmin, n_flooded, n_land,
    str(snakemake.input.land_polygons), str(snakemake.input.river_network),
    str(snakemake.output.plot_inundation_ratio),
    basin_id=snakemake.wildcards.basin_id,
    water_bodies_path=str(snakemake.input.landuse),
    run_label="post-adaptation",
)
log.info(f"Inundation ratio plot written: {snakemake.output.plot_inundation_ratio}")

row = {
    "basin_id": snakemake.wildcards.basin_id,
    "scenario": snakemake.wildcards.scenario,
    "strategy": snakemake.wildcards.strategy,
    "method": "post",
    **metrics,
}
pd.DataFrame([row]).to_csv(snakemake.output.metrics_csv, index=False)
