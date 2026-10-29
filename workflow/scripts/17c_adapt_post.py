"""Apply a strategy's measures to the baseline max flood depth raster, without SFINCS.

Run by rule adapt_metrics_post (cheap, approximate). Measures are chained: each
reads the previous max_flood_depth.tif plus scenario_root/attribution_mask.tif
and writes output_dir/max_flood_depth.tif, which is output.flood_map_tif.
"""

import json
from pathlib import Path

import pandas as pd
import rioxarray as rxr

from src.adaptation_method_post import dispatch_rules
from src.adaptation_scaling import resolve_forcing_params
from src.domain import load_domain
from src.log import setup_logging
from src.plots import plot_inundation_check, plot_max_inundation_map
from src.postprocessing import (
    compute_max_inundation,
    compute_risk_metrics,
    read_land_bed_level,
)

log = setup_logging(snakemake.log[0])

# Params
measures_def = snakemake.params.measures_def
adaptation_root = Path(snakemake.params.adaptation_root)
baseline_sfincs_root = Path(snakemake.params.baseline_sfincs_root)
hmin = float(snakemake.params.hmin)
urban_code = int(snakemake.params.urban_code)

# Same resolver as 17b_adapt_pre.py, so pre and post see the same crest/discharge values.
strategy_def = resolve_forcing_params(
    snakemake.params.strategy_def,
    baseline_sfincs_root=baseline_sfincs_root,
    baseline_map_nc=snakemake.input.baseline_sfincs_map_nc,
    skeleton_weir=snakemake.input.skeleton_weir,
    adaptation_root=adaptation_root,
    log=log,
)

# Inputs baseline_sfincs_map_nc, attribution_mask_tif and measure_data are for
# dependency tracking only; they are read indirectly via baseline_sfincs_root.

output_dir = Path(snakemake.output.metrics_csv).parent
output_dir.mkdir(parents=True, exist_ok=True)

# Thread the flood map path (and, for retreat, the landuse path) through the measures.
flood_map_path = str(snakemake.input.baseline_flood_map_tif)
landuse_path = str(snakemake.input.landuse)

for measure_type, raw_params in strategy_def["measures"].items():
    measure_def = measures_def[measure_type]
    resolved = {
        k: (str(adaptation_root / v) if k == "locations" and isinstance(v, str) else v)
        for k, v in raw_params.items()
    }
    if measure_type == "water_retention":
        # From rule attribution_mask, the same reference the pre method reads.
        with open(snakemake.input.baseline_excess_volume) as fh:
            resolved["baseline_excess_volume"] = json.load(fh)["baseline_excess_volume"]
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

# da_dep (land-domain reference grid) is not on disk; re-derive it as rule 16 does.
_, da_dep = compute_max_inundation(
    baseline_sfincs_root,
    read_land_bed_level(snakemake.input.land_bed_level),
    hmin=hmin,
)
if da_dep is None:
    raise RuntimeError(
        "bed level unavailable — did the baseline event run (rule 15) finish?"
    )

# masked=True: measures write a literal nodata (e.g. -9999) and
# compute_risk_metrics treats every non-null cell as flooded.
da_hmax = rxr.open_rasterio(flood_map_path, masked=True).squeeze(drop=True)

metrics = compute_risk_metrics(
    da_hmax,
    da_dep,
    landuse_path,  # original landuse.tif, unless retreat reclassified it above
    snakemake.input.delta_polygon,
    urban_code=urban_code,
)

# Inundation ratio plot as rule 15's Check 1, from the adapted raster (no zsmax here).
n_land = int(da_dep.notnull().sum().item())
n_flooded = int(da_hmax.notnull().sum().item())
plot_inundation_check(
    da_hmax,
    hmin,
    n_flooded,
    n_land,
    str(snakemake.input.land_mask_on_grid),
    str(snakemake.input.clean_river_network),
    str(snakemake.output.plot_inundation_ratio),
    basin_id=snakemake.wildcards.basin_id,
    run_label=snakemake.params.run_label,
)
log.info(f"Inundation ratio plot written: {snakemake.output.plot_inundation_ratio}")
plot_max_inundation_map(
    da_hmax,
    load_domain(snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg)[2],
    snakemake.input.land_mask_on_grid,
    snakemake.input.clean_river_network,
    snakemake.output.plot_max_depth,
    basin_id=snakemake.wildcards.basin_id,
    run_label=snakemake.params.run_label,
)

row = {
    "basin_id": snakemake.wildcards.basin_id,
    "scenario": snakemake.wildcards.scenario,
    "strategy": snakemake.wildcards.strategy,
    "method": "post",
    **metrics,
}
pd.DataFrame([row]).to_csv(snakemake.output.metrics_csv, index=False)
