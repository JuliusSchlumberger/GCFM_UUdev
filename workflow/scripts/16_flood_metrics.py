"""Compute the max flood depth raster and risk metrics of one SFINCS event run.

Run by rules compute_flood_metrics and adapt_flood_metrics_pre.
"""
from pathlib import Path
import pandas as pd

from src.postprocessing import compute_max_inundation, compute_risk_metrics, read_land_bed_level
from src.raster import FLOAT_GTIFF_COMPRESSION, FLOAT_NODATA

sfincs_root = Path(snakemake.params.sfincs_root)
# The subgrid reference raster lives here; sfincs_root only references it.

da_hmax, da_dep = compute_max_inundation(
    sfincs_root,
    read_land_bed_level(snakemake.input.land_bed_level),
    hmin=float(snakemake.params.hmin),
)
if da_hmax is None:
    raise RuntimeError("zsmax/bed level unavailable — did the event run (rule 15) finish?")

tif_path = Path(snakemake.output.flood_map_tif)
tif_path.parent.mkdir(parents=True, exist_ok=True)
da_hmax.raster.to_raster(str(tif_path), dtype="float32", nodata=FLOAT_NODATA, **FLOAT_GTIFF_COMPRESSION)

# A water_retention zone is excluded from the risk metrics: water stored there
# is the measure working, not flood risk. strategy_def is only set by rule
# adapt_flood_metrics_pre.
strategy_def = getattr(snakemake.params, "strategy_def", None)
exclude_geom_path = None
if strategy_def is not None:
    retention_measure = strategy_def["measures"].get("water_retention")
    if retention_measure is not None:
        adaptation_root = Path(snakemake.params.adaptation_root)
        exclude_geom_path = str(adaptation_root / retention_measure["locations"])

metrics = compute_risk_metrics(
    da_hmax,
    da_dep,
    snakemake.input.landuse,
    snakemake.input.delta_polygon,
    urban_code=int(snakemake.params.urban_code),
    exclude_geom_path=exclude_geom_path,
)
row = {"basin_id": snakemake.wildcards.basin_id,
       "scenario": snakemake.wildcards.scenario, **metrics}

pd.DataFrame([row]).to_csv(snakemake.output.metrics_csv, index=False)