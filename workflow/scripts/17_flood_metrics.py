"""
17_flood_metrics.py - Compute flood metrics for a given SFINCS event run.

"""
from pathlib import Path
import pandas as pd

from src.postprocessing import compute_max_inundation, compute_risk_metrics

# inputs / params
sfincs_root = Path(snakemake.params.sfincs_root)
# Subgrid reference raster for postprocessing lives here, not physically in
# sfincs_root -- this scenario's own model only references it via a
# relative path in its own sfincs.inp (see 13_build_sfincs.py).
skeleton_root = Path(snakemake.params.skeleton_root)

da_hmax, da_dep = compute_max_inundation(
    sfincs_root,                    # run_dir: event output lives in the model root
    skeleton_root,
    snakemake.input.sea_mask,
    hmin=float(snakemake.params.hmin),
    include_subgrid=snakemake.params.include_subgrid,
)
if da_hmax is None:
    raise RuntimeError("zsmax/bed level unavailable — did the event run (rule 16) finish?")

tif_path = Path(snakemake.output.flood_map_tif)
tif_path.parent.mkdir(parents=True, exist_ok=True)
da_hmax.raster.to_raster(str(tif_path), dtype="float32", nodata=-9999.0)

# water_retention/water_retention_greening's own retention zone gets
# excluded from every risk metric below (see compute_risk_metrics' own
# exclude_geom_path docstring) -- water intentionally captured there is the
# measure doing its job, not flood risk, and left unmasked it inflates
# flooded_area_km2/urban_exposed_km2 identically to real damage elsewhere.
# Only set for an adapted pre-strategy run that actually has one of those
# measures -- rule adapt_flood_metrics_pre declares strategy_def/
# adaptation_root; rule compute_flood_metrics (the plain baseline scenario,
# no strategy at all) does not, so getattr's default keeps this script
# running exactly as before for that case (and for every other strategy).
strategy_def = getattr(snakemake.params, "strategy_def", None)
exclude_geom_path = None
if strategy_def is not None:
    retention_measure = strategy_def["measures"].get(
        "water_retention", strategy_def["measures"].get("water_retention_greening")
    )
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