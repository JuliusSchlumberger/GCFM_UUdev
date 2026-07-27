"""
17_flood_metrics.py - Compute flood metrics for a given SFINCS event run.

"""
from pathlib import Path
import pandas as pd
import geopandas as gpd

from src.postprocessing import compute_max_inundation, compute_risk_metrics

# inputs / params
sfincs_root = Path(snakemake.params.sfincs_root)

da_hmax, da_dep = compute_max_inundation(
    sfincs_root,                    # run_dir: event output lives in the model root
    sfincs_root,
    snakemake.input.landuse,
    hmin=float(snakemake.params.hmin),
    include_subgrid=snakemake.params.include_subgrid,
)
if da_hmax is None:
    raise RuntimeError("zsmax/bed level unavailable — did the event run (rule 16) finish?")

tif_path = Path(snakemake.output.flood_map_tif)
tif_path.parent.mkdir(parents=True, exist_ok=True)
da_hmax.raster.to_raster(str(tif_path), dtype="float32", nodata=-9999.0)

metrics = compute_risk_metrics(
    da_hmax, 
    da_dep, 
    snakemake.input.landuse,
    snakemake.input.delta_polygon,
    urban_code=int(snakemake.params.urban_code),
)
row = {"basin_id": snakemake.wildcards.basin_id,
       "scenario": snakemake.wildcards.scenario, **metrics}

pd.DataFrame([row]).to_csv(snakemake.output.metrics_csv, index=False)