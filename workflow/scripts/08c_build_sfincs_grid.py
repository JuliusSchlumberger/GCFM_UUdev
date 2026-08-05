"""
08c_build_sfincs_grid.py -- Build the SFINCS model's regular grid ONCE and
persist its exact transform/shape/CRS, so every later step needing a
SFINCS-resolution raster (rule enforce_river_monotonicity's coarse
conditioned-DEM resample, rule modelled_depth_estimation's modelled depth
calibration, rule 13's production build) targets the identical grid
instead of each independently calling sf.grid.create_from_region(...).

create_from_region is a deterministic function of (domain polygon,
resolution, "utm" CRS rule) -- rule modelled_depth_estimation and rule 13 still each instantiate
their own SfincsModel and call it themselves to get a populated
sf.grid.data, reproducing the identical grid this rule already built; this
rule's own output is a small JSON (transform, shape, CRS, resolution) so a
plain rasterio consumer (rule 10's resample step) can target the same grid
without needing to spin up hydromt just to read it.

Calls hydromt.model.processes.create_grid_from_region directly -- the same
standalone function SfincsModel.grid.create_from_region() itself delegates
to internally -- instead of instantiating a SfincsModel, so this rule never
creates a Model root directory at all (a SfincsModel(mode="w+") would
create one on init even though nothing is ever written to it).
"""

import json
from pathlib import Path

import geopandas as gpd
from hydromt.model.processes.grid import create_grid_from_region

from src.log import setup_logging

log = setup_logging(snakemake.log[0])

with open(snakemake.input.grid_resolution) as f:
    resolution = float(json.load(f)["resolution"])

delta_domain = gpd.read_file(snakemake.input.domain_gpkg)

ds = create_grid_from_region(
    region={"geom": delta_domain}, res=resolution, crs="utm", region_crs=4326,
    rotated=False, add_mask=False, align=True,
)
transform = ds.raster.transform
height, width = ds.raster.shape
crs = ds.raster.crs
log.info(f"Shared SFINCS grid created: {resolution} m, auto-UTM CRS={crs}")

grid_def = {
    "resolution": resolution,
    "crs": crs.to_string(),
    "height": int(height),
    "width": int(width),
    # affine six-tuple (a, b, c, d, e, f) -- rasterio.Affine(*transform_six)
    # reconstructs the exact transform.
    "transform": [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
}

Path(snakemake.output.sfincs_grid).parent.mkdir(parents=True, exist_ok=True)
with open(snakemake.output.sfincs_grid, "w") as fh:
    json.dump(grid_def, fh, indent=2)
log.info(f"Written: {snakemake.output.sfincs_grid} ({height}x{width} px @ {resolution} m)")
log.info("Done")
