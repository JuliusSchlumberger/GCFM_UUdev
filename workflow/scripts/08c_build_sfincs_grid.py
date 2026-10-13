"""
08c_build_sfincs_grid.py -- Build the SFINCS model's regular grid ONCE and
persist it, so every later step works on the identical grid: rule
enforce_river_monotonicity's coarse conditioned-DEM resample (09), rule
grid_align_landuse (09b), rule modelled_depth_estimation's calibration
model (10) and rule 13's production build.

The grid frame (axis-aligned, or the minimum rotated rectangle around the
domain polygon -- sfincs.grid.rotated) was fitted by rule
determine_model_domain (02); this rule subdivides it at the basin's own
resolution (src.grid.build_grid_def). The output JSON holds the SFINCS grid
parameters (x0, y0, dx, dy, mmax, nmax, rotation, epsg) and the equivalent
affine transform/shape, in SfincsModel's own orientation (row 0 on the
origin side, i.e. y ascending for an unrotated grid) -- see src/grid.py.

No SfincsModel is instantiated here (that would create a Model root
directory even though nothing is ever written to it).
"""

import json
from pathlib import Path

import geopandas as gpd

from src.grid import build_grid_def
from src.log import setup_logging

log = setup_logging(snakemake.log[0])

resolution = float(snakemake.params.resolution)

delta_domain = gpd.read_file(snakemake.input.domain_gpkg)
with open(snakemake.input.spec_basins_meta) as fh:
    grid_frame = json.load(fh)["grid_frame"]

grid_def = build_grid_def(delta_domain, grid_frame, resolution)
log.info(
    f"Shared SFINCS grid: {grid_def['nmax']}x{grid_def['mmax']} cells @ {resolution} m, "
    f"CRS={grid_def['crs']}, origin ({grid_def['x0']:.0f}, {grid_def['y0']:.0f}), "
    f"rotation {grid_def['rotation']:.3f} deg"
)

Path(snakemake.output.sfincs_grid).parent.mkdir(parents=True, exist_ok=True)
with open(snakemake.output.sfincs_grid, "w") as fh:
    json.dump(grid_def, fh, indent=2)
log.info(f"Written: {snakemake.output.sfincs_grid}")
log.info("Done")
