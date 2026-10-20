"""Build and persist the shared SFINCS grid definition; run by rule build_sfincs_grid.

Subdivides rule 02's grid frame at the basin resolution; rules 09, 09b, 10 and 13
all use this grid. The JSON holds x0, y0, dx, dy, mmax, nmax, rotation, epsg and
the affine transform/shape in SfincsModel orientation (row 0 on the origin side,
i.e. y ascending for an unrotated grid). No SfincsModel is instantiated: it would
create an empty model root directory.
"""

import json
from pathlib import Path

import geopandas as gpd

from src.grid import build_grid_def
from src.log import setup_logging

log = setup_logging(snakemake.log[0])

resolution_m = float(snakemake.params.resolution_m)

delta_domain = gpd.read_file(snakemake.input.domain_gpkg)
with open(snakemake.input.spec_basins_meta) as fh:
    grid_frame = json.load(fh)["grid_frame"]

grid_def = build_grid_def(delta_domain, grid_frame, resolution_m)
log.info(
    f"Shared SFINCS grid: {grid_def['nmax']}x{grid_def['mmax']} cells @ {resolution_m} m, "
    f"CRS={grid_def['crs']}, origin ({grid_def['x0']:.0f}, {grid_def['y0']:.0f}), "
    f"rotation {grid_def['rotation']:.3f} deg"
)

Path(snakemake.output.sfincs_grid).parent.mkdir(parents=True, exist_ok=True)
with open(snakemake.output.sfincs_grid, "w") as fh:
    json.dump(grid_def, fh, indent=2)
log.info(f"Written: {snakemake.output.sfincs_grid}")
log.info("Done")
