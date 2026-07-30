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

Only the REGULAR grid is shared -- quadtree refinement (rule 13's own
quadtree branch) stays built separately inside 13_build_sfincs.py when
sfincs.grid.quadtree.enabled, since calibration never uses quadtree.
"""

import json
from pathlib import Path

import geopandas as gpd
from hydromt_sfincs import SfincsModel

from src.log import setup_logging

log = setup_logging(snakemake.log[0])

with open(snakemake.input.grid_resolution) as f:
    resolution = float(json.load(f)["resolution"])

delta_domain = gpd.read_file(snakemake.input.domain_gpkg)

sf = SfincsModel(root=snakemake.params.sfincs_grid_root, mode="w+", write_gis=False)
sf.grid.create_from_region(region={"geom": delta_domain}, res=resolution, crs="utm", rotated=False)
sf.mask.create_active(include_polygon=delta_domain)
log.info(f"Shared SFINCS grid created: {resolution} m, auto-UTM CRS={sf.crs}")

# mask is always present after mask.create_active(); "dep" only exists once
# sf.elevation.create() has been called, which this shared-grid-only step
# deliberately never does -- sf.grid.data only has
# ['y', 'x', 'spatial_ref', 'mask'] at this point, no 'dep'.
mask_da = sf.grid.data["mask"]
transform = mask_da.raster.transform
height, width = mask_da.shape

grid_def = {
    "resolution": resolution,
    "crs": sf.crs.to_string(),
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
