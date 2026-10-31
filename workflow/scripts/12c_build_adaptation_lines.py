"""Build a basin's adaptation measure lines from its skeleton's baseline weir.

Run by rule build_adaptation_lines (12c), once per basin; the method is in
src.adaptation_lines. Fails after writing the outputs if a mouth closure does
not seal its river.
"""

import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import rasterio
from hydromt_sfincs import SfincsModel
from shapely.geometry import LineString
from src.adaptation_lines import (
    build_adaptation_lines,
    build_info,
    measure_line_layers,
    plot_adaptation_lines,
    require_sealed,
)
from src.grid import load_grid_def
from src.log import setup_logging

log = setup_logging(snakemake.log[0])
basin_id = snakemake.wildcards.basin_id
skeleton_root = Path(snakemake.params.skeleton_root)

with rasterio.open(snakemake.input.landuse_on_grid) as src:
    landuse, transform, crs = src.read(1), src.transform, src.crs

sf = SfincsModel(root=str(skeleton_root), mode="r")
# The model's own mask, on the same y-ascending grid as landuse_on_grid
# (gis/mask.tif holds the same cells, but north-up).
sf.grid.read()
mask = sf.grid.data["mask"].values
if mask.shape != landuse.shape or not sf.grid.data.raster.transform.almost_equals(
    transform, precision=1e-6
):
    raise ValueError(
        f"skeleton grid {mask.shape} does not match landuse_on_grid {landuse.shape} "
        "-- expected pixel-identical grids."
    )

# The weir as the model holds it (and as levee measures merge against):
# hydromt_sfincs drops rule-10 segments outside the model region.
sf.weirs.read()
weir = gpd.GeoDataFrame(
    geometry=[LineString([c[:2] for c in g.coords]) for g in sf.weirs.data.geometry],
    crs=sf.crs,
).to_crs(crs)
rivers = gpd.read_file(snakemake.input.river_network).to_crs(crs)

lines = build_adaptation_lines(
    weir,
    landuse,
    mask > 0,
    transform,
    rivers,
    min_component_cells=int(snakemake.params.min_component_cells),
    unprotected_ocean_wetlands=bool(snakemake.params.unprotected_ocean_wetlands),
    closure=dict(snakemake.params.closure),
)

for name, gdf in measure_line_layers(lines).items():
    gdf.to_file(snakemake.output[name], driver="GeoJSON")
    log.info(f"Written: {snakemake.output[name]} ({len(gdf)} feature(s))")
gpkg = snakemake.output.gpkg
lines.weir.to_file(gpkg, layer="baseline_weir_labelled", driver="GPKG")
if not lines.closure.empty:
    lines.closure.to_file(gpkg, layer="closure", driver="GPKG")
if not lines.pump.empty:
    lines.pump.to_file(gpkg, layer="pump", driver="GPKG")
plot_adaptation_lines(snakemake.output.plot, lines, transform, basin_id)

info = build_info(
    lines,
    basin_id,
    built=datetime.now().isoformat(timespec="seconds"),
    source_weir=str(snakemake.input.sfincs_weir),
    grid_def=load_grid_def(snakemake.input.sfincs_grid),
)
Path(snakemake.output.build_info).write_text(json.dumps(info, indent=2))
log.info(f"Written: {snakemake.output.plot}, {snakemake.output.build_info}")
require_sealed(lines.report)
