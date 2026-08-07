from pathlib import Path

import numpy as np
import rasterio

from src.domain import load_domain
from src.log import setup_logging
from src.plots import plot_landuse, plot_sea_mask
from src.profiling import ScriptProfiler
from src.raster import reproject_to_reference_grid

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
reproject_to_reference_grid = profiler.wrap(reproject_to_reference_grid)

wgs84_bounds, domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}")

# Reproject onto elevation_merged.tif's exact UTM grid rather than landuse's
# own native WGS84 resolution -- so landuse/roughness share one pixel grid
# with elevation/zsini instead of each downstream consumer reprojecting
# landuse independently.
with rasterio.open(snakemake.input.elevation_merged) as ref:
    ref_meta = ref.meta.copy()
    elevation_arr = ref.read(1).astype(np.float32)
    elevation_nodata = ref.nodata

data, out_meta = reproject_to_reference_grid(
    snakemake.input.global_landuse, wgs84_bounds, ref_meta
)

Path(snakemake.output.spec_landuse).parent.mkdir(parents=True, exist_ok=True)
with rasterio.open(snakemake.output.spec_landuse, "w", **out_meta) as dst:
    dst.write(data, 1)
log.info(f"Written: {snakemake.output.spec_landuse}")

# ── sea mask: landuse-only classification ─────────────────────────────────────
# landuse==200 (sea) alone, on elevation_merged.tif's exact grid (the same
# grid landuse was just reprojected to) -- 1.0 (sea) vs nodata (land). Any
# cell with no elevation data (outside the domain polygon / no DEM+GEBCO
# coverage) is excluded. This is a pure classification -- it carries no
# water-level value. The actual initial water level (zsini.tif, sea cells =
# baseline_m) is built later in rule build_sfincs (13), which is the first
# point in the pipeline baseline_m is actually known.
#
# 2026-08-06: previously unioned with OSM land polygons (~land_mask |
# lu_arr==200) so a cell OSM didn't consider land was also sea regardless of
# its own landuse code. Dropped -- OSM's own coastline data disagreed with
# landuse at this basin's tidal flats/lagoons (landuse==80, "inland water",
# not 200), and since src.protection_weir's own ocean_mask (used to trace
# the coastal protection weir) has only ever checked landuse==200, that
# mismatch let zsini start cells wet that the weir never walled off against
# -- already-flooded land at t=0 with no barrier. Landuse-only makes
# zsini and the weir agree by construction: they were already reading the
# same landuse==200 criterion, this just stops sea_mask from disagreeing
# with it via a second, inconsistent source.
lu_arr = data[0] if data.ndim == 3 else data
NODATA = np.float32(-9999.0)
sea_mask = lu_arr == 200
sea_mask_arr = np.where(sea_mask, np.float32(1.0), NODATA).astype(np.float32)

if elevation_nodata is not None:
    sea_mask_arr[elevation_arr == np.float32(elevation_nodata)] = NODATA
sea_mask_arr[~np.isfinite(elevation_arr)] = NODATA

sea_mask_meta = ref_meta.copy()
sea_mask_meta.update(nodata=float(NODATA))

n_water = int((sea_mask_arr == 1.0).sum())
log.info(f"sea_mask: {n_water:,} sea px (landuse==200 only, no OSM land-polygon input)")

with rasterio.open(snakemake.output.sea_mask, "w", **sea_mask_meta) as zdst:
    zdst.write(sea_mask_arr, 1)
log.info(f"Written: {snakemake.output.sea_mask}")

# ── plots ──────────────────────────────────────────────────────────────────────
# land_polygons here is rule get_land_polygons' (03) own output -- landuse-
# derived since 2026-08-06 (see that rule's own module docstring), NOT OSM
# -- used only for these plots' own background overlay, same as every other
# consumer project-wide.
plot_landuse(
    snakemake.output.spec_landuse, domain_poly,
    snakemake.input.land_polygons, snakemake.output.plot_landuse,
    water_bodies_path=snakemake.output.spec_landuse,
)
plot_sea_mask(
    sea_mask_path=snakemake.output.sea_mask,
    bbox_poly=domain_poly,
    osm_land_path=snakemake.input.land_polygons,
    output_path=snakemake.output.plot_sea_mask,
    water_bodies_path=snakemake.output.spec_landuse,
)
profiler.stop()
log.info("Done")
