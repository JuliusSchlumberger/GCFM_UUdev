import numpy as np
import rasterio

from src.domain import load_domain
from src.landuse import warp_landuse_to_grid
from src.log import setup_logging
from src.plots import plot_landuse, plot_sea_mask
from src.profiling import ScriptProfiler

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
warp_landuse_to_grid = profiler.wrap(warp_landuse_to_grid)

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

# Mode (most frequent class in each target cell), not nearest: the ESA
# WorldCover source is finer (10 m) than this grid (~30 m), so nearest would
# keep one arbitrary source pixel per cell and throw away the rest. With the
# 100 m LC100 source this is pure upsampling, where mode and nearest agree
# by construction. Classes stay categorical either way -- the CONTINUOUS
# quantity derived from them, Manning's n, is area-averaged instead (rule
# get_roughness, src.landuse.aggregate_manning). Streamed in row blocks: a
# 10 m source window reaches 2.0 Gpx on the largest delta.
histogram = warp_landuse_to_grid(
    snakemake.input.landuse_source, ref_meta, snakemake.output.spec_landuse
)
_total = sum(histogram.values())
log.info(
    "landuse.tif class histogram (% of grid): "
    + ", ".join(f"{c}={100 * n / _total:.2f}" for c, n in sorted(histogram.items()))
)
log.info(f"Written: {snakemake.output.spec_landuse}")

with rasterio.open(snakemake.output.spec_landuse) as lu_src:
    data = lu_src.read(1)

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
# land_polygons: rule get_land_polygons' (03) land mask from the native
# land-use raster (land use != 200) -- this plot's background only (the
# SFINCS grid, and its own grid-aligned mask, don't exist yet; see
# src.plots' module docstring).
plot_landuse(
    snakemake.output.spec_landuse, domain_poly,
    snakemake.input.land_polygons, snakemake.output.plot_landuse,
)
plot_sea_mask(
    sea_mask_path=snakemake.output.sea_mask,
    bbox_poly=domain_poly,
    land_polygons_path=snakemake.input.land_polygons,
    output_path=snakemake.output.plot_sea_mask,
)
profiler.stop()
log.info("Done")
