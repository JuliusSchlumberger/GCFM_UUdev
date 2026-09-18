"""
03_get_land_polygons.py — Vectorize this basin's own land-use raster
(landuse != 200, i.e. not sea; rule prepare_landuse, 02b) in its native
WGS84 resolution/CRS.

A land MASK for figure backgrounds only -- the figures made before the
SFINCS grid exists (rules 04-09, 11b). Every figure of model output uses
the grid-aligned mask rule grid_align_landuse (09b) traces from
landuse_on_grid.tif instead, and the models' own water-level boundary uses
landuse_on_grid.tif directly (src.raster.restrict_waterlevel_boundary_to_sea)
-- so nothing computational depends on this file anymore (2026-09-11). See
src.plots' module docstring.

2026-08-06: replaced the previous OSM land-polygons clip entirely (OSM's own
coastline data disagreed with landuse at some basins' tidal flats/lagoons --
see CHANGELOG); no OSM data is used anywhere since. Same landuse==200
criterion sea_mask.tif (rule 05b) and src.protection_weir's own ocean_mask
use -- and, for ESA WorldCover, 200 is what rule prepare_landuse derived
(WorldCover has no sea class; see src.landuse).
"""

from pathlib import Path

from src.domain import load_domain
from src.log import setup_logging
from src.profiling import ScriptProfiler
from src.raster import vectorize_land_from_landuse

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
vectorize_land_from_landuse = profiler.wrap(vectorize_land_from_landuse)

# ── domain bounds ─────────────────────────────────────────────────────────────
wgs84_bounds, _, _ = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}")

# ── vectorize landuse != 200 ──────────────────────────────────────────────────
land = vectorize_land_from_landuse(snakemake.input.landuse_source, wgs84_bounds)
log.info(f"Landuse land polygons: {len(land)} polygon(s) (landuse != 200)")

# ── write ─────────────────────────────────────────────────────────────────────
out_path = Path(snakemake.output.land_polygons)
out_path.parent.mkdir(parents=True, exist_ok=True)
land.to_file(out_path, driver="GPKG")
profiler.stop()
log.info(f"Written: {out_path}")
