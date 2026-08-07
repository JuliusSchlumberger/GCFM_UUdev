"""
03_get_land_polygons.py — Vectorize the global landuse raster (landuse != 200,
i.e. not sea) clipped to the basin model domain, in its native WGS84
resolution/CRS.

2026-08-06: replaces the previous OSM land-polygons clip entirely (OSM's own
coastline data disagreed with landuse at some basins' tidal flats/lagoons --
see CHANGELOG). Every other consumer of "land_polygons" project-wide (every
diagnostic plot's own background overlay, plus the two real, functional
exclude_polygon uses in 10_depth_estimation_modelled.py/
13_build_sfincs_skeleton.py's own waterlevel boundary mask) now traces back
to this same landuse==200 criterion -- the same one sea_mask.tif (rule 05b)
and src.protection_weir's own ocean_mask already use -- so there is exactly
one source of truth for "is this land or sea" everywhere, not several
independently-derived opinions that can disagree with each other.
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
land = vectorize_land_from_landuse(snakemake.input.global_landuse, wgs84_bounds)
log.info(f"Landuse land polygons: {len(land)} polygon(s) (landuse != 200)")

# ── write ─────────────────────────────────────────────────────────────────────
out_path = Path(snakemake.output.land_polygons)
out_path.parent.mkdir(parents=True, exist_ok=True)
land.to_file(out_path, driver="GPKG")
profiler.stop()
log.info(f"Written: {out_path}")
