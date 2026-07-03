"""
03_get_land_polygons.py — Clip OSM land polygons to the domain bounding box.

The clipped file is passed to the SFINCS mask setup (rule 13 / build_sfincs.py)
as a catalog source under the key 'local_land_polygons'.  Active-domain edge
cells NOT covered by land become waterlevel boundary cells (mask=2), which
represents the coastal / open-water perimeter of the model domain.
"""

from pathlib import Path

import geopandas as gpd

from src.domain import load_domain
from src.log import setup_logging
from src.profiling import ScriptProfiler

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
gpd_read_file = profiler.wrap(gpd.read_file)

# ── domain bounds ─────────────────────────────────────────────────────────────
wgs84_bounds, _, _ = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}")

# ── clip OSM land to domain bbox ──────────────────────────────────────────────
# bbox= in read_file uses (minx, miny, maxx, maxy) — same as wgs84_bounds.
# This GeoPackage carries two layers ('marine_buffer' — an x/y tile-grid
# index — and 'land_polygons' — the actual polygon geometries). Without an
# explicit layer=, gpd.read_file silently defaults to 'marine_buffer', whose
# tile footprints rarely overlap a basin's small bbox, so the clip came back
# empty even over basins that are clearly on land.
land = gpd_read_file(snakemake.input.osm_land, bbox=wgs84_bounds, engine="pyogrio",
                     layer="land_polygons")
log.info(f"OSM land clipped: {len(land)} polygon(s)")

# ── write ─────────────────────────────────────────────────────────────────────
out_path = Path(snakemake.output.land_polygons)
out_path.parent.mkdir(parents=True, exist_ok=True)
land.to_file(out_path, driver="GPKG")
profiler.stop()
log.info(f"Written: {out_path}")
