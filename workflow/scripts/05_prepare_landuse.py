"""Write the basin land-use raster in pipeline codes and its land mask; run by rule prepare_landuse.

Codes: source classes + 200 = open sea, 255 = nodata; WGS84 at the source's own
resolution, clipped to the domain bbox. ESA WorldCover has no sea class; 200 is
derived in src.landuse. The land mask (land use != 200, vectorised) is the
background of figures made before the SFINCS grid exists; nothing computational
reads it.
"""

from pathlib import Path

from src.domain import load_domain
from src.landuse import write_landuse_source
from src.log import setup_logging
from src.profiling import ScriptProfiler
from src.raster import find_fathomdem_tiles, vectorize_land_from_landuse

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
write_landuse_source = profiler.wrap(write_landuse_source)
vectorize_land_from_landuse = profiler.wrap(vectorize_land_from_landuse)

wgs84_bounds, _, _ = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
source = str(snakemake.params.source)
worldcover_path = snakemake.input.worldcover or None
river_network_path = snakemake.input.river_network or None
fathomdem_tiles = (
    find_fathomdem_tiles(snakemake.input.fathomdem_tiles, wgs84_bounds)
    if snakemake.input.fathomdem_tiles else None
)
log.info(f"Land-use source: {source}; domain WGS84 bounds: {wgs84_bounds}")

# Streamed: at 10 m a delta-sized window runs to 2.0 Gpx (Mississippi), so
# neither this raster nor the sea morphology is held whole in memory.
meta, stats = write_landuse_source(
    source,
    wgs84_bounds,
    snakemake.input.global_landuse,
    snakemake.output.landuse_source,
    worldcover_path=worldcover_path,
    river_network_path=river_network_path,
    river_barrier_width_factor=float(snakemake.params.river_barrier_width_factor),
    river_barrier_min_width_m=float(snakemake.params.river_barrier_min_width_m),
    fathomdem_tiles=fathomdem_tiles,
    sea_max_elevation_m=float(snakemake.params.sea_max_elevation_m),
)
log.info(
    f"Land-use window: {meta['width']}x{meta['height']} px @ "
    f"{stats.get('source_resolution_m')} m. Diagnostics: {stats}"
)
if source == "esa_worldcover":
    log.info(
        f"Open sea = water reaching the domain border, blocked by the river barrier: "
        f"{stats['bodies_kept_as_sea']} of {stats['water_bodies']:,} water bodies "
        f"({stats['bodies_at_border']} reach the border, the rest of those hold no "
        f"LC100 open-sea pixel) -> {stats['sea_px']:,} px at "
        f"{stats['source_resolution_m']} m. Connectivity resolved on a "
        f"{stats['working_resolution_m']} m working grid (factor "
        f"{stats['working_factor']}); barrier {stats['working_barrier_px']:,} cells. "
        f"{stats['inland_water_px']:,} px stay inland water (80); "
        f"{stats['land_inside_lc100_sea_px']:,} px LC100 called sea are land at the "
        f"source's own resolution."
    )
if stats["sea_px"] == 0:
    log.warning(
        "No open-sea (200) pixel in this basin's land use -- the coastal weir, "
        "zsini and the water-level boundary all key on it downstream."
    )
log.info(f"Written: {snakemake.output.landuse_source}")

land = vectorize_land_from_landuse(snakemake.output.landuse_source, wgs84_bounds)
log.info(f"Landuse land polygons: {len(land)} polygon(s) (landuse != 200)")
out_path = Path(snakemake.output.land_polygons)
out_path.parent.mkdir(parents=True, exist_ok=True)
land.to_file(out_path, driver="GPKG")
profiler.stop()
log.info(f"Written: {out_path}")
