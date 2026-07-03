from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize as rio_rasterize

from src.domain import load_domain
from src.log import setup_logging
from src.plots import plot_landuse, plot_zsini
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
# landuse independently (and risking a land/sea split that disagrees with
# the one already baked into elevation/zsini from OSM land polygons).
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

# ── zsini: land/sea split + water-body override ──────────────────────────────
# Rasterise OSM land polygons onto elevation_merged.tif's exact grid (the
# same grid landuse was just reprojected to) to get the initial land/sea
# split: 0.0 m (open water at model start) at sea, nodata (dry) on land.
# Cells with landuse==200 (permanent inland water body) are then overridden
# to 0.0 regardless of the land polygon, and any cell with no elevation data
# (outside the domain polygon / no DEM+GEBCO coverage) is excluded.
land_gdf = gpd.read_file(snakemake.input.land_polygons).to_crs(ref_meta["crs"])
if land_gdf.empty:
    land_mask = np.zeros((ref_meta["height"], ref_meta["width"]), dtype=bool)
    log.warning("No land polygons — zsini is all-sea (within domain)")
else:
    land_mask = rio_rasterize(
        shapes=[(geom, 1) for geom in land_gdf.geometry if geom is not None],
        out_shape=(ref_meta["height"], ref_meta["width"]),
        transform=ref_meta["transform"],
        fill=0, dtype=np.uint8, all_touched=False,
    ).astype(bool)

lu_arr = data[0] if data.ndim == 3 else data
NODATA = np.float32(-9999.0)
sea_mask = ~land_mask | (lu_arr == 200)
zsini_arr = np.where(sea_mask, np.float32(0.0), NODATA).astype(np.float32)

if elevation_nodata is not None:
    zsini_arr[elevation_arr == np.float32(elevation_nodata)] = NODATA
zsini_arr[~np.isfinite(elevation_arr)] = NODATA

zsini_meta = ref_meta.copy()
zsini_meta.update(nodata=float(NODATA))

n_wb = int(((lu_arr == 200) & land_mask).sum())
n_water = int((zsini_arr == 0.0).sum())
log.info(
    f"zsini: {n_water:,} initial-water px "
    f"({n_wb:,} inland water-body px added inside the land mask)"
)

with rasterio.open(snakemake.output.zsini, "w", **zsini_meta) as zdst:
    zdst.write(zsini_arr, 1)
log.info(f"Written: {snakemake.output.zsini}")

# ── plots ──────────────────────────────────────────────────────────────────────
plot_landuse(
    snakemake.output.spec_landuse, domain_poly,
    snakemake.input.land_polygons, snakemake.output.plot_landuse,
    water_bodies_path=snakemake.output.spec_landuse,
)
plot_zsini(
    zsini_path=snakemake.output.zsini,
    bbox_poly=domain_poly,
    osm_land_path=snakemake.input.land_polygons,
    output_path=snakemake.output.plot_zsini,
    water_bodies_path=snakemake.output.spec_landuse,
)
profiler.stop()
log.info("Done")
