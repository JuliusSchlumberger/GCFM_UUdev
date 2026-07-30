from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize as rio_rasterize

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

# ── sea mask: land/sea classification + water-body override ──────────────────
# Rasterise OSM land polygons onto elevation_merged.tif's exact grid (the
# same grid landuse was just reprojected to) to get the land/sea split: 1.0
# (sea/inland-water) vs nodata (land). Cells with landuse==200 (permanent
# inland water body) are then overridden to sea regardless of the land
# polygon, and any cell with no elevation data (outside the domain polygon /
# no DEM+GEBCO coverage) is excluded. This is a pure classification -- it
# carries no water-level value. The actual initial water level (zsini.tif,
# sea cells = baseline_m) is built later in rule build_sfincs (13), which is
# the first point in the pipeline baseline_m is actually known.
land_gdf = gpd.read_file(snakemake.input.land_polygons).to_crs(ref_meta["crs"])
if land_gdf.empty:
    land_mask = np.zeros((ref_meta["height"], ref_meta["width"]), dtype=bool)
    log.warning("No land polygons — sea mask is all-sea (within domain)")
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
sea_mask_arr = np.where(sea_mask, np.float32(1.0), NODATA).astype(np.float32)

if elevation_nodata is not None:
    sea_mask_arr[elevation_arr == np.float32(elevation_nodata)] = NODATA
sea_mask_arr[~np.isfinite(elevation_arr)] = NODATA

sea_mask_meta = ref_meta.copy()
sea_mask_meta.update(nodata=float(NODATA))

n_wb = int(((lu_arr == 200) & land_mask).sum())
n_water = int((sea_mask_arr == 1.0).sum())
log.info(
    f"sea_mask: {n_water:,} sea/inland-water px "
    f"({n_wb:,} inland water-body px added inside the land mask)"
)

with rasterio.open(snakemake.output.sea_mask, "w", **sea_mask_meta) as zdst:
    zdst.write(sea_mask_arr, 1)
log.info(f"Written: {snakemake.output.sea_mask}")

# ── plots ──────────────────────────────────────────────────────────────────────
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
