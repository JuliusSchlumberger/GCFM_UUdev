"""Raster clipping, merging, tile lookup, grid alignment and coastal blending.

Land-use preparation, the sea class and Manning's n live in src.landuse.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.transform import Affine
from scipy import ndimage
from shapely.geometry import box

log = logging.getLogger(__name__)

FLOAT_NODATA = -9999.0  # nodata of every float32 raster the pipeline writes
# Lossless creation options for float32 GeoTIFFs (rasterio/GDAL keywords).
FLOAT_GTIFF_COMPRESSION = {
    "compress": "deflate",
    "predictor": 3,
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
}


def write_float_raster(path: str | Path, arr: np.ndarray, transform, crs) -> None:
    """Write a 2-D array as a float32 GeoTIFF, NaN as FLOAT_NODATA."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tiled = min(arr.shape) >= FLOAT_GTIFF_COMPRESSION["blockxsize"]
    options = (
        FLOAT_GTIFF_COMPRESSION if tiled else {"compress": "deflate", "predictor": 3}
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=FLOAT_NODATA,
        **options,
    ) as dst:
        dst.write(np.where(np.isfinite(arr), arr, FLOAT_NODATA).astype(np.float32), 1)


def read_float_raster(path: str | Path) -> tuple[np.ndarray, object, object]:
    """Return (array with NaN for nodata, transform, crs) of a single-band raster."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == np.float32(src.nodata)] = np.nan
        return arr, src.transform, src.crs


def reproject_nan_aware(
    source: np.ndarray,
    src_transform,
    src_crs,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs,
    resampling=None,
) -> np.ndarray:
    """Reproject a NaN-as-nodata float32 array with coverage weighting.

    Resamples ``value * valid`` and ``valid`` separately and divides, so a
    pixel near a NaN region is reweighted from its valid neighbours instead of
    picking up NaN from any one invalid source pixel (which erodes data at
    domain edges, tile gaps and the land/sea split). Output is NaN only where
    the destination pixel has no valid source coverage (weight <= 1e-6).
    resampling defaults to bilinear.
    """
    from rasterio.warp import Resampling as _RS
    from rasterio.warp import reproject as _rp

    resampling = resampling if resampling is not None else _RS.bilinear

    valid = (~np.isnan(source)).astype(np.float32)
    filled = np.where(valid > 0, source, np.float32(0.0)).astype(np.float32)

    dst_value = np.zeros(dst_shape, dtype=np.float32)
    dst_weight = np.zeros(dst_shape, dtype=np.float32)
    _rp(
        source=filled,
        destination=dst_value,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
    )
    _rp(
        source=valid,
        destination=dst_weight,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        dst = dst_value / dst_weight
    dst[dst_weight <= 1e-6] = np.nan
    return dst.astype(np.float32)


def find_fathomdem_tiles(
    topo_dir: str | Path,
    bounds: tuple[float, float, float, float],
) -> list[str]:
    """Return paths of existing FathomDEM tiles covering (lon_min, lat_min, lon_max, lat_max).

    Tiles are named in lowercase ``{n/s}{lat:02d}{e/w}{lon:03d}.tif`` (e.g.
    ``n00e010.tif``), matched exactly so case-sensitive filesystems resolve
    them. Absent tiles (ocean-only cells) are skipped.
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    tiles: list[str] = []
    for lat in range(math.floor(lat_min), math.ceil(lat_max)):
        for lon in range(math.floor(lon_min), math.ceil(lon_max)):
            ns = "n" if lat >= 0 else "s"
            ew = "e" if lon >= 0 else "w"
            fname = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}.tif"
            fpath = Path(topo_dir) / fname
            if fpath.exists():
                tiles.append(str(fpath))
            else:
                log.debug(f"Tile absent (likely ocean): {fname}")
    return tiles


def merge_tiles(
    tile_paths: list[str],
    bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, Affine, dict]:
    """Merge raster tiles clipped to WGS84 bounds on the tiles' own pixel lattice.

    Returns the (band, row, col) array, its transform and the first tile's
    profile updated to the merged shape. Raises FileNotFoundError if
    tile_paths is empty.
    """
    if not tile_paths:
        raise FileNotFoundError("No raster tiles provided for merging")
    open_ds = [rasterio.open(p) for p in tile_paths]
    try:
        dtype = open_ds[0].dtypes[0]
        src_nodata = open_ds[0].nodata
        # rasterio checks nodata with np.can_cast(float64 -> dtype, "safe"), which
        # warns spuriously for float32 even when the value round-trips exactly.
        merge_nodata = (
            np.dtype(dtype).type(src_nodata) if src_nodata is not None else None
        )
        # Keep the tiles' own pixel lattice; with bare bounds rasterio resamples
        # onto a grid starting at the bounds' corner, shifting data by up to half
        # a pixel.
        merged, transform = rio_merge(
            open_ds,
            bounds=bounds,
            dtype=dtype,
            nodata=merge_nodata,
            target_aligned_pixels=True,
        )
        meta = open_ds[0].meta.copy()
    finally:
        for ds in open_ds:
            ds.close()
    meta.update(
        {
            "height": merged.shape[1],
            "width": merged.shape[2],
            "transform": transform,
        }
    )
    return merged, transform, meta


def latlon_dims(obj) -> tuple[str, str]:
    """Return the (lat, lon) dimension names: the first dims containing "lat"/"lon".

    Case-insensitive; raises StopIteration when either is missing.
    """
    lat_dim = next(d for d in obj.dims if "lat" in d.lower())
    lon_dim = next(d for d in obj.dims if "lon" in d.lower())
    return lat_dim, lon_dim


def read_on_grid(
    path: str | Path,
    shape: tuple[int, int],
    transform: Affine | None = None,
) -> np.ndarray:
    """Read band 1 of a raster that must be pixel-identical to the model grid.

    Raises ValueError when the shape differs, or the transform when given.
    """
    with rasterio.open(path) as src:
        arr, arr_transform = src.read(1), src.transform
    if tuple(arr.shape) != tuple(shape) or (
        transform is not None
        and not arr_transform.almost_equals(transform, precision=1e-6)
    ):
        raise ValueError(
            f"{Path(path).name} shape {arr.shape} does not match the model grid "
            f"{tuple(shape)} -- expected pixel-identical grids."
        )
    return arr


def merge_tiled_raster(
    tile_paths: list[str],
    bounds: tuple[float, float, float, float],
    out_path: str | Path,
) -> None:
    """Merge raster tiles, clip to WGS84 bounds and write a GeoTIFF to out_path.

    Raises FileNotFoundError if tile_paths is empty.
    """
    if not tile_paths:
        raise FileNotFoundError("No raster tiles provided for merging")
    log.info(f"Merging {len(tile_paths)} tile(s)")
    merged, _, meta = merge_tiles(tile_paths, bounds)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(merged)


def vectorize_land_from_landuse(
    landuse_path: str | Path,
    wgs84_bounds: tuple[float, float, float, float],
    max_pixels: int | None = 4_000_000,
):
    """Vectorize landuse != 200 (not sea) within wgs84_bounds into land polygons.

    No reprojection: the result keeps the source CRS (EPSG:4326). Used by rules
    prepare_landuse (05) and get_protection_levels (03). The window is
    decimated by mode (no class invented) to at most max_pixels before tracing;
    the polygons are a figure background only, and a 10 m source (ESA
    WorldCover) gives ~26M pixels per delta. None disables decimation.
    """
    import geopandas as gpd
    from affine import Affine
    from rasterio.enums import Resampling as _RS
    from rasterio.features import shapes as rio_shapes
    from rasterio.windows import from_bounds
    from shapely.geometry import shape as shapely_shape
    from src.landuse import LANDUSE_SEA_CODE

    with rasterio.open(landuse_path) as src:
        window = from_bounds(*wgs84_bounds, transform=src.transform)
        win_h, win_w = int(round(window.height)), int(round(window.width))
        factor = 1
        if max_pixels is not None and win_h * win_w > max_pixels:
            factor = int(math.ceil(math.sqrt(win_h * win_w / max_pixels)))
        out_shape = (max(win_h // factor, 1), max(win_w // factor, 1))
        lu_arr = src.read(1, window=window, out_shape=out_shape, resampling=_RS.mode)
        win_transform = src.window_transform(window) * Affine.scale(
            win_w / out_shape[1], win_h / out_shape[0]
        )
        nodata = src.nodata
        src_crs = src.crs
    if factor > 1:
        log.info(
            f"vectorize_land_from_landuse: {win_w}x{win_h} px window decimated by "
            f"{factor} (mode) to {out_shape[1]}x{out_shape[0]} px before tracing"
        )

    land_bool = lu_arr != LANDUSE_SEA_CODE
    if nodata is not None:
        land_bool &= lu_arr != nodata
    land_bool_u8 = land_bool.astype(np.uint8)

    land_geoms = [
        shapely_shape(geom)
        for geom, val in rio_shapes(
            land_bool_u8, mask=land_bool_u8, transform=win_transform
        )
        if val == 1
    ]
    return gpd.GeoDataFrame(geometry=land_geoms, crs=src_crs)


def vectorize_land_mask_on_grid(landuse_on_grid_path: str | Path):
    """Return land polygons (land use != 200, != nodata) traced on the SFINCS grid, in EPSG:4326.

    Traced from landuse_on_grid.tif (rule grid_align_landuse), so edges follow
    model cell edges and the figure background matches the model's own
    land/sea split. Works for either row order of the grid transform.
    """
    import geopandas as gpd
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape as shapely_shape
    from src.landuse import LANDUSE_SEA_CODE

    with rasterio.open(landuse_on_grid_path) as src:
        lu_arr = src.read(1)
        transform, crs, nodata = src.transform, src.crs, src.nodata
    land = lu_arr != LANDUSE_SEA_CODE
    if nodata is not None:
        land &= lu_arr != nodata
    land_u8 = land.astype(np.uint8)
    geoms = [
        shapely_shape(geom)
        for geom, val in rio_shapes(land_u8, mask=land_u8, transform=transform)
        if val == 1
    ]
    return gpd.GeoDataFrame(geometry=geoms, crs=crs).to_crs("EPSG:4326")


def restrict_water_level_boundary_to_sea(
    mask: np.ndarray, landuse_on_grid: np.ndarray
) -> tuple[np.ndarray, int]:
    """Turn water-level boundary cells (mask == 2) not on grid sea (200) into active cells (1).

    Applied after hydromt create_boundary(btype="waterlevel") by rules
    modelled_depth_estimation (10) and build_sfincs_skeleton (13) instead of
    exclude_polygon=<land polygons>: those are traced from the native land-use
    raster and can disagree with the grid's land/sea split at the domain edge.
    Returns (new_mask, n_cells_moved_off_the_boundary).
    """
    # Function-level: src.landuse imports src.raster.
    from src.landuse import LANDUSE_SEA_CODE

    if mask.shape != landuse_on_grid.shape:
        raise ValueError(
            f"mask {mask.shape} and landuse_on_grid {landuse_on_grid.shape} are not the same grid"
        )
    on_land = (mask == 2) & (landuse_on_grid != LANDUSE_SEA_CODE)
    out = mask.copy()
    out[on_land] = 1
    return out, int(on_land.sum())


def dry_isolated_sea_cells(
    zsini: np.ndarray, dep: np.ndarray, dep_nodata: float, calm_sea_level_m: float
) -> tuple[np.ndarray, int]:
    """Start sea cells outside the open ocean dry at bed level; return (zsini, n_dried).

    Sea cells are those at ``calm_sea_level_m`` in ``zsini`` (NaN elsewhere). The open
    ocean is the largest 4-connected sea component: domains are polygon clips in
    a NaN-padded raster, so an edge-touching rule fails. Cells of other components
    with dep < calm_sea_level_m would hold trapped water and are set to dep instead.
    """
    dep = dep.astype(np.float32)
    dep_valid = np.where(np.isclose(dep, np.float32(dep_nodata)), np.float32(1e6), dep)
    sea = np.isclose(zsini, np.float32(calm_sea_level_m))
    labeled, n_components = ndimage.label(sea)
    if n_components > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        connected = labeled == int(np.argmax(sizes))
    else:
        connected = np.zeros_like(sea)
    isolated_wet = sea & ~connected & (dep_valid < np.float32(calm_sea_level_m))
    n_dried = int(isolated_wet.sum())
    if n_dried > 0:
        zsini = np.where(isolated_wet, dep_valid, zsini)
    return zsini, n_dried


def open_water_mask(
    water: np.ndarray, pixel_size_m: float, min_half_width_m: float
) -> np.ndarray:
    """Return the part of water at least 2 * min_half_width_m wide (sea, bays, estuaries).

    A morphological opening with a disc of radius min_half_width_m, done with
    two Euclidean distance transforms. The raster edge does not count as a
    bank, so sea running off the edge stays open water.
    """
    from scipy.ndimage import distance_transform_edt

    if min_half_width_m <= 0 or not water.any():
        return water.copy()
    padded = np.pad(water, 1, mode="edge")
    core = distance_transform_edt(padded, sampling=pixel_size_m) >= min_half_width_m
    if not core.any():
        return np.zeros_like(water)
    near_core = distance_transform_edt(~core, sampling=pixel_size_m) <= min_half_width_m
    return (padded & near_core)[1:-1, 1:-1]


def blend_coast(
    merged: np.ndarray,
    land: np.ndarray,
    sea_bed: np.ndarray,
    pixel_size_m: float,
    blend_m: float,
    where: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Smooth the land/sea seam of a hard-merged elevation raster.

    Non-land pixels within blend_m (m) of land are interpolated linearly with
    distance from the nearest land pixel's elevation at the coast to sea_bed
    (GEBCO, NaN = no data) at blend_m. Land pixels (True where the DEM is
    valid) are never changed. Distances are on the pixel lattice, so the result
    is orientation-independent. where limits blending (e.g. to open water);
    blend_m <= 0 returns merged unchanged. Returns (blended, n_blended).
    """
    from scipy.ndimage import distance_transform_edt

    if blend_m <= 0 or not land.any() or land.all():
        return merged, 0
    dist, (near_row, near_col) = distance_transform_edt(
        ~land, sampling=pixel_size_m, return_indices=True
    )
    band = ~land & (dist < blend_m) & np.isfinite(sea_bed)
    if where is not None:
        band &= where
    weight_land = 1.0 - dist[band] / blend_m
    land_value = merged[near_row[band], near_col[band]]
    blended = merged.copy()
    blended[band] = weight_land * land_value + (1.0 - weight_land) * sea_bed[band]
    return blended, int(band.sum())


def _tile_intersects(fp: Path, bbox) -> bool:
    with rasterio.open(fp) as src:
        b = src.bounds
        return box(b.left, b.bottom, b.right, b.top).intersects(bbox)


def compute_geoid_offset_arr(
    goco_path,
    egm_path,
):
    """Return the global geoid offset N_EGM2008 − N_GOCO06s (m) from ICGEM .gfc files.

    EGM2008 is truncated to GOCO06s's maximum degree (≈ 300) before synthesis so
    both grids share the same spectral bandwidth.  Adding this offset to a DEM that
    carries EGM2008 heights converts it to GOCO06s-referenced heights, aligning it
    with the MDT_CNES-CLS22 product (whose MDT is the mean sea surface height
    above the GOCO06s geoid).

    Returns (offset_arr, transform, "EPSG:4326"); offset_arr is float32, global,
    north-up, lon in −180…180. Requires pyshtools and boule (conda-forge).
    """
    try:
        import pyshtools as pysh
    except ImportError:
        raise ImportError(
            "pyshtools is required for geoid height computation.\n"
            "Install with: conda install -c conda-forge pyshtools"
        )
    try:
        import boule as _boule
    except ImportError:
        raise ImportError(
            "boule is required by pyshtools for ellipsoid definitions.\n"
            "Install with: conda install -c conda-forge boule"
        )
    from rasterio.transform import from_origin as _fro

    goco = pysh.SHGravCoeffs.from_file(str(goco_path), format="icgem")
    egm = pysh.SHGravCoeffs.from_file(str(egm_path), format="icgem")
    lmax = goco.lmax
    egm_trunc = egm.pad(lmax)

    wgs84 = _boule.WGS84
    grid_goco = goco.geoid(ellipsoid=wgs84, lmax=lmax)
    grid_egm = egm_trunc.geoid(ellipsoid=wgs84, lmax=lmax)

    da_goco = grid_goco.to_xarray()
    da_egm = grid_egm.to_xarray()
    offset = (da_egm.values - da_goco.values).astype(np.float32)

    lat_dim, lon_dim = latlon_dims(da_goco)
    lats = da_goco[lat_dim].values
    lons = da_goco[lon_dim].values
    half = len(lons) // 2
    offset = np.roll(offset, -half, axis=1)
    lons = np.concatenate([lons[half:] - 360.0, lons[:half]])

    dlat = float(np.abs(lats[0] - lats[1]))
    dlon = float(lons[1] - lons[0])
    transform = _fro(
        west=float(lons[0]) - dlon / 2,
        north=float(lats[0]) + dlat / 2,
        xsize=dlon,
        ysize=dlat,
    )
    return offset, transform, "EPSG:4326"
