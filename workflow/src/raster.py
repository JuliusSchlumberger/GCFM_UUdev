"""Raster clipping, merging, tile lookup, grid alignment and coastal blending.

Land-use preparation, the sea class and Manning's n live in src.landuse.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from shapely.geometry import box

log = logging.getLogger(__name__)


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

    Tiles are named ``{n/s}{lat:02d}{e/w}{lon:03d}.tif`` (e.g. ``n00e010.tif``)
    and matched here with uppercase N/E/S/W, which also resolves on
    case-insensitive filesystems (e.g. Windows/NTFS). Absent tiles (ocean-only
    cells) are skipped.
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    tiles: list[str] = []
    for lat in range(math.floor(lat_min), math.ceil(lat_max)):
        for lon in range(math.floor(lon_min), math.ceil(lon_max)):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            fname = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}.tif"
            fpath = Path(topo_dir) / fname
            if fpath.exists():
                tiles.append(str(fpath))
            else:
                log.debug(f"Tile absent (likely ocean): {fname}")
    return tiles


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
    open_ds = [rasterio.open(p) for p in tile_paths]
    dtype = open_ds[0].dtypes[0]
    src_nodata = open_ds[0].nodata
    # rasterio checks nodata with np.can_cast(float64 -> dtype, "safe"), which
    # warns spuriously for float32 even when the value round-trips exactly.
    merge_nodata = np.dtype(dtype).type(src_nodata) if src_nodata is not None else None
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
    meta.update(
        {
            "height": merged.shape[1],
            "width": merged.shape[2],
            "transform": transform,
        }
    )
    for ds in open_ds:
        ds.close()
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
    get_land_polygons (03) and get_protection_levels (04). The window is
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

    land_bool = lu_arr != 200
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

    with rasterio.open(landuse_on_grid_path) as src:
        lu_arr = src.read(1)
        transform, crs, nodata = src.transform, src.crs, src.nodata
    land = lu_arr != 200
    if nodata is not None:
        land &= lu_arr != nodata
    land_u8 = land.astype(np.uint8)
    geoms = [
        shapely_shape(geom)
        for geom, val in rio_shapes(land_u8, mask=land_u8, transform=transform)
        if val == 1
    ]
    return gpd.GeoDataFrame(geometry=geoms, crs=crs).to_crs("EPSG:4326")


def restrict_waterlevel_boundary_to_sea(
    mask: np.ndarray, landuse_on_grid: np.ndarray
) -> tuple[np.ndarray, int]:
    """Turn water-level boundary cells (mask == 2) not on grid sea (200) into active cells (1).

    Applied after hydromt create_boundary(btype="waterlevel") by rules
    modelled_depth_estimation (10) and build_sfincs_skeleton (13) instead of
    exclude_polygon=<land polygons>: those are traced from the native land-use
    raster and can disagree with the grid's land/sea split at the domain edge.
    Returns (new_mask, n_cells_moved_off_the_boundary).
    """
    if mask.shape != landuse_on_grid.shape:
        raise ValueError(
            f"mask {mask.shape} and landuse_on_grid {landuse_on_grid.shape} are not the same grid"
        )
    on_land = (mask == 2) & (landuse_on_grid != 200)
    out = mask.copy()
    out[on_land] = 1
    return out, int(on_land.sum())


def reproject_to_reference_grid(
    src_path: str | Path,
    wgs84_bounds: tuple[float, float, float, float],
    ref_meta: dict,
    resampling=None,
) -> tuple[np.ndarray, dict]:
    """Clip a global raster to wgs84_bounds and reproject it onto ref_meta's grid.

    ref_meta needs 'height', 'width', 'transform', 'crs'. resampling defaults
    to nearest, so categorical land-use codes are never blended. Aligning
    landuse (and roughness) with elevation_merged.tif/sea_mask.tif keeps one
    land/sea split (landuse == 200) for all consumers. Returns (data, out_meta)
    with the source dtype/nodata and a GeoTIFF meta dict.
    """
    from rasterio.warp import Resampling as _RS
    from rasterio.warp import reproject as _rp

    resampling = resampling if resampling is not None else _RS.nearest

    with rasterio.open(src_path) as src:
        src_nodata = src.nodata
        src_dtype = src.dtypes[0]
        src_crs = src.crs
        geom = [box(*wgs84_bounds).__geo_interface__]
        clipped, clipped_transform = rio_mask(src, geom, crop=True, all_touched=True)

    fill = src_nodata if src_nodata is not None else 0
    dst = np.full((ref_meta["height"], ref_meta["width"]), fill, dtype=src_dtype)
    _rp(
        source=clipped[0],
        destination=dst,
        src_transform=clipped_transform,
        src_crs=src_crs,
        dst_transform=ref_meta["transform"],
        dst_crs=ref_meta["crs"],
        src_nodata=src_nodata,
        dst_nodata=src_nodata,
        resampling=resampling,
    )

    out_meta = {
        "driver": "GTiff",
        "dtype": src_dtype,
        "count": 1,
        "height": ref_meta["height"],
        "width": ref_meta["width"],
        "crs": ref_meta["crs"],
        "transform": ref_meta["transform"],
        "nodata": src_nodata,
    }
    return dst, out_meta


def clip_ocean_from_topo(
    topo_utm: np.ndarray,
    mask_path: str | Path,
    wgs84_bounds: tuple[float, float, float, float],
    ref_meta: dict,
    ocean_value: int = 1,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Set FathomDEM pixels to NaN where the DeltaDTM mask marks ocean.

    FathomDEM is terrestrial and reports spurious near-zero elevation over open
    water well past the coastline; NaN there lets the FathomDEM/GEBCO merge fall
    back to GEBCO bathymetry. Only ocean_value (default 1) is ocean: classes
    0, 2, 3 are land, and mask nodata (no DeltaDTM coverage) is not reliably
    ocean (see ``deltadtm_mask`` in data_catalogue.yml). topo_utm is on the UTM
    working grid of ref_meta and is not modified in place. Returns
    (topo_clipped, n_clipped, ocean) with n_clipped counting formerly valid pixels.
    """
    mask_utm, _ = reproject_to_reference_grid(mask_path, wgs84_bounds, ref_meta)
    ocean = mask_utm == ocean_value
    topo_clipped = topo_utm.copy()
    n_clipped = int((ocean & ~np.isnan(topo_clipped)).sum())
    topo_clipped[ocean] = np.nan
    return topo_clipped, n_clipped, ocean


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

    lat_dim = next(d for d in da_goco.dims if "lat" in d.lower())
    lon_dim = next(d for d in da_goco.dims if "lon" in d.lower())
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
