"""Raster clipping, merging, tile lookup, and roughness reclassification."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from shapely.geometry import box

log = logging.getLogger(__name__)


def resample_to_utm_array(
    src_arr: np.ndarray,
    src_transform,
    src_crs: str,
    dst_meta: dict,
) -> np.ndarray:
    """
    Bilinear-resample *src_arr* onto the UTM grid described by *dst_meta*
    (keys: height, width, transform, crs).  Thin wrapper around rasterio.warp.reproject
    that keeps heavy scripts DRY.  Returns float32, same shape as target grid.
    """
    from rasterio.warp import reproject as _rp, Resampling as _RS

    dest = np.empty((dst_meta["height"], dst_meta["width"]), dtype=np.float32)
    _rp(
        source=src_arr,
        destination=dest,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_meta["transform"],
        dst_crs=dst_meta["crs"],
        resampling=_RS.bilinear,
    )
    return dest


def reproject_nan_aware(
    source: np.ndarray,
    src_transform,
    src_crs,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs,
    resampling=None,
) -> np.ndarray:
    """
    Coverage-weighted reproject of a NaN-as-nodata array.

    Resamples ``value * valid`` and ``valid`` (a 0/1 mask) separately, then
    divides, instead of relying on the warp backend's own nodata handling.
    A destination pixel only comes out NaN when its entire receptive field
    on the source grid is nodata; a pixel near the edge of a NaN region is
    reweighted from its valid neighbours only, instead of plain bilinear/
    cubic resampling propagating NaN in from any single invalid contributing
    source pixel — which erodes/blurs real data right at nodata edges (a
    domain-polygon boundary, a DEM tile gap, the land/sea split, etc).

    Args:
        source:                 float32 array, NaN = nodata.
        src_transform, src_crs: source grid georeferencing.
        dst_shape:              (height, width) of the destination grid.
        dst_transform, dst_crs: destination grid georeferencing.
        resampling:             rasterio.warp.Resampling enum; defaults to bilinear.

    Returns:
        float32 array, shape ``dst_shape``, NaN where the destination pixel
        has no valid source coverage at all.
    """
    from rasterio.warp import reproject as _rp, Resampling as _RS

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


def load_raster_to_utm_array(
    src_path: str | Path,
    wgs84_bounds: tuple[float, float, float, float],
    dst_meta: dict,
) -> np.ndarray:
    """
    Load a raster (single GeoTIFF or directory of tiles) and reproject to the
    UTM working grid described by *dst_meta*.  Handles the tile-directory case by
    merging overlapping tiles first.  Returns float32 with nodata → NaN.
    """
    import rasterio
    from rasterio.merge import merge as _merge
    from rasterio.warp import reproject as _rp, Resampling as _RS
    from shapely.geometry import box as _box

    src_path = Path(src_path)
    dest = np.full((dst_meta["height"], dst_meta["width"]), np.nan, dtype=np.float32)

    if src_path.is_dir():
        bbox_geom = _box(*wgs84_bounds)
        candidates = [
            str(fp)
            for fp in sorted(src_path.glob("*.tif"))
            if _tile_intersects(fp, bbox_geom)
        ]
        if not candidates:
            log.warning(f"No raster tiles found overlapping domain in {src_path}")
            return dest
        open_ds = [rasterio.open(p) for p in candidates]
        try:
            merged_arr, merged_transform = _merge(open_ds, bounds=wgs84_bounds)
            src_nodata = open_ds[0].nodata
            src_crs = open_ds[0].crs
        finally:
            for ds in open_ds:
                ds.close()
        src_arr = merged_arr[0].astype(np.float32)
        if src_nodata is not None:
            src_arr[src_arr == src_nodata] = np.nan
    else:
        with rasterio.open(src_path) as src:
            src_arr = src.read(1).astype(np.float32)
            merged_transform = src.transform
            src_crs = src.crs
            if src.nodata is not None:
                src_arr[src_arr == src.nodata] = np.nan

    _rp(
        source=src_arr,
        destination=dest,
        src_transform=merged_transform,
        src_crs=src_crs,
        dst_transform=dst_meta["transform"],
        dst_crs=dst_meta["crs"],
        resampling=_RS.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dest


def find_fathomdem_tiles(
    topo_dir: str | Path,
    bounds: tuple[float, float, float, float],
) -> list[str]:
    """
    Return paths of existing FathomDEM tiles covering the given WGS84 bounds.

    FathomDEM tiles follow the (lowercase) naming convention
    ``{n/s}{lat:02d}{e/w}{lon:03d}.tif`` (e.g. ``n00e010.tif``,
    ``s01w090.tif``). Matched here with uppercase N/E/S/W, which also
    resolves on case-insensitive filesystems (e.g. Windows/NTFS).
    Tiles that do not exist on disk (e.g. ocean-only cells) are silently
    skipped with a debug log entry.

    Args:
        topo_dir: Directory containing FathomDEM .tif tiles.
        bounds:   (lon_min, lat_min, lon_max, lat_max) in WGS84.

    Returns:
        List of absolute file path strings for existing tiles.
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
    """
    Merge a list of raster tiles, clip to bounds, and write to out_path.

    Args:
        tile_paths: List of .tif file paths to merge.
        bounds:     (lon_min, lat_min, lon_max, lat_max) clipping bounds in WGS84.
        out_path:   Destination GeoTIFF path.

    Raises:
        FileNotFoundError: If tile_paths is empty.
    """
    if not tile_paths:
        raise FileNotFoundError("No raster tiles provided for merging")
    log.info(f"Merging {len(tile_paths)} tile(s)")
    open_ds = [rasterio.open(p) for p in tile_paths]
    dtype = open_ds[0].dtypes[0]
    src_nodata = open_ds[0].nodata
    # Cast to the source dtype before handing it to rio_merge: rasterio compares
    # the (float64) Python nodata value against the destination dtype with
    # np.can_cast(..., casting="safe"), which is always False for float64 -> float32
    # and triggers a spurious "cannot safely be represented" warning even though the
    # value itself round-trips exactly.
    merge_nodata = np.dtype(dtype).type(src_nodata) if src_nodata is not None else None
    merged, transform = rio_merge(
        open_ds, bounds=bounds, dtype=dtype, nodata=merge_nodata
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


def clip_raster(
    src_path: str | Path,
    bounds: tuple[float, float, float, float],
    out_path: str | Path,
) -> None:
    """
    Clip a single raster to WGS84 bounds and write to out_path.

    Handles both single-file rasters and directory sources by checking whether
    src_path points to a directory and delegating to merge_tiled_raster in that
    case (scanning all *.tif files that overlap bounds).

    Args:
        src_path: Path to a raster file or a directory of .tif tiles.
        bounds:   (lon_min, lat_min, lon_max, lat_max) in WGS84.
        out_path: Destination GeoTIFF path.
    """
    src_path = Path(src_path)
    if src_path.is_dir():
        bbox = box(*bounds)
        candidates = [
            str(fp) for fp in src_path.glob("*.tif") if _tile_intersects(fp, bbox)
        ]
        if not candidates:
            raise FileNotFoundError(f"No .tif tiles overlap domain bbox in {src_path}")
        merge_tiled_raster(candidates, bounds, out_path)
        return

    geom = [box(*bounds).__geo_interface__]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        out_data, out_transform = rio_mask(src, geom, crop=True, all_touched=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": out_data.shape[1],
                "width": out_data.shape[2],
                "transform": out_transform,
            }
        )
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(out_data)


def reproject_to_reference_grid(
    src_path: str | Path,
    wgs84_bounds: tuple[float, float, float, float],
    ref_meta: dict,
    resampling=None,
) -> tuple[np.ndarray, dict]:
    """
    Clip a global raster to wgs84_bounds, then reproject it onto the exact
    grid described by ref_meta (height, width, transform, crs).

    Used to put landuse.tif (and, by inheritance, roughness.tif -- reclassified
    pixel-for-pixel from it) on the same UTM working grid as
    elevation_merged.tif/zsini.tif, instead of each staying on its own
    independent native-resolution WGS84 grid. Left unaligned, every downstream
    consumer (hydromt's model build, compute_max_inundation's water-body mask)
    would have to reproject landuse independently, risking a land/sea split
    that disagrees with the one already baked into elevation_merged.tif/zsini.tif
    from OSM land polygons.

    Args:
        src_path:     Path to the global source raster (single GeoTIFF).
        wgs84_bounds: (lon_min, lat_min, lon_max, lat_max) -- clip extent,
                      same convention as clip_raster.
        ref_meta:     Reference grid spec (e.g. an opened elevation_merged.tif's
                      .meta) -- must contain 'height', 'width', 'transform', 'crs'.
        resampling:   rasterio.warp.Resampling enum; defaults to nearest
                      (appropriate for categorical land-use codes -- avoids
                      inventing fractional/blended class values).

    Returns:
        (data, out_meta): reprojected array (same dtype/nodata as the
        source) and a GeoTIFF meta dict ready for rasterio.open(..., "w").
    """
    from rasterio.warp import reproject as _rp, Resampling as _RS

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


def _tile_intersects(fp: Path, bbox) -> bool:
    """Return True if the raster at fp spatially overlaps bbox."""
    with rasterio.open(fp) as src:
        b = src.bounds
        return box(b.left, b.bottom, b.right, b.top).intersects(bbox)


def build_roughness_raster(
    landuse_path: str | Path,
    lookup_path: str | Path,
    out_path: str | Path,
) -> set[int]:
    """
    Reclassify a Copernicus LC100 land-use raster to Manning's n roughness values.

    Each land-use code is mapped to a Manning's n value via a CSV lookup table.
    Pixels without a matching lookup entry are set to the raster nodata value
    (-9999).

    Args:
        landuse_path: Path to the clipped land-use GeoTIFF (Copernicus LC100 codes).
        lookup_path:  CSV with columns 'copernicus_worldcover' (int) and
                      'manning_n' (float).
        out_path:     Destination roughness GeoTIFF path.

    Returns:
        Set of integer land-use codes that were present in the raster but absent
        from the lookup table (useful for logging warnings in the calling script).
    """
    lookup = pd.read_csv(lookup_path)
    lu_to_n = dict(
        zip(
            lookup["copernicus_worldcover"].astype(int),
            lookup["manning_n"].astype(float),
        )
    )

    with rasterio.open(landuse_path) as src:
        lu_data = src.read(1)
        meta = src.meta.copy()
        nodata_lu = int(src.nodata) if src.nodata is not None else 255

    meta.update({"dtype": "float32", "nodata": -9999.0})
    roughness = np.full(lu_data.shape, -9999.0, dtype=np.float32)
    for code, n_val in lu_to_n.items():
        roughness[lu_data == code] = n_val

    unmapped = set(np.unique(lu_data).tolist()) - set(lu_to_n.keys()) - {nodata_lu}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(roughness, 1)

    return unmapped


def compute_geoid_offset_arr(
    goco_path,
    egm_path,
):
    """
    Compute the per-pixel geoid height offset N_EGM2008 − N_GOCO06s from .gfc files.

    EGM2008 is truncated to GOCO06s's maximum degree (≈ 300) before synthesis so
    both grids share the same spectral bandwidth.  Adding this offset to a DEM that
    carries EGM2008 heights converts it to GOCO06s-referenced heights, aligning it
    with the MDT_CNES-CLS22 product before MDT subtraction.

    Requires: conda install -c conda-forge pyshtools boule

    Args:
        goco_path: Path to GOCO06s.gfc (ICGEM format).
        egm_path:  Path to EGM2008.gfc  (ICGEM format).

    Returns:
        offset_arr: float32 ndarray (nlat × nlon), global, north-up, lon in −180…180.
        transform:  rasterio Affine for the array.
        crs:        "EPSG:4326"
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
