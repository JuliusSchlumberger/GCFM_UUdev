"""Raster clipping, merging, tile lookup, and grid alignment.

Land-use specifics (per-basin source preparation, the sea class, and
Manning's n aggregation) live in src.landuse instead.
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


def vectorize_land_from_landuse(
    landuse_path: str | Path,
    wgs84_bounds: tuple[float, float, float, float],
    max_pixels: int | None = 4_000_000,
):
    """
    Windowed-read a landuse raster clipped to wgs84_bounds and vectorize
    landuse != 200 (i.e. not sea) into a "land" GeoDataFrame, native WGS84
    (no reprojection -- both sources are already EPSG:4326).

    Used by rule get_land_polygons (03, the per-basin land_polygons.gpkg
    product, from this basin's own landuse_source.tif) and by rule
    get_protection_levels (04, which deliberately depends only on the delta
    polygon -- see that rule's own docstring -- so it vectorizes its own
    wider window straight from the raw LC100 catalogue source instead).

    Args:
        landuse_path: Path to a landuse GeoTIFF in pipeline codes.
        wgs84_bounds: (lon_min, lat_min, lon_max, lat_max) -- clip extent.
        max_pixels:   Decimate (by MODE, so no class is invented) to at most
                      this many pixels before tracing. These polygons are a
                      figure background only, and a 10 m source (ESA
                      WorldCover) puts 26M pixels in a delta-sized window --
                      tracing that raw produces enormous speckle for no
                      visible gain. None disables decimation.

    Returns:
        geopandas.GeoDataFrame of land polygons, CRS EPSG:4326 (matching the
        source raster's own CRS).
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
    """
    Land mask (land use != 200 and != nodata) traced from the land use
    resampled onto the SFINCS grid (landuse_on_grid.tif, rule
    grid_align_landuse) -- polygon edges run exactly along model cell edges,
    in the grid's own CRS, then reprojected to WGS84 for the plots. The land
    background of every figure of model output, so it can't disagree with
    the model's own land/sea classification (unlike a mask from the native
    land-use raster, vectorize_land_from_landuse). Works for either row
    order of the grid transform.

    Returns:
        geopandas.GeoDataFrame of land polygons, CRS EPSG:4326.
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
    """
    Water-level boundary cells (SFINCS mask == 2) only on open sea: every
    boundary cell whose land use ON THE MODEL GRID (landuse_on_grid.tif,
    rule grid_align_landuse) isn't 200 goes back to a normal active cell
    (1). Used right after hydromt's create_boundary(btype="waterlevel") by
    rules modelled_depth_estimation (10) and build_sfincs_skeleton (13), in
    place of its exclude_polygon=<land polygons> -- those polygons were
    traced from the NATIVE land-use raster, a different geometry from the
    model grid's own land/sea classification, so the two could disagree at
    the domain edge.

    Returns:
        (new_mask, n_cells_moved_off_the_boundary)
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
    """
    Clip a global raster to wgs84_bounds, then reproject it onto the exact
    grid described by ref_meta (height, width, transform, crs).

    Used to put landuse.tif (and, by inheritance, roughness.tif -- reclassified
    pixel-for-pixel from it) on the same UTM working grid as
    elevation_merged.tif/sea_mask.tif, instead of each staying on its own
    independent native-resolution WGS84 grid. Left unaligned, every downstream
    consumer (hydromt's model build, compute_max_inundation's water-body mask)
    would have to reproject landuse independently, risking a land/sea split
    that disagrees with the one sea_mask.tif (landuse == 200) is built from.

    Args:
        src_path:     Path to the global source raster (single GeoTIFF).
        wgs84_bounds: (lon_min, lat_min, lon_max, lat_max) -- clip extent.
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


def clip_ocean_from_topo(
    topo_utm: np.ndarray,
    mask_path: str | Path,
    wgs84_bounds: tuple[float, float, float, float],
    ref_meta: dict,
    ocean_value: int = 1,
) -> tuple[np.ndarray, int]:
    """
    Set FathomDEM pixels to NaN wherever the DeltaDTM validity mask marks
    them as ocean.

    FathomDEM is a terrestrial DEM, not bathymetry, and reports spurious
    near-zero/shallow "elevation" over open water instead of nodata --
    extending seaward well past the coastline. NaN-ing those pixels here
    lets the later FathomDEM/GEBCO hard merge fall back to GEBCO's real
    bathymetry there, with no change needed to the merge step itself.

    Only ``ocean_value`` (default 1) is treated as ocean, NOT the mask's
    nodata value or its other land-related classes (0, 2, 3): those are all
    land in some form, and nodata (no DeltaDTM tile coverage at all) is NOT
    reliably ocean, since DeltaDTM's own tile footprint doesn't necessarily
    reach as far inland as this pipeline's domains do -- see the
    ``deltadtm_mask`` data_catalogue.yml entry for the full class breakdown.

    Args:
        topo_utm: FathomDEM array already reprojected to the UTM working
            grid (NaN = no FathomDEM data). Not modified in place.
        mask_path: Path to the DeltaDTM mask VRT/GeoTIFF.
        wgs84_bounds: (lon_min, lat_min, lon_max, lat_max) of the domain.
        ref_meta: UTM working grid spec (height, width, transform, crs) --
            same dict used to write elevation_merged.tif.
        ocean_value: Mask value that means "ocean" (default 1).

    Returns:
        (topo_clipped, n_clipped): the modified array and the number of
        previously-valid FathomDEM pixels that were set to NaN (for
        logging).
    """
    mask_utm, _ = reproject_to_reference_grid(mask_path, wgs84_bounds, ref_meta)
    ocean = mask_utm == ocean_value
    topo_clipped = topo_utm.copy()
    n_clipped = int((ocean & ~np.isnan(topo_clipped)).sum())
    topo_clipped[ocean] = np.nan
    return topo_clipped, n_clipped


def _tile_intersects(fp: Path, bbox) -> bool:
    """Return True if the raster at fp spatially overlaps bbox."""
    with rasterio.open(fp) as src:
        b = src.bounds
        return box(b.left, b.bottom, b.right, b.top).intersects(bbox)


def compute_geoid_offset_arr(
    goco_path,
    egm_path,
):
    """
    Compute the per-pixel geoid height offset N_EGM2008 − N_GOCO06s from .gfc files.

    EGM2008 is truncated to GOCO06s's maximum degree (≈ 300) before synthesis so
    both grids share the same spectral bandwidth.  Adding this offset to a DEM that
    carries EGM2008 heights converts it to GOCO06s-referenced heights, aligning it
    with the MDT_CNES-CLS22 product (whose MDT is the mean sea surface height
    above the GOCO06s geoid).

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
