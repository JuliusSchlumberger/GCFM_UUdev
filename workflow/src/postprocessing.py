"""Derive analysis quantities from SFINCS run map outputs (sfincs_map.nc).

Each ``compute_*`` function returns data for the matching ``plot_*`` /
``animate_*`` function in ``src.plots``; workflow scripts call them directly.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 -- registers the .rio accessor used below
import xarray as xr
from affine import Affine
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel
from hydromt_sfincs import utils as sfincs_utils

# Open sea (not inland water/rivers, which stay visible on purpose) is excluded
# from flood-extent stats and plots via sea_mask_path, which points at
# zsini_sea_cells_on_grid.tif (rule modelled_depth_estimation; sea = landuse 200,
# weir-protected fringe counts as land). It is coarse and grid-aligned (the file
# zsini is built from): reprojecting onto the subgrid/cell output grid is only
# well-defined from a grid-aligned source (subgrid is an exact integer
# subdivision of the coarse grid, sharing its origin/axes).

# Memory budget for area/volume statistics. Stats are computed one frame at a
# time, so this bounds a SINGLE frame and can exceed _coarsen_for_memory's
# animation default (5e8, shared across all frames).
STATS_MAX_BYTES: float = 1.5e9


def load_sfincs_output(run_dir: str | Path) -> SfincsModel:
    """Load a SFINCS run's map output via SfincsModel (HydroMT-aware spatial dims).

    Reads ONLY sfincs_map.nc: for a run with zero observation points SFINCS still
    writes a zero-station sfincs_his.nc, and hydromt_sfincs's his-file reader
    crashes indexing station 0 of an empty dimension.
    """
    mod = SfincsModel(root=str(run_dir), mode="r")
    # Pre-initialize with skip_read=True so read_map_file's own internal
    # `self.set(...)` call doesn't re-trigger the full (map+his) `read()` --
    # `set()` calls `_initialize()` too, which only acts while `_data is None`.
    mod.output._initialize(skip_read=True)
    fn_map = Path(run_dir) / "sfincs_map.nc"
    if fn_map.is_file():
        mod.output.read_map_file(fn_map=str(fn_map))
    return mod


# In-memory cost of one pixel of a ROTATED raster relative to an axis-aligned
# one: on top of the float32 value, xarray/hydromt carry 2-D float64 x and y
# coordinates for it (an axis-aligned raster only needs two 1-D vectors), and
# every reprojection onto it builds them again. Measured on basin 1416812:
# compute_max_inundation peaks at ~100 bytes per subgrid pixel on a rotated
# grid (8.2 GB for 79 M pixels) -- 5 keeps that peak around 4 GB.
_ROTATED_MEMORY_FACTOR = 5.0


def get_bed_level(
    data_catalog: DataCatalog,
    sfincs_root: str | Path,
    max_bytes: float | None = None,
) -> xr.DataArray:
    """Return the subgrid bed level (``subgrid/dep_subgrid.tif``) for inundation depths.

    ``max_bytes`` bounds memory: a larger raster is read block-averaged by the
    smallest integer factor that fits (GDAL decimated read; the full-resolution
    array is never held). On a rotated grid, opening the full raster allocates
    16 bytes of 2-D coordinates per pixel (5 GB for the 317 M subgrid pixels of
    basin 1416812). ``None`` returns the full-resolution raster.
    Raises FileNotFoundError when missing; the run's ``zb`` (the LOWEST subgrid
    pixel of each cell) is not a substitute.
    """
    import os
    import tempfile

    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window

    dep_subgrid_path = Path(sfincs_root) / "subgrid" / "dep_subgrid.tif"
    if not dep_subgrid_path.exists():
        raise FileNotFoundError(
            f"{dep_subgrid_path} not found -- the model's subgrid reference raster "
            f"(written by sf.subgrid.create(write_dep_tif=True)) is required"
        )
    if max_bytes is None:
        return data_catalog.get_rasterdataset(str(dep_subgrid_path))

    with rasterio.open(dep_subgrid_path) as src:
        rotated = src.transform.b != 0 or src.transform.d != 0
        cost = src.width * src.height * np.dtype(src.dtypes[0]).itemsize
        cost *= _ROTATED_MEMORY_FACTOR if rotated else 1.0
        factor = int(np.ceil((cost / max_bytes) ** 0.5)) if cost > max_bytes else 1
        if factor == 1:
            return data_catalog.get_rasterdataset(str(dep_subgrid_path))
        # Whole factor x factor blocks only (the remainder at the far edges is
        # trimmed), so the origin stays put and each coarse pixel is the plain
        # mean of its block's valid pixels.
        out_h, out_w = src.height // factor, src.width // factor
        coarse = src.read(
            1,
            window=Window(0, 0, out_w * factor, out_h * factor),
            out_shape=(out_h, out_w),
            resampling=Resampling.average,
        )
        profile = src.profile.copy()
        profile.update(
            height=out_h, width=out_w, transform=src.transform * Affine.scale(factor)
        )
    # Round-trip through a small GeoTIFF so the result is the same kind of
    # object (coordinates, CRS, nodata handling) get_rasterdataset returns
    # for the full raster.
    fd, tmp_path = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    try:
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(coarse, 1)
        del coarse
        da = data_catalog.get_rasterdataset(tmp_path).load()
        da.close()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return da


def _coarsen_for_memory(da_ref: xr.DataArray, max_bytes: float = 5e8) -> xr.DataArray:
    """Coarsen the downscale_floodmap target grid to fit ``max_bytes``; no-op if it fits."""
    total_bytes = da_ref.size * da_ref.dtype.itemsize
    if total_bytes <= max_bytes:
        return da_ref
    factor = int(np.ceil((total_bytes / max_bytes) ** 0.5))
    # da_ref is eager; a non-chunked coarsen().mean() allocates array-sized
    # temporaries (skipna's isnan mask) on top of it. Chunking first keeps
    # the temporaries per-chunk.
    da_chunked = da_ref.chunk({"y": 2000, "x": 2000})
    da_coarse = da_chunked.coarsen(x=factor, y=factor, boundary="trim").mean().compute()
    # coarsen().mean() does not reliably keep the "spatial_ref" CRS coordinate.
    # inplace=True avoids write_crs's full-array deep copy.
    da_coarse.rio.write_crs(da_ref.rio.crs, inplace=True)
    # Same for the transform: on a ROTATED raster rioxarray keeps the stored
    # (fine-pixel) GeoTransform instead of deriving one from the coordinates,
    # so the coarsened array would otherwise be georeferenced at the wrong
    # scale. boundary="trim" keeps the origin, so the pixel just grows.
    da_coarse.rio.write_transform(
        da_ref.raster.transform * Affine.scale(factor), inplace=True
    )
    return da_coarse


def land_bed_level(
    sfincs_root: str | Path,
    sea_mask_path: str | Path,
    max_bytes: float = STATS_MAX_BYTES,
) -> xr.DataArray:
    """Return a model's bed level on its flood-depth grid, NaN on open sea.

    The grid is ``get_bed_level`` within ``max_bytes``; open sea is where
    ``sea_mask_path`` reads 1.0. Pass the CORRECTED sea/land classification
    (``zsini_sea_cells_on_grid.tif``, rule modelled_depth_estimation) so
    weir-protected fringe cells count as land. Its non-null pixels are the land
    domain of every flood statistic. Rule land_bed_level writes it once per model.
    """
    catalog = DataCatalog()
    # get_bed_level reads an oversized raster block-averaged from disk;
    # _coarsen_for_memory is a no-op safety net.
    da_dep = get_bed_level(catalog, sfincs_root, max_bytes=max_bytes)
    da_dep = _coarsen_for_memory(da_dep, max_bytes=max_bytes)
    da_sea = catalog.get_rasterdataset(str(sea_mask_path))
    # Eager: a dask-backed mask would make da_dep lazy, breaking the .item()
    # calls in 15_sanity_checks.py.
    da_sea_grid = da_sea.raster.reproject_like(da_dep, method="nearest").compute()
    return da_dep.where(da_sea_grid != 1.0).compute()


def write_land_bed_level(da_land_dep: xr.DataArray, path: str | Path) -> None:
    """Write ``land_bed_level`` output to netCDF, zlib-compressed.

    netCDF keeps the x/y coordinates exactly; a GeoTIFF stores only the affine
    transform, and on a rotated grid the coordinates rebuilt from it differ in
    the last bits, which shifts pixel areas.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    da = da_land_dep.rename("land_bed_level")
    da.encoding = {}
    # The 2-D coordinates of a rotated grid are most of the file; shuffle lets
    # zlib shrink them about tenfold.
    zlib = {"zlib": True, "complevel": 4, "shuffle": True}
    encoding = {da.name: zlib, **{c: zlib for c in da.coords if da[c].ndim > 0}}
    da.to_netcdf(path, encoding=encoding)


def read_land_bed_level(path: str | Path) -> xr.DataArray:
    """Return a ``write_land_bed_level`` raster in memory."""
    with xr.open_dataarray(path) as da:
        return da.load()


def compute_max_inundation(
    run_dir: str | Path,
    da_land_dep: xr.DataArray,
    hmin: float = 0.05,
) -> tuple[xr.DataArray, xr.DataArray] | tuple[None, None]:
    """Return max inundation depth (zsmax − dep), subgrid-downscaled and land-masked.

    Uses ``hydromt_sfincs.utils.downscale_floodmap`` on ``da_land_dep``
    (``land_bed_level``), so open sea is NaN in the depth too and drops out of
    both the flooded count and the land-domain denominator.
    Returns (da_hmax, da_land_dep); both are None when ``zsmax`` is unavailable.
    """
    mod = load_sfincs_output(run_dir)
    if "zsmax" not in mod.output.data:
        return None, None

    da_zsmax = mod.output.data["zsmax"]
    if "timemax" in da_zsmax.dims:
        da_zsmax = da_zsmax.max(dim="timemax")

    da_hmax = sfincs_utils.downscale_floodmap(
        zsmax=da_zsmax, dep=da_land_dep, hmin=hmin
    )
    return da_hmax.compute(), da_land_dep


def inundation_stats(da_hmax: xr.DataArray, da_dep: xr.DataArray) -> dict:
    """Count flooded and land pixels of compute_max_inundation's output.

    Returns n_land, n_flooded, frac (0.0 without land) and the areas
    flooded_km2 and land_km2 (rotation-aware pixel area).
    """
    n_land = int(da_dep.notnull().sum().item())
    n_flooded = int(da_hmax.notnull().sum().item())
    frac = n_flooded / n_land if n_land > 0 else 0.0

    res = abs(da_dep.raster.res[0] * da_dep.raster.res[1])  # rotation-aware
    return {
        "n_land": n_land,
        "n_flooded": n_flooded,
        "frac": frac,
        "flooded_km2": n_flooded * res / 1e6,
        "land_km2": n_land * res / 1e6,
    }


def compute_flood_progression(
    run_dir: str | Path,
    sea_mask_path: str | Path,
    variable: str = "depth",
) -> xr.DataArray | None:
    """Return an instantaneous time series for a flood-progression animation.

    ``variable="depth"``: ``h = max(zs - zb, 0)`` at cell resolution (one frame
    per ``dtmapout``, NO subgrid downscaling), masked to exclude open sea via
    ``sea_mask_path`` (pass ``zsini_sea_cells_on_grid.tif``, on the same coarse
    grid as ``zb``). ``variable="level"``: raw ``zs``, unmasked, since water
    level is meaningful over sea too.
    Returns None when ``zs`` is not in the output, or for depth when ``zb`` is missing.
    """
    if variable not in ("depth", "level"):
        raise ValueError(f"variable must be 'depth' or 'level', got {variable!r}")

    mod = load_sfincs_output(run_dir)
    if "zs" not in mod.output.data:
        return None
    da_zs_native = mod.output.data["zs"]

    if variable == "level":
        return da_zs_native.rename("zs")

    if "zb" not in mod.output.data:
        return None
    da_zb_native = mod.output.data["zb"].squeeze()
    da_h = (da_zs_native - da_zb_native).clip(min=0.0)
    da_h.name = "h"

    da_sea = mod.data_catalog.get_rasterdataset(str(sea_mask_path))
    da_sea_grid = da_sea.raster.reproject_like(da_h, method="nearest")
    water_mask = da_sea_grid == 1.0
    return da_h.where(~water_mask)


def _nearest_cell_index(da_cells: xr.DataArray, da_target: xr.DataArray) -> np.ndarray:
    """Return the flat index into ``da_cells`` of each ``da_target`` pixel (-1 outside).

    Runs the same ``reproject_like(method="nearest")`` that ``downscale_floodmap``
    applies to the water level, on a raster of cell indices instead, so indexing
    a frame with the result reproduces that reprojection.
    """
    da_index = da_cells.copy(
        data=np.arange(da_cells.size, dtype=np.int32).reshape(da_cells.shape)
    )
    # A float _FillValue (NaN in sfincs_map.nc) in the encoding would override
    # the integer nodata below.
    da_index.encoding = {}
    da_index.raster.set_nodata(-1)
    da_index = da_index.raster.reproject_like(da_target, method="nearest")
    return np.asarray(da_index.values)


def _downscale_depth(
    zs_cells: np.ndarray,
    zs_nodata: float | None,
    cell_index: np.ndarray,
    dep: np.ndarray,
) -> np.ndarray:
    """Return ``downscale_floodmap(zs, dep, hmin=0.0)`` values via ``_nearest_cell_index``."""
    zs_flat = np.asarray(zs_cells).ravel()
    # One extra NaN slot, so index -1 (outside the cell grid) reads NaN.
    zs_ext = np.empty(zs_flat.size + 1, dtype=zs_flat.dtype)
    zs_ext[:-1] = zs_flat
    zs_ext[-1] = np.nan
    if zs_nodata is not None and not np.isnan(zs_nodata):
        zs_ext[zs_ext == zs_nodata] = np.nan
    h = (zs_ext[cell_index] - dep).astype("float32")
    h[~(h > 0.0)] = np.nan
    return h


def compute_flood_timeseries_stats(
    run_dir: str | Path,
    da_land_dep: xr.DataArray,
    threshold_m: float,
) -> pd.DataFrame | None:
    """Return per-timestep flooded area (km^2) and flood volume (m^3), subgrid-downscaled.

    Feeds ``flood_timeseries.csv``. Each frame is downscaled onto ``da_land_dep``
    (``land_bed_level``) as ``downscale_floodmap`` does at ``hmin=0.0`` (NOT
    ``threshold_m``), so ``flood_volume_m3`` sums ALL positive depth;
    ``threshold_m`` only decides which pixels count towards ``flooded_area_km2``.
    Frames are processed one at a time. Returns None when ``zs`` is unavailable.
    """
    import pandas as pd

    mod = load_sfincs_output(run_dir)
    if "zs" not in mod.output.data:
        return None

    da_zs_native = mod.output.data["zs"]
    da_dep = da_land_dep

    res_x, res_y = da_dep.raster.res  # rotation-aware, unlike rio.resolution()
    pixel_area_m2 = abs(res_x * res_y)

    n_frames = da_zs_native.sizes["time"]
    times = da_zs_native["time"].values
    cell_index = _nearest_cell_index(da_zs_native.isel(time=0), da_dep)
    dep = np.asarray(da_dep.values)
    area_km2 = np.empty(n_frames)
    volume_m3 = np.empty(n_frames)
    for i in range(n_frames):
        da_zs_i = da_zs_native.isel(time=i)
        h = _downscale_depth(da_zs_i.values, da_zs_i.raster.nodata, cell_index, dep)
        # Open sea is NaN in dep, so h is NaN there and never counts.
        flooded = h > 0.0
        area_km2[i] = (
            np.count_nonzero(flooded & (h > threshold_m)) * pixel_area_m2 / 1e6
        )
        volume_m3[i] = (
            float(np.where(flooded, h, np.float32(0.0)).sum()) * pixel_area_m2
        )

    return pd.DataFrame(
        {"time": times, "flooded_area_km2": area_km2, "flood_volume_m3": volume_m3}
    )


def compute_risk_metrics(
    da_hmax: xr.DataArray,
    da_dep: xr.DataArray,
    landuse_path: str | Path,
    delta_polygon_path: str | Path,
    urban_code: int = 50,
    exclude_geom_path: str | Path | None = None,
) -> dict:
    """Return scalar flood-risk metrics from a downscaled, land-masked flood map.

    ``da_hmax``/``da_dep`` come from ``compute_max_inundation`` (areas are shares
    of the land domain). Column names match risk_metrics.csv as read by
    downstream adaptation measures (e.g. retreat).
    ``exclude_geom_path``: polygon(s) where da_hmax AND da_dep are set to NaN
    before any metric, so the zone drops out of numerator and denominator.
    Meant for a water_retention strategy's retention zone, where captured water
    is the measure working, not flood risk. None excludes nothing.
    """
    import rioxarray as rxr

    if exclude_geom_path is not None:
        from rasterio.features import rasterize

        excl_gdf = gpd.read_file(str(exclude_geom_path)).to_crs(da_dep.raster.crs)
        excl_mask = rasterize(
            [(geom, 1) for geom in excl_gdf.geometry],
            out_shape=da_dep.shape,
            transform=da_dep.raster.transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)
        da_hmax = da_hmax.where(~excl_mask)
        da_dep = da_dep.where(~excl_mask)

    res_x, res_y = da_dep.raster.res
    pixel_area_m2 = abs(res_x * res_y)
    land_area_km2 = int(da_dep.notnull().sum()) * pixel_area_m2 / 1e6

    da_lu = rxr.open_rasterio(str(landuse_path)).squeeze(drop=True)
    da_lu = da_lu.raster.reproject_like(da_dep, method="nearest")
    urban_mask = (da_lu == urban_code).values & da_dep.notnull().values
    urban_area_km2 = int(urban_mask.sum()) * pixel_area_m2 / 1e6

    flooded = da_hmax.notnull().values
    flooded_area_km2 = float(flooded.sum()) * pixel_area_m2 / 1e6
    urban_exposed_km2 = float((urban_mask & flooded).sum()) * pixel_area_m2 / 1e6
    volume_m3 = float((da_hmax.fillna(0.0) * pixel_area_m2).sum())

    delta = gpd.read_file(str(delta_polygon_path))
    model_domain_area_km2 = float(delta.to_crs(da_dep.raster.crs).area.sum()) / 1e6

    return {
        "model_domain_km2": round(model_domain_area_km2, 2),
        "land_area_km2": round(land_area_km2, 2),
        "urban_area_km2": round(urban_area_km2, 2),
        "flooded_area_km2": round(flooded_area_km2, 2),
        "flood_extent_pct": round(flooded_area_km2 / land_area_km2 * 100, 2),
        "urban_exposed_km2": round(urban_exposed_km2, 2),
        "mean_depth_m": round(float(da_hmax.mean()), 2),
        "max_depth_m": round(float(da_hmax.max()), 2),
        "volume_m3": round(volume_m3, 0),
        "volume_km3": round(volume_m3 / 1e9, 4),
    }


def compute_excess_volume(
    flood_map_path: str | Path,
    attribution_mask_path: str | Path,
    classes: tuple = (1, 3),
) -> float:
    """Return flood volume [m3] in the given attribution classes of an uncontrolled run.

    ``flood_map_path`` is max_flood_depth.tif of the UNCONTROLLED (no adaptation)
    baseline run. ``classes``: attribution mask classes (1=river, 2=coastal,
    3=compound); default river and compound. Called once per basin x scenario by
    rule attribution_mask (baseline_excess_volume.json), so every
    storage_fraction run of apply_water_retention is sized against the same
    fixed reference.
    """
    import rioxarray as rxr

    da = rxr.open_rasterio(str(flood_map_path), masked=True).squeeze(drop=True)
    da_attr = rxr.open_rasterio(str(attribution_mask_path)).squeeze(drop=True)

    res_x, res_y = da.raster.res
    pixel_area_m2 = abs(res_x * res_y)

    target_mask = da.notnull().values & np.isin(da_attr.values, list(classes))
    return float(da.values[target_mask].sum() * pixel_area_m2)
