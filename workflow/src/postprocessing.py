"""
Postprocessing of SFINCS scenario run outputs (sfincs_map.nc / sfincs_his.nc)
for output analysis and visualisation.

Each ``compute_*`` function loads a run directory's map output via
``SfincsModel`` (so spatial dimensions are HydroMT-aware), derives the
quantity needed for one type of analysis, and returns an ``xr.DataArray``
ready to be handed to the matching ``plot_*`` / ``animate_*`` function in
``src.plots``.  ``postprocess_sfincs_output`` dispatches to these by keyword.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 -- registers the .rio accessor used below
import xarray as xr
import geopandas as gpd

from hydromt_sfincs import SfincsModel
from hydromt_sfincs import utils as sfincs_utils

# Copernicus LC100 land-use codes that represent water bodies (see _LC_NAMES
# in src.plots): 80 = "Inland water", 200 = "Sea". Deliberately not (80, 200)
# here -- set to values that match no real land-use class, so rivers/ocean
# stay unmasked and visible in these diagnostics.
WATER_LANDUSE_CODES: tuple[int, ...] = (0, 2000)

# Memory budget for area/volume STATISTICS (compute_max_inundation,
# compute_flood_timeseries_stats) rather than _coarsen_for_memory's
# animation-oriented default (5e8, shared across all of a time series'
# frames at once). Stats are computed one frame at a time (see
# compute_flood_timeseries_stats), so this only ever has to bound a SINGLE
# rasterized frame -- can afford a more generous, more accurate resolution.
STATS_MAX_BYTES: float = 1.5e9


def load_sfincs_output(run_dir: str | Path) -> SfincsModel:
    """Load a SFINCS run's map output via SfincsModel (HydroMT-aware spatial dims).

    Reads ONLY sfincs_map.nc, not sfincs_his.nc -- none of this module's
    compute_* functions use station/observation-point data (only zsmax/zs/zb
    from the map file), and reading the his file unconditionally breaks for a
    run with zero observation points (e.g. river depth calibration's own
    disposable model, src.river_depth_calibration): SFINCS still writes a
    zero-station sfincs_his.nc in that case, and hydromt_sfincs's his-file
    reader crashes indexing station 0 of an empty dimension.
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


def get_bed_level(
    mod: SfincsModel,
    sfincs_root: str | Path,
    include_subgrid: bool = True,
) -> xr.DataArray | None:
    """
    Bed level (dep) used to convert water levels to inundation depths.

    Prefers the subgrid reference raster (``subgrid/dep_subgrid.tif``) --
    the resolution SFINCS uses internally for subgrid runs -- and falls
    back to the coarser model-grid ``zb`` written to the run output.
    Returns None when neither is available.
    """
    subgrid_dir = Path(sfincs_root) / "subgrid"
    dep_subgrid_path = subgrid_dir / "dep_subgrid.tif"
    if include_subgrid and dep_subgrid_path.exists():
        return mod.data_catalog.get_rasterdataset(str(dep_subgrid_path))
    if "zb" in mod.output.data:
        return mod.output.data["zb"].squeeze()
    return None


def _coarsen_for_memory(da_ref: xr.DataArray, max_bytes: float = 5e8) -> xr.DataArray:
    """
    Coarsen a fine subgrid reference raster before using it as the
    ``hydromt_sfincs.utils.downscale_floodmap`` target grid.

    ``get_bed_level``'s subgrid-resolution reference raster can be large
    enough on its own (for a big domain + fine subgrid resolution,
    hundreds of millions of pixels — multiple GiB per array) to blow past
    available memory for a single 2D field such as ``compute_max_inundation``'s
    ``zsmax``. Neither of this module's stats computations need subgrid-level
    detail, so this keeps memory bounded by coarsening down to ``max_bytes``
    (default 500 MB) regardless of the model's actual subgrid resolution or
    domain size. A no-op (returns ``da_ref`` unchanged) whenever it already fits.
    """
    total_bytes = da_ref.size * da_ref.dtype.itemsize
    if total_bytes <= max_bytes:
        return da_ref
    factor = int(np.ceil((total_bytes / max_bytes) ** 0.5))
    # da_ref arrives already fully materialized (eager, numpy-backed --
    # get_bed_level's underlying rioxarray/data_catalog reads are not
    # chunked). A plain (non-chunked) .coarsen().mean() on an array this
    # large needs to build temporary reduction bookkeeping arrays (skipna's
    # internal isnan mask, in particular) comparable in size to da_ref
    # itself, ON TOP OF da_ref already being resident -- for a big enough
    # native array this alone can exceed available memory even though the
    # coarsened OUTPUT is tiny. Chunking first makes the reduction
    # dask-backed, so it's computed chunk-by-chunk with small bounded
    # per-chunk temporaries instead of one array-sized allocation;
    # .compute() at the end materializes only the already-small coarsened
    # result.
    da_chunked = da_ref.chunk({"y": 2000, "x": 2000})
    da_coarse = da_chunked.coarsen(x=factor, y=factor, boundary="trim").mean().compute()
    # coarsen().mean() does not reliably carry the "spatial_ref" CRS
    # coordinate through the reduction -- restore it explicitly. inplace=True
    # avoids write_crs's default full-array deep copy, which matters at this
    # array size.
    da_coarse.rio.write_crs(da_ref.rio.crs, inplace=True)
    return da_coarse


def compute_max_inundation(
    run_dir: str | Path,
    sfincs_root: str | Path,
    landuse_path: str | Path,
    hmin: float = 0.0,
    include_subgrid: bool = True,
    max_bytes: float = STATS_MAX_BYTES,
) -> tuple[xr.DataArray, xr.DataArray] | tuple[None, None]:
    """
    Max inundation depth (zsmax − dep) for a SFINCS run, downscaled to the
    (sub)grid resolution and masked to the land domain.

    Takes the max of ``zsmax`` over the ``timemax`` dimension, determines the
    bed level via ``get_bed_level`` (subgrid-aware), derives the flood depth
    via ``hydromt_sfincs.utils.downscale_floodmap``, and masks both the flood
    depth and the bed-level reference grid to pixels where the land-use raster
    is in ``WATER_LANDUSE_CODES`` (80 = "Inland water", 200 = "Sea") — so all
    water bodies are excluded from both the flooded count and the land-domain
    denominator.

    Returns:
        (da_hmax, da_dep): the downscaled, land-masked flood depth and the
        bed-level raster used as the land-domain reference grid — its non-null
        pixel count gives the total number of land-domain pixels, e.g. for a
        flooded-area fraction (see 15_sanity_checks.py).  Both are None when
        ``zsmax`` or the bed level is unavailable.
    """
    mod = load_sfincs_output(run_dir)
    if "zsmax" not in mod.output.data:
        return None, None

    da_zsmax = mod.output.data["zsmax"]
    if "timemax" in da_zsmax.dims:
        da_zsmax = da_zsmax.max(dim="timemax")

    da_dep = get_bed_level(mod, sfincs_root, include_subgrid)
    if da_dep is None:
        return None, None
    # Bound da_dep's memory -- a large enough full-extent subgrid reference
    # raster (get_bed_level's dep_subgrid.tif) can exceed max_bytes;
    # _coarsen_for_memory is already a no-op below its own threshold.
    da_dep = _coarsen_for_memory(da_dep, max_bytes=max_bytes)

    da_hmax = sfincs_utils.downscale_floodmap(zsmax=da_zsmax, dep=da_dep, hmin=hmin)

    da_lu = mod.data_catalog.get_rasterdataset(str(landuse_path))
    # Compute eagerly: da_lu is dask-backed, and a dask-backed mask would force
    # da_dep/da_hmax to become lazy too via .where() below — breaking the
    # .item() calls in 15_sanity_checks.py ("'item' is not yet a valid method
    # on dask arrays").
    da_lu_grid = da_lu.raster.reproject_like(da_dep, method="nearest").compute()
    water_mask = da_lu_grid.isin(list(WATER_LANDUSE_CODES))

    da_dep = da_dep.where(~water_mask).compute()
    da_hmax = da_hmax.where(~water_mask).compute()
    return da_hmax, da_dep


def compute_flood_progression(
    run_dir: str | Path,
    landuse_path: str | Path,
    water_landuse_codes: tuple[int, ...] = WATER_LANDUSE_CODES,
    variable: str = "depth",
) -> xr.DataArray | None:
    """
    Instantaneous time series for an animation of flood progression.

    ``variable="depth"`` (default): land-surface inundation depth
    ``h = max(zs - zb, 0)``, derived from the instantaneous water level
    ``zs`` (one frame per ``dtmapout``) and the static bed level ``zb`` from
    the run's map output — both already at the resolution SFINCS itself
    wrote to the map output (cell resolution), with NO subgrid downscaling
    applied, since this is for animation only (see ``compute_max_inundation``
    for the downscaled, subgrid-aware version used for area/volume
    statistics). ``h`` is additionally masked to exclude water bodies via
    the land-use raster reprojected onto the model grid.

    ``variable="level"``: the raw water level ``zs`` itself, returned
    unmasked over the whole grid -- unlike depth, water level is physically
    meaningful over open sea and inland water too (e.g. watching a surge
    propagate), so there's no water body to exclude.

    Returns None when ``zs`` (the full time-series map output, as opposed to
    just the ``zsmax`` envelope) is not present in the run output, or (for
    ``variable="depth"``) when the bed level ``zb`` is unavailable.
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

    da_lu = mod.data_catalog.get_rasterdataset(str(landuse_path))
    da_lu_grid = da_lu.raster.reproject_like(da_h, method="nearest")
    water_mask = da_lu_grid.isin(list(water_landuse_codes))
    return da_h.where(~water_mask)


def compute_flood_timeseries_stats(
    run_dir: str | Path,
    sfincs_root: str | Path,
    landuse_path: str | Path,
    threshold_m: float,
    water_landuse_codes: tuple[int, ...] = WATER_LANDUSE_CODES,
    include_subgrid: bool = True,
    max_bytes: float = STATS_MAX_BYTES,
) -> pd.DataFrame | None:
    """
    Per-timestep flooded area (km^2) and total flood volume (m^3), at the
    same subgrid-downscaled resolution ``compute_max_inundation`` uses
    (unlike ``compute_flood_progression``, which stays at mesh/cell
    resolution for animation only) — this is what feeds
    ``flood_timeseries.csv``.

    Each frame is downscaled via ``hydromt_sfincs.utils.downscale_floodmap``
    (the same function ``compute_max_inundation`` uses) with ``hmin=0.0`` —
    NOT ``threshold_m`` — so ``flood_volume_m3`` sums ALL positive water
    depth present in the domain at that instant, matching what a literal
    total flood volume means; ``threshold_m`` is applied afterwards, only to
    decide which pixels count towards ``flooded_area_km2``. Using
    ``downscale_floodmap`` (rather than a plain ``dep`` subtraction) matters
    because SFINCS subgrid refines the grid's own coarse cells onto the fine
    subgrid pixel grid, so the real downscaling step is needed to land on
    the correct (fine) resolution.

    Processes ONE timestep at a time: downscales that single frame onto the
    (already memory-bounded, see ``get_bed_level`` / ``_coarsen_for_memory``)
    subgrid reference grid, computes its area/volume, then discards it before
    moving to the next frame. This decouples the memory cost from the number
    of output timesteps, so ``max_bytes`` here only has to bound a SINGLE
    frame, and can afford a more generous, more accurate resolution
    (``STATS_MAX_BYTES``) than an animation's per-frame budget would.

    Returns None when ``zs`` or the bed level is unavailable.
    """
    import pandas as pd

    mod = load_sfincs_output(run_dir)
    if "zs" not in mod.output.data:
        return None

    da_zs_native = mod.output.data["zs"]

    da_dep = get_bed_level(mod, sfincs_root, include_subgrid)
    if da_dep is None:
        return None
    # See compute_max_inundation's identical fix: bound da_dep -- a large
    # full-extent subgrid reference raster needs this protection.
    da_dep = _coarsen_for_memory(da_dep, max_bytes=max_bytes)

    da_lu = mod.data_catalog.get_rasterdataset(str(landuse_path))
    da_lu_grid = da_lu.raster.reproject_like(da_dep, method="nearest").compute()
    water_mask = da_lu_grid.isin(list(water_landuse_codes))

    try:
        res_x, res_y = da_dep.rio.resolution()
        pixel_area_m2 = abs(res_x * res_y)
    except Exception:
        pixel_area_m2 = np.nan

    n_frames = da_zs_native.sizes["time"]
    times = da_zs_native["time"].values
    area_km2 = np.empty(n_frames)
    volume_m3 = np.empty(n_frames)
    for i in range(n_frames):
        da_h_i = (
            sfincs_utils.downscale_floodmap(
                zsmax=da_zs_native.isel(time=i), dep=da_dep, hmin=0.0
            )
            .where(~water_mask)
            .compute()
        )
        area_km2[i] = float((da_h_i > threshold_m).sum().item()) * pixel_area_m2 / 1e6
        volume_m3[i] = float(da_h_i.sum(skipna=True).item()) * pixel_area_m2

    return pd.DataFrame(
        {"time": times, "flooded_area_km2": area_km2, "flood_volume_m3": volume_m3}
    )


# NOTE: KL added
def compute_risk_metrics(
    da_hmax: xr.DataArray,
    da_dep: xr.DataArray,
    landuse_path: str | Path,
    delta_polygon_path: str | Path,
    urban_code: int = 50,
) -> dict:
    """
    Scalar flood-risk metrics from a downscaled flood map.

    ``da_hmax``/``da_dep`` as returned by ``compute_max_inundation`` (already
    land-masked, so areas are shares of the land domain). Column names match
    the legacy analyse.py risk_metrics.csv so downstream adaptation measures
    (e.g. retreat) can consume them unchanged.
    """
    import rioxarray as rxr

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
