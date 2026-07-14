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
import xugrid as xu

from hydromt_sfincs import SfincsModel
from hydromt_sfincs import utils as sfincs_utils

# Copernicus LC100 land-use codes that represent water bodies (see _LC_NAMES
# in src.plots): 80 = "Inland water", 200 = "Sea".
WATER_LANDUSE_CODES: tuple[int, ...] = (0, 200)  # (80, 200)

# Memory budget for area/volume STATISTICS (compute_max_inundation,
# compute_flood_timeseries_stats) rather than _coarsen_for_memory's
# animation-oriented default (5e8, shared across all of a time series'
# frames at once). Stats are computed one frame at a time (see
# compute_flood_timeseries_stats), so this only ever has to bound a SINGLE
# rasterized frame -- can afford a more generous, more accurate resolution.
STATS_MAX_BYTES: float = 1.5e9


def load_sfincs_output(run_dir: str | Path) -> SfincsModel:
    """Load a SFINCS run's map output via SfincsModel (HydroMT-aware spatial dims)."""
    mod = SfincsModel(root=str(run_dir), mode="r")
    mod.output.read()
    return mod


def _mosaic_quadtree_dep_levels(
    level_paths: list[Path], max_bytes: float = 5e8
) -> xr.DataArray:
    """
    Mosaic per-refinement-level ``dep_subgrid_lev*.tif`` files (written by
    ``quadtree_subgrid.create(write_dep_tif=True)`` — one full-domain raster
    per quadtree refinement level, each NaN outside that level's mesh cells)
    into a single regular raster, reading every level pre-downsampled to a
    shared, memory-bounded target resolution via rasterio's own decimated
    read (``out_shape`` + ``Resampling.average``) rather than at native
    resolution.

    The finest quadtree level's native pixel size, applied across the WHOLE
    domain (not just its own actually-refined footprint), can be hundreds of
    millions of pixels for a big domain — multiple GiB per array — which
    used to blow past available memory while just BUILDING the mosaic
    (reprojecting every coarser level onto that huge native grid), well
    before anything downstream (nothing needs subgrid-level pixel detail —
    see ``_coarsen_for_memory``) actually used that resolution. Levels are
    still combined finest-first, so refined-zone detail still takes priority
    over the coarser base level where they overlap — just built at a bounded
    resolution from the start instead of coarsening a huge array after the
    fact.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds

    metas = []
    for p in level_paths:
        with rasterio.open(p) as src:
            metas.append(
                {
                    "path": p,
                    "res": abs(src.res[0]),
                    "crs": src.crs,
                    "bounds": src.bounds,
                    "height": src.height,
                    "width": src.width,
                    "dtype": src.dtypes[0],
                    "nodata": src.nodata,
                }
            )
    metas.sort(key=lambda m: m["res"])  # finest first

    finest = metas[0]
    itemsize = np.dtype(finest["dtype"]).itemsize
    total_bytes = finest["height"] * finest["width"] * itemsize
    if total_bytes > max_bytes:
        factor = int(np.ceil((total_bytes / max_bytes) ** 0.5))
        target_h = max(1, finest["height"] // factor)
        target_w = max(1, finest["width"] // factor)
    else:
        target_h, target_w = finest["height"], finest["width"]
    target_transform = from_bounds(*finest["bounds"], target_w, target_h)

    mosaic_arr = None
    for m in metas:
        with rasterio.open(m["path"]) as src:
            arr = src.read(
                1, out_shape=(target_h, target_w), resampling=Resampling.average
            ).astype("float64")
        if m["nodata"] is not None:
            arr = np.where(arr == m["nodata"], np.nan, arr)
        mosaic_arr = (
            arr
            if mosaic_arr is None
            else np.where(np.isnan(mosaic_arr), arr, mosaic_arr)
        )

    x_coords = target_transform.c + (np.arange(target_w) + 0.5) * target_transform.a
    y_coords = target_transform.f + (np.arange(target_h) + 0.5) * target_transform.e
    mosaic = xr.DataArray(
        mosaic_arr,
        dims=("y", "x"),
        coords={"y": y_coords, "x": x_coords},
        name="dep",
    )
    # inplace=True: .rio.write_crs()/.rio.write_transform() deep-copy the
    # WHOLE array by default even though they only touch metadata -- for an
    # array already sized right at the memory budget, chaining two
    # out-of-place calls needs multiple simultaneous full-size copies (hit
    # for basin 4267691: a 1.76 GiB array's deep copy inside write_transform
    # failed to allocate). inplace=True sets the metadata directly with no copy.
    mosaic.rio.write_crs(finest["crs"], inplace=True)
    mosaic.rio.write_transform(target_transform, inplace=True)
    return mosaic


def get_bed_level(
    mod: SfincsModel,
    sfincs_root: str | Path,
    include_subgrid: bool = True,
    max_bytes: float = 5e8,
) -> xr.DataArray | None:
    """
    Bed level (dep) used to convert water levels to inundation depths.

    Prefers the subgrid reference raster — the resolution SFINCS uses
    internally for subgrid runs — and falls back to the coarser model-grid
    ``zb`` written to the run output. For a regular grid this is the single
    ``subgrid/dep_subgrid.tif``; for a quadtree grid, ``write_dep_tif=True``
    instead writes one full-domain raster per refinement level
    (``dep_subgrid_lev0.tif``, ``lev1.tif``, ...), which are mosaicked here
    into one combined raster (bounded to ``max_bytes`` — see
    ``_mosaic_quadtree_dep_levels``). Returns None when neither is available.

    Per-level files are checked first: their presence reliably indicates the
    model currently on disk is quadtree, whereas a single ``dep_subgrid.tif``
    can be a stale leftover from an earlier regular-grid build of the same
    ``sfincs_root`` (rebuilding doesn't clear the subgrid directory) and
    would silently give the wrong bed level if trusted unconditionally.
    """
    subgrid_dir = Path(sfincs_root) / "subgrid"
    if include_subgrid:
        level_paths = sorted(subgrid_dir.glob("dep_subgrid_lev*.tif"))
        if level_paths:
            return _mosaic_quadtree_dep_levels(level_paths, max_bytes=max_bytes)
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
    enough on its own (for a big domain + fine subgrid/quadtree resolution,
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
    # da_ref arrives already fully materialized (eager, eager numpy-backed --
    # get_bed_level's underlying rioxarray/data_catalog reads are not
    # chunked). A plain (non-chunked) .coarsen().mean() on an array this
    # large needs to build temporary reduction bookkeeping arrays (skipna's
    # internal isnan mask, in particular) comparable in size to da_ref
    # itself, ON TOP OF da_ref already being resident -- for a big enough
    # native array this alone can exceed available memory even though the
    # coarsened OUTPUT is tiny (hit for basin 4267691: a 900 MiB mask
    # allocation failed with the ~7.5 GiB source array already in memory).
    # Chunking first makes the reduction dask-backed, so it's computed
    # chunk-by-chunk with small bounded per-chunk temporaries instead of one
    # array-sized allocation; .compute() at the end materializes only the
    # already-small coarsened result.
    da_chunked = da_ref.chunk({"y": 2000, "x": 2000})
    da_coarse = da_chunked.coarsen(x=factor, y=factor, boundary="trim").mean().compute()
    # coarsen().mean() does not reliably carry the "spatial_ref" CRS
    # coordinate through the reduction -- restore it explicitly. inplace=True
    # avoids write_crs's default full-array deep copy (see
    # _mosaic_quadtree_dep_levels for why that matters at this array size).
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

    da_dep = get_bed_level(mod, sfincs_root, include_subgrid, max_bytes=max_bytes)
    if da_dep is None:
        return None, None
    if isinstance(da_zsmax, xu.UgridDataArray):
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


def _ensure_ugrid_crs(da: xu.UgridDataArray, crs) -> xu.UgridDataArray:
    """
    Ensure a ``UgridDataArray``'s mesh carries CRS metadata, restoring it
    from the model's own CRS (``SfincsModel.crs``) if missing.

    Needed for mesh-native animation (``plots.animate_flood_progression``),
    which reprojects the land/river overlay layers onto the mesh's own
    native CRS via ``da.ugrid.grid.crs`` rather than reprojecting the mesh
    itself (xugrid has no simple ``.rio.reproject()`` equivalent for a mesh)
    — a missing CRS there would silently misalign the overlays instead of
    raising, so it's worth restoring defensively rather than assuming the
    mesh always carries it already.
    """
    if da.ugrid.grid.crs is None and crs is not None:
        da.ugrid.grid.set_crs(crs)
    return da


def compute_flood_progression(
    run_dir: str | Path,
    landuse_path: str | Path,
    water_landuse_codes: tuple[int, ...] = WATER_LANDUSE_CODES,
) -> xr.DataArray | xu.UgridDataArray | None:
    """
    Instantaneous land-surface inundation depth time series for an animation
    of flood progression.

    Loads the instantaneous water level ``zs`` (one frame per ``dtmapout``)
    and the static bed level ``zb`` from the run's map output and derives the
    depth ``h = max(zs - zb, 0)`` — both already at the resolution SFINCS
    itself wrote to the map output (cell/mesh resolution), with NO subgrid
    downscaling applied, since this is for animation only (see
    ``compute_max_inundation`` for the downscaled, subgrid-aware version used
    for area/volume statistics).

    For a REGULAR grid, ``h`` is additionally masked to exclude water bodies
    via the land-use raster reprojected onto the model grid (``landuse_path``
    required in that case). For a QUADTREE run, ``h`` is returned as a
    mesh-native ``UgridDataArray`` with NO land/water masking — plotting it
    directly (``plots.animate_flood_progression``'s mesh-native path) avoids
    ever rasterizing the mesh, and the land polygon overlay already shows the
    water boundary visually, so per-cell land-use sampling isn't worth the
    extra spatial-join step for what both approaches use only cosmetically.

    Returns None when ``zs`` (the full time-series map output, as opposed to
    just the ``zsmax`` envelope) is not present in the run output.
    """
    mod = load_sfincs_output(run_dir)
    if "zs" not in mod.output.data or "zb" not in mod.output.data:
        return None

    da_zs_native = mod.output.data["zs"]
    da_zb_native = mod.output.data["zb"].squeeze()
    da_h = (da_zs_native - da_zb_native).clip(min=0.0)
    da_h.name = "h"

    if isinstance(da_h, xu.UgridDataArray):
        return _ensure_ugrid_crs(da_h, mod.crs)

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
    for a REGULAR grid too, not just quadtree: SFINCS subgrid refines a
    regular grid's own coarse cells onto the fine subgrid pixel grid the
    same way it projects a quadtree mesh onto one, so both grid types need
    the real downscaling step to land on the correct (fine) resolution — a
    plain mesh-to-raster projection (``_rasterize_like``) is a no-op for a
    regular grid and would silently leave it at the coarse model resolution.

    Processes ONE timestep at a time: downscales that single frame onto the
    (already memory-bounded, see ``get_bed_level`` / ``_coarsen_for_memory``)
    subgrid reference grid, computes its area/volume, then discards it before
    moving to the next frame. This decouples the memory cost from the number
    of output timesteps entirely (previously, when this calculation shared
    ``compute_flood_progression``'s output, the WHOLE multi-frame time series
    had to be rasterized onto the reference grid at once, forcing a much
    coarser resolution to fit the same memory budget) — so ``max_bytes`` here
    only has to bound a SINGLE frame, and can afford a more generous, more
    accurate resolution (``STATS_MAX_BYTES``) than an animation's per-frame
    budget would.

    Returns None when ``zs`` or the bed level is unavailable.
    """
    import pandas as pd

    mod = load_sfincs_output(run_dir)
    if "zs" not in mod.output.data:
        return None

    da_zs_native = mod.output.data["zs"]

    da_dep = get_bed_level(mod, sfincs_root, include_subgrid, max_bytes=max_bytes)
    if da_dep is None:
        return None
    if isinstance(da_zs_native, xu.UgridDataArray):
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
