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
import rioxarray  # noqa: F401 -- registers the .rio accessor used below
import xarray as xr
import xugrid as xu

from hydromt_sfincs import SfincsModel
from hydromt_sfincs import utils as sfincs_utils

# Copernicus LC100 land-use codes that represent water bodies (see _LC_NAMES
# in src.plots): 80 = "Inland water", 200 = "Sea".
WATER_LANDUSE_CODES: tuple[int, ...] = (80, 200)  # (80, 200)


def load_sfincs_output(run_dir: str | Path) -> SfincsModel:
    """Load a SFINCS run's map output via SfincsModel (HydroMT-aware spatial dims)."""
    mod = SfincsModel(root=str(run_dir), mode="r")
    mod.output.read()
    return mod


def _mosaic_quadtree_dep_levels(level_paths: list[Path]) -> xr.DataArray:
    """
    Mosaic per-refinement-level ``dep_subgrid_lev*.tif`` files (written by
    ``quadtree_subgrid.create(write_dep_tif=True)`` — one full-domain raster
    per quadtree refinement level, each NaN outside that level's mesh cells)
    into a single regular raster at the finest level's resolution. Levels are
    combined finest-first via ``rioxarray`` (rather than ``rasterio.merge``,
    which rejects some of these per-level tifs as "upside down") so
    refined-zone detail takes priority over the coarser base level where they
    overlap.
    """
    das = [rioxarray.open_rasterio(p).squeeze("band", drop=True) for p in level_paths]
    das.sort(key=lambda da: abs(da.rio.resolution()[0]))  # finest first

    mosaic = das[0]
    for da in das[1:]:
        mosaic = mosaic.where(~mosaic.isnull(), da.rio.reproject_match(mosaic))
    mosaic.name = "dep"
    return mosaic


def get_bed_level(
    mod: SfincsModel,
    sfincs_root: str | Path,
    include_subgrid: bool = True,
) -> xr.DataArray | None:
    """
    Bed level (dep) used to convert water levels to inundation depths.

    Prefers the subgrid reference raster — the resolution SFINCS uses
    internally for subgrid runs — and falls back to the coarser model-grid
    ``zb`` written to the run output. For a regular grid this is the single
    ``subgrid/dep_subgrid.tif``; for a quadtree grid, ``write_dep_tif=True``
    instead writes one full-domain raster per refinement level
    (``dep_subgrid_lev0.tif``, ``lev1.tif``, ...), which are mosaicked here
    into one combined raster. Returns None when neither is available.

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
            return _mosaic_quadtree_dep_levels(level_paths)
    dep_subgrid_path = subgrid_dir / "dep_subgrid.tif"
    if include_subgrid and dep_subgrid_path.exists():
        return mod.data_catalog.get_rasterdataset(str(dep_subgrid_path))
    if "zb" in mod.output.data:
        return mod.output.data["zb"].squeeze()
    return None


def _rasterize_like(
    da: xr.DataArray | xu.UgridDataArray, da_ref: xr.DataArray
) -> xr.DataArray:
    """
    Project a quadtree mesh ``UgridDataArray`` onto the regular reference
    raster ``da_ref`` (no-op for a regular-grid ``xr.DataArray``).

    Mirrors ``hydromt_sfincs.utils.downscale_floodmap``'s internal
    zsmax-onto-dep projection: non-rotated grids use xugrid's
    ``ugrid.rasterize_like`` directly; rotated grids go through
    ``xu.CentroidLocatorRegridder`` since ``rasterize_like`` only supports
    north-up rasters.
    """
    if not isinstance(da, xu.UgridDataArray):
        return da
    if da_ref.raster.transform[1] == 0 and da_ref.raster.transform[3] == 0:
        result = da.ugrid.rasterize_like(da_ref)
    else:
        uda_ref = xu.UgridDataArray.from_structured2d(da_ref, "xc", "yc")
        regridder = xu.CentroidLocatorRegridder(source=da, target=uda_ref)
        regridded = regridder.regrid(da)
        result = da_ref.copy(data=regridded.values.reshape(da_ref.shape))
    # xugrid's rasterize_like/regridders don't carry da_ref's CRS onto the
    # result, leaving downstream .raster/.rio calls (e.g. reproject_like) to
    # fail with "CRS is invalid: None" — restore it explicitly.
    return result.rio.write_crs(da_ref.rio.crs)


def _coarsen_for_timeseries(
    da_ref: xr.DataArray, n_frames: int, max_bytes: float = 5e8
) -> xr.DataArray:
    """
    Coarsen a fine subgrid reference raster before rasterizing a full
    multi-frame quadtree mesh time series onto it via ``_rasterize_like``.

    ``get_bed_level``'s subgrid-resolution reference raster is fine for a
    single 2D field (e.g. ``compute_max_inundation``'s ``zsmax``), but
    rasterizing every frame of an animation time series (e.g. 289 frames) at
    that resolution can need many GiB. Animations don't need subgrid-level
    detail, so this keeps memory bounded by coarsening just enough to fit
    ``n_frames`` copies within ``max_bytes`` (default 500 MB), regardless of
    the model's actual subgrid resolution or frame count.
    """
    total_bytes = da_ref.size * n_frames * da_ref.dtype.itemsize
    if total_bytes <= max_bytes:
        return da_ref
    factor = int(np.ceil((total_bytes / max_bytes) ** 0.5))
    # coarsen().mean() does not reliably carry the "spatial_ref" CRS
    # coordinate through the reduction -- restore it explicitly.
    da_coarse = da_ref.coarsen(x=factor, y=factor, boundary="trim").mean()
    return da_coarse.rio.write_crs(da_ref.rio.crs)


def _n_frames(da: xu.UgridDataArray) -> int:
    """Number of non-mesh ("time") frames in a quadtree mesh time series."""
    n = 1
    for dim in set(da.dims) - set(da.ugrid.grid.dims):
        n *= da.sizes[dim]
    return n


def compute_max_inundation(
    run_dir: str | Path,
    sfincs_root: str | Path,
    landuse_path: str | Path,
    hmin: float = 0.0,
    include_subgrid: bool = True,
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
    sfincs_root: str | Path,
    landuse_path: str | Path,
    water_landuse_codes: tuple[int, ...] = WATER_LANDUSE_CODES,
    include_subgrid: bool = True,
) -> xr.DataArray | None:
    """
    Instantaneous land-surface inundation depth time series for an animation
    of flood progression.

    Loads the instantaneous water level ``zs`` (one frame per ``dtmapout``)
    and the static bed level ``zb`` from the run's map output, derives the
    depth ``h = max(zs - zb, 0)``, and masks out water bodies using the
    land-use raster reprojected onto the model grid — so only land-surface
    inundation remains.

    For a quadtree run, ``zs``/``zb`` are mesh-native ``UgridDataArray``s;
    both are projected onto the regular ``get_bed_level`` reference raster
    (via ``_rasterize_like``) before subtracting, so the result is always a
    plain raster ``xr.DataArray`` regardless of grid type.

    Returns None when ``zs`` (the full time-series map output, as opposed to
    just the ``zsmax`` envelope) is not present in the run output.
    """
    mod = load_sfincs_output(run_dir)
    if "zs" not in mod.output.data or "zb" not in mod.output.data:
        return None

    da_zs_native = mod.output.data["zs"]
    da_dep_ref = get_bed_level(mod, sfincs_root, include_subgrid)
    if isinstance(da_zs_native, xu.UgridDataArray) and da_dep_ref is not None:
        da_dep_ref = _coarsen_for_timeseries(da_dep_ref, _n_frames(da_zs_native))
    da_zs = _rasterize_like(da_zs_native, da_dep_ref)
    da_zb = _rasterize_like(mod.output.data["zb"].squeeze(), da_dep_ref)
    da_h = (da_zs - da_zb).clip(min=0.0)
    da_h.name = "h"

    da_lu = mod.data_catalog.get_rasterdataset(str(landuse_path))
    da_lu_grid = da_lu.raster.reproject_like(da_h, method="nearest")
    water_mask = da_lu_grid.isin(list(water_landuse_codes))

    return da_h.where(~water_mask)


def compute_velocity_timeseries(
    run_dir: str | Path,
    sfincs_root: str | Path,
    include_subgrid: bool = True,
) -> tuple[xr.DataArray, xr.DataArray] | None:
    """
    Instantaneous u/v velocity component time series from a SFINCS map output.

    Requires ``storevel = 1`` in the run's sfincs.inp (set automatically by
    rule 14 when ``velocity_animation.enabled: true``). Returns ``(da_u, da_v)``
    or None when no velocity data is found. For a quadtree run, both
    components are projected onto the ``get_bed_level`` reference raster
    (via ``_rasterize_like``) so the result is always a plain raster.
    """
    mod = load_sfincs_output(run_dir)
    u_var = next((v for v in ("u", "u1") if v in mod.output.data), None)
    v_var = next((v for v in ("v", "v1") if v in mod.output.data), None)
    if u_var is None or v_var is None:
        return None
    da_u_native = mod.output.data[u_var]
    da_dep_ref = get_bed_level(mod, sfincs_root, include_subgrid)
    if isinstance(da_u_native, xu.UgridDataArray) and da_dep_ref is not None:
        da_dep_ref = _coarsen_for_timeseries(da_dep_ref, _n_frames(da_u_native))
    da_u = _rasterize_like(da_u_native, da_dep_ref)
    da_v = _rasterize_like(mod.output.data[v_var], da_dep_ref)
    return da_u, da_v


_ANALYSES = {
    "max_inundation": compute_max_inundation,
    "flood_progression": compute_flood_progression,
}


def postprocess_sfincs_output(
    run_dir: str | Path,
    analysis: str,
    **kwargs,
):
    """
    Run the postprocessing required for a named SFINCS output analysis.

    Args:
        run_dir:  Path to the SFINCS run directory containing ``sfincs_map.nc``
                  / ``sfincs_his.nc`` (e.g. ``<sfincs_root>/spinup`` or a
                  scenario subdirectory).
        analysis: One of ``"max_inundation"`` or ``"flood_progression"``.
        **kwargs: Forwarded to the matching ``compute_*`` function — see their
                  docstrings for the required dataset paths, options, and
                  return values (``compute_max_inundation`` returns a
                  ``(da_hmax, da_dep)`` tuple, ``compute_flood_progression``
                  returns a single ``xr.DataArray``).

    Returns:
        The result of the matching ``compute_*`` function, ready to be passed
        to ``plot_max_inundation_map`` / ``animate_flood_progression`` in
        ``src.plots``.
    """
    try:
        compute_fn = _ANALYSES[analysis]
    except KeyError:
        raise ValueError(
            f"Unknown analysis {analysis!r}; expected one of {sorted(_ANALYSES)}"
        ) from None
    return compute_fn(run_dir, **kwargs)
